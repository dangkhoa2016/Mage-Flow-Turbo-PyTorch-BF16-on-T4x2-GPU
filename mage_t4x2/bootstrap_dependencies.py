"""provenance corrective: deterministic offline bootstrap dependency provisioning.

Converts implicit environmental assumptions about third-party packages into
explicit, pinned, hashed, offline-verifiable authority inputs.  The module
provides fail-closed functions for:

  1. loading and parsing the pinned bootstrap requirements lock
  2. loading and verifying the wheelhouse manifest
  3. verifying wheelhouse integrity (every wheel hash, every manifest entry)
  4. inspecting installed bootstrap dependency state
  5. planning provisioning (MISSING / PRESENT_EXACT / WRONG_VERSION / etc.)
  6. offline provisioning from local wheelhouse only (no network)
  7. post-provision exact version and import verification

All authority semantics are explicit.  The module never imports ``mage_flow``
at module scope and is CPU-safe.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Authority constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REQUIREMENTS_FILE = "requirements-bootstrap.lock"
DEFAULT_WHEELHOUSE_DIR = "vendor/bootstrap-wheelhouse"
DEFAULT_WHEELHOUSE_MANIFEST = "vendor/bootstrap-wheelhouse-manifest.json"
DEFAULT_BOOTSTRAP_SITE = ".runtime/bootstrap-site"

_NETWORK_URL_PATTERNS = (
    "https://pypi.org",
    "https://files.pythonhosted.org",
    "https://download.pytorch.org",
    "https://github.com",
)

# ---------------------------------------------------------------------------
# Named failure codes
# ---------------------------------------------------------------------------

REQUIREMENTS_FILE_MISSING = "BOOTSTRAP_REQUIREMENTS_FILE_MISSING"
REQUIREMENTS_PARSE_ERROR = "BOOTSTRAP_REQUIREMENTS_PARSE_ERROR"
REQUIREMENTS_UNPINNED_ENTRY = "BOOTSTRAP_REQUIREMENTS_UNPINNED_ENTRY"
REQUIREMENTS_DUPLICATE_ENTRY = "BOOTSTRAP_REQUIREMENTS_DUPLICATE_ENTRY"
REQUIREMENTS_RANGE_ENTRY = "BOOTSTRAP_REQUIREMENTS_RANGE_ENTRY"
WHEELHOUSE_MANIFEST_MISSING = "BOOTSTRAP_WHEELHOUSE_MANIFEST_MISSING"
WHEELHOUSE_MANIFEST_PARSE_ERROR = "BOOTSTRAP_WHEELHOUSE_MANIFEST_PARSE_ERROR"
WHEELHOUSE_MANIFEST_HASH_MISMATCH = "BOOTSTRAP_WHEELHOUSE_MANIFEST_HASH_MISMATCH"
WHEELHOUSE_WHEEL_MISSING = "BOOTSTRAP_WHEELHOUSE_WHEEL_MISSING"
WHEELHOUSE_WHEEL_HASH_MISMATCH = "BOOTSTRAP_WHEELHOUSE_WHEEL_HASH_MISMATCH"
WHEELHOUSE_UNMANIFESTED_ARTIFACT = "BOOTSTRAP_WHEELHOUSE_UNMANIFESTED_ARTIFACT"
BOOTSTRAP_SITE_CONTAMINATION = "BOOTSTRAP_SITE_CONTAMINATION"
BOOTSTRAP_PROVISION_FAILED = "BOOTSTRAP_PROVISION_FAILED"
BOOTSTRAP_POSTVERIFY_FAILED = "BOOTSTRAP_POST_VERIFY_FAILED"
WRONG_VERSION_ALREADY_LOADED = "BOOTSTRAP_WRONG_VERSION_ALREADY_LOADED"
NETWORK_PROVISIONING_BLOCKED = "BOOTSTRAP_NETWORK_PROVISIONING_BLOCKED"


class BootstrapDependencyError(RuntimeError):
    """Fail-closed bootstrap dependency failure."""

    def __init__(self, failure_code: str, message: str) -> None:
        self.failure_code = failure_code
        super().__init__(f"{failure_code}: {message}")


# ---------------------------------------------------------------------------
# Dependency state classification
# ---------------------------------------------------------------------------

PRESENT_EXACT = "PRESENT_EXACT"
MISSING = "MISSING"
WRONG_VERSION_NOT_LOADED = "WRONG_VERSION_NOT_LOADED"
WRONG_VERSION_ALREADY_LOADED = "WRONG_VERSION_ALREADY_LOADED"
UNVERIFIABLE = "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapPin:
    """One exact dependency pin from the bootstrap lock file."""

    distribution: str
    version: str
    raw_line: str

    def __str__(self) -> str:
        return f"{self.distribution}=={self.version}"


@dataclass(frozen=True)
class WheelArtifact:
    """One wheel artifact from the wheelhouse manifest."""

    filename: str
    size: int
    sha256: str
    distribution: str
    version: str


@dataclass(frozen=True)
class DependencyState:
    """Observed state of one bootstrap dependency."""

    distribution: str
    expected_version: str
    state: str
    installed_version: Optional[str] = None
    installed_location: Optional[str] = None
    importable: Optional[bool] = None
    resolves_from_bootstrap_site: Optional[bool] = None


# ---------------------------------------------------------------------------
# Requirements lock parsing
# ---------------------------------------------------------------------------


def _parse_requirements_line(line: str) -> Optional[BootstrapPin]:
    """Parse a single requirements line into a BootstrapPin.

    Accepts only exact pins: ``distribution==version``.
    Rejects ranges, wildcards, unpinned entries.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "==" not in line:
        return None
    parts = line.split("==", 1)
    dist = parts[0].strip()
    ver = parts[1].strip()
    if not dist or not ver:
        return None
    # Reject ranges / wildcards
    for forbidden in (">", "<", "!=", "~=", ">=", "<=", "*"):
        if forbidden in ver:
            return None
    return BootstrapPin(distribution=dist, version=ver, raw_line=line)


def load_bootstrap_requirements(
    requirements_path: Optional[str] = None,
) -> List[BootstrapPin]:
    """Load and validate the bootstrap requirements lock file.

    Fail-closed: raises on missing file, parse errors, unpinned entries,
    duplicates, or range specifiers.
    """
    if requirements_path is None:
        path = _PROJECT_ROOT / DEFAULT_REQUIREMENTS_FILE
    else:
        path = Path(requirements_path).resolve()

    if not path.is_file():
        raise BootstrapDependencyError(
            REQUIREMENTS_FILE_MISSING,
            f"bootstrap requirements file missing: {path}",
        )

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise BootstrapDependencyError(
            REQUIREMENTS_PARSE_ERROR,
            f"cannot read requirements file: {path}: {exc}",
        ) from exc

    pins: List[BootstrapPin] = []
    seen: Dict[str, int] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        pin = _parse_requirements_line(raw_line)
        if pin is None:
            # Check for unpinned (non-comment, non-empty, no ==)
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#"):
                if "==" not in stripped:
                    raise BootstrapDependencyError(
                        REQUIREMENTS_UNPINNED_ENTRY,
                        f"unpinned entry at line {lineno}: {stripped!r}",
                    )
                # Has == but rejected by parser (range etc.)
                raise BootstrapDependencyError(
                    REQUIREMENTS_RANGE_ENTRY,
                    f"range/pinned entry rejected at line {lineno}: {stripped!r}",
                )
            continue
        key = pin.distribution.lower()
        if key in seen:
            raise BootstrapDependencyError(
                REQUIREMENTS_DUPLICATE_ENTRY,
                f"duplicate distribution {pin.distribution!r} at line {lineno} "
                f"(first at line {seen[key]})",
            )
        seen[key] = lineno
        pins.append(pin)

    return pins


def requirements_sha256(requirements_path: Optional[str] = None) -> str:
    """Compute SHA-256 of the bootstrap requirements lock file."""
    if requirements_path is None:
        path = _PROJECT_ROOT / DEFAULT_REQUIREMENTS_FILE
    else:
        path = Path(requirements_path).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Wheelhouse manifest
# ---------------------------------------------------------------------------


def load_wheelhouse_manifest(
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the wheelhouse manifest JSON.  Fail-closed on parse errors."""
    if manifest_path is None:
        path = _PROJECT_ROOT / DEFAULT_WHEELHOUSE_MANIFEST
    else:
        path = Path(manifest_path).resolve()

    if not path.is_file():
        raise BootstrapDependencyError(
            WHEELHOUSE_MANIFEST_MISSING,
            f"wheelhouse manifest missing: {path}",
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BootstrapDependencyError(
            WHEELHOUSE_MANIFEST_PARSE_ERROR,
            f"wheelhouse manifest unparsable: {path}: {exc}",
        ) from exc

    if not isinstance(data, dict):
        raise BootstrapDependencyError(
            WHEELHOUSE_MANIFEST_PARSE_ERROR,
            "wheelhouse manifest root must be a JSON object",
        )

    return data


def verify_wheelhouse(
    requirements_pins: List[BootstrapPin],
    manifest_path: Optional[str] = None,
    wheelhouse_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify complete wheelhouse integrity.

    Checks:
      1. manifest requirements hash matches the lock file
      2. every manifest wheel file exists on disk
      3. every wheel file SHA-256 matches its manifest entry
      4. every wheel in the directory is in the manifest
      5. every required distribution/version has an artifact
    """
    if manifest_path is None:
        manifest = load_wheelhouse_manifest()
    else:
        manifest = load_wheelhouse_manifest(manifest_path)

    if wheelhouse_dir is None:
        whl_dir = _PROJECT_ROOT / DEFAULT_WHEELHOUSE_DIR
    else:
        whl_dir = Path(wheelhouse_dir).resolve()

    # 1. Requirements hash match
    actual_req_hash = requirements_sha256()
    expected_req_hash = manifest.get("requirements_sha256")
    req_hash_ok = actual_req_hash == expected_req_hash

    # 2 & 3. Verify each manifest wheel
    artifacts_raw = manifest.get("artifacts") or []
    artifacts: List[WheelArtifact] = []
    wheel_errors: List[str] = []
    for entry in artifacts_raw:
        art = WheelArtifact(
            filename=entry["filename"],
            size=entry["size"],
            sha256=entry["sha256"],
            distribution=entry["distribution"],
            version=entry["version"],
        )
        artifacts.append(art)
        wheel_path = whl_dir / art.filename
        if not wheel_path.is_file():
            wheel_errors.append(f"wheel missing: {art.filename}")
            continue
        actual_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        if actual_hash != art.sha256:
            wheel_errors.append(
                f"wheel hash mismatch: {art.filename}: "
                f"expected {art.sha256}, got {actual_hash}"
            )

    # 4. No unmanifested artifacts
    manifest_filenames = {a.filename for a in artifacts}
    unmanifested: List[str] = []
    if whl_dir.is_dir():
        for f in whl_dir.iterdir():
            if f.is_file() and f.name not in manifest_filenames:
                unmanifested.append(f.name)

    # 5. Every required pin has an artifact
    missing_pins: List[str] = []
    artifact_map = {(a.distribution.lower(), a.version): a for a in artifacts}
    for pin in requirements_pins:
        key = (pin.distribution.lower(), pin.version)
        if key not in artifact_map:
            missing_pins.append(str(pin))

    ok = bool(
        req_hash_ok
        and not wheel_errors
        and not unmanifested
        and not missing_pins
    )

    return {
        "status": "PASS" if ok else "FAIL",
        "requirements_hash_match": req_hash_ok,
        "expected_requirements_sha256": expected_req_hash,
        "actual_requirements_sha256": actual_req_hash,
        "wheel_count": len(artifacts),
        "wheel_errors": wheel_errors,
        "unmanifested_artifacts": unmanifested,
        "missing_pin_artifacts": missing_pins,
        "total_bytes": manifest.get("total_bytes", 0),
    }


# ---------------------------------------------------------------------------
# Dependency state inspection
# ---------------------------------------------------------------------------


def inspect_installed_bootstrap_dependencies(
    pins: List[BootstrapPin],
) -> List[DependencyState]:
    """Inspect the installed state of each pinned bootstrap dependency.

    Classifies each as PRESENT_EXACT, MISSING, WRONG_VERSION_NOT_LOADED,
    WRONG_VERSION_ALREADY_LOADED, or UNVERIFIABLE.
    """
    states: List[DependencyState] = []
    for pin in pins:
        try:
            installed_ver = importlib.metadata.version(pin.distribution)
        except importlib.metadata.PackageNotFoundError:
            states.append(
                DependencyState(
                    distribution=pin.distribution,
                    expected_version=pin.version,
                    state=MISSING,
                )
            )
            continue
        except Exception:
            states.append(
                DependencyState(
                    distribution=pin.distribution,
                    expected_version=pin.version,
                    state=UNVERIFIABLE,
                )
            )
            continue

        try:
            dist = importlib.metadata.distribution(pin.distribution)
            location = str(dist._path) if hasattr(dist, "_path") else None
        except Exception:
            location = None

        # Check if already loaded at wrong version
        already_loaded = False
        mod_name = pin.distribution.replace("-", "_").lower()
        for loaded_name, loaded_mod in list(sys.modules.items()):
            if loaded_name.replace("-", "_").lower() == mod_name:
                loaded_ver = getattr(loaded_mod, "__version__", None)
                if loaded_ver and str(loaded_ver) != pin.version:
                    already_loaded = True
                break

        if already_loaded:
            state = WRONG_VERSION_ALREADY_LOADED
        elif installed_ver == pin.version:
            state = PRESENT_EXACT
        else:
            state = WRONG_VERSION_NOT_LOADED

        states.append(
            DependencyState(
                distribution=pin.distribution,
                expected_version=pin.version,
                state=state,
                installed_version=installed_ver,
                installed_location=location,
            )
        )

    return states


# ---------------------------------------------------------------------------
# Provisioning plan
# ---------------------------------------------------------------------------


def plan_bootstrap_provisioning(
    states: List[DependencyState],
) -> Dict[str, Any]:
    """Determine what provisioning is needed based on dependency states.

    Returns a plan with:
      - needs_install: list of distributions needing installation
      - fail_closed: list of distributions that force FAIL_CLOSED
      - provisioning_needed: bool
    """
    needs_install: List[str] = []
    fail_closed: List[str] = []

    for ds in states:
        if ds.state == PRESENT_EXACT:
            continue
        elif ds.state == MISSING:
            needs_install.append(ds.distribution)
        elif ds.state == WRONG_VERSION_NOT_LOADED:
            needs_install.append(ds.distribution)
        elif ds.state == WRONG_VERSION_ALREADY_LOADED:
            fail_closed.append(ds.distribution)
        elif ds.state == UNVERIFIABLE:
            fail_closed.append(ds.distribution)

    return {
        "needs_install": needs_install,
        "fail_closed": fail_closed,
        "provisioning_needed": bool(needs_install),
        "provisioning_blocked": bool(fail_closed),
    }


# ---------------------------------------------------------------------------
# Offline provisioning
# ---------------------------------------------------------------------------


def _check_no_network_in_args(argv: List[str]) -> None:
    """Reject any command that could reach the network."""
    joined = " ".join(argv)
    for pattern in _NETWORK_URL_PATTERNS:
        if pattern in joined:
            raise BootstrapDependencyError(
                NETWORK_PROVISIONING_BLOCKED,
                f"network URL detected in provisioning command: {pattern}",
            )
    # Check for pip install without --no-index
    if "pip" in joined and "install" in joined and "--no-index" not in joined:
        raise BootstrapDependencyError(
            NETWORK_PROVISIONING_BLOCKED,
            "pip install without --no-index detected",
        )
