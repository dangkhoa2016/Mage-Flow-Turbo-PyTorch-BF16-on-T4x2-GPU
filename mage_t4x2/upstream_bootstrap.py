"""Pinned upstream ``mage_flow`` source bootstrap (R3 corrective).

Provisions the exact pinned Microsoft Mage ``mage_flow`` source from
``<PROJECT_ROOT>/vendor/mage_upstream`` into a fresh kernel, fail-closed:

  1. source root exists
  2. provenance manifest exists and parses
  3. repository URL exactly matches the authority value
  4. commit exactly matches the authority pin
  5. package-tree identity in the manifest matches the authority value
  6. every required runtime file exists
  7. every required runtime file matches its pinned Git blob SHA
     (local ``sha256`` from the manifest is verified when present)
  8. prepend the intended vendor source root to ``sys.path``
  9. invalidate import caches
  10. import ``mage_flow``
  11. prove ``mage_flow.__file__`` resolves inside the intended vendor root
  12. prove ``mage_flow.__version__ == "0.1.0"``
  13. prove the actual top-level public exports contract
  14. prove the deep runtime symbol contract
  15. return an immutable provenance result

A random installed ``site-packages/mage_flow`` is never accepted.  If
``mage_flow`` is already present in ``sys.modules`` from another location the
bootstrap fails closed (``UPSTREAM_PREIMPORT_CONTAMINATION``) instead of
silently reusing it.

This module does NOT import upstream ``mage_flow`` at module-import time.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- authority pin (runbook section 1/6; mirrors mage_t4x2.constants) -------
UPSTREAM_REPOSITORY = "https://github.com/microsoft/Mage"
UPSTREAM_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
UPSTREAM_MAGE_FLOW_TREE = "946b91bcb2cac75e6cfe8399f0f7f330a2280adf"
PACKAGE_NAME = "mage_flow"
PACKAGE_VERSION = "0.1.0"
MANIFEST_FILENAME = "UPSTREAM_SOURCE_PROVENANCE.json"

# --- named failure codes (runbook section 7) --------------------------------
UPSTREAM_SOURCE_MISSING = "UPSTREAM_SOURCE_MISSING"
UPSTREAM_PROVENANCE_MANIFEST_MISSING = "UPSTREAM_PROVENANCE_MANIFEST_MISSING"
UPSTREAM_PROVENANCE_MISMATCH = "UPSTREAM_PROVENANCE_MISMATCH"
UPSTREAM_RUNTIME_FILE_MISSING = "UPSTREAM_RUNTIME_FILE_MISSING"
UPSTREAM_RUNTIME_FILE_HASH_MISMATCH = "UPSTREAM_RUNTIME_FILE_HASH_MISMATCH"
UPSTREAM_PREIMPORT_CONTAMINATION = "UPSTREAM_PREIMPORT_CONTAMINATION"
UPSTREAM_IMPORT_FAILED = "UPSTREAM_IMPORT_FAILED"
UPSTREAM_MODULE_PATH_MISMATCH = "UPSTREAM_MODULE_PATH_MISMATCH"
UPSTREAM_VERSION_MISMATCH = "UPSTREAM_VERSION_MISMATCH"
UPSTREAM_SYMBOL_CONTRACT_MISMATCH = "UPSTREAM_SYMBOL_CONTRACT_MISMATCH"

_RUNTIME_FILES: List[str] = [
    "mage_flow/__init__.py",
    "mage_flow/pipeline.py",
    "mage_flow/models/__init__.py",
    "mage_flow/models/mage_flow.py",
    "mage_flow/models/utils.py",
    "mage_flow/models/modules/__init__.py",
    "mage_flow/models/modules/_attn_backend.py",
    "mage_flow/models/modules/mage_latent.py",
    "mage_flow/models/modules/mage_layers.py",
    "mage_flow/models/modules/mage_text.py",
    "mage_flow/models/modules/mage_vae.py",
    "mage_flow/models/modules/text_encoder.py",
]

# Actual pinned upstream top-level exports (not MageFlow / MageVAE).
_TOP_LEVEL_SYMBOLS = ("MageFlowPipeline", "ModelConfig", "load_from_repo")

# Deep runtime symbol contract.
_DEEP_SYMBOLS: Dict[str, List[str]] = {
    "mage_flow.models.mage_flow": ["MageFlow", "MageFlowModel", "ModelConfig"],
    "mage_flow.models.modules.mage_vae": ["MageVAE"],
}


@dataclass(frozen=True)
class UpstreamMageProvenance:
    """Immutable result of a successful upstream bootstrap."""

    source_root: str
    package_file: str
    upstream_repository: str
    upstream_commit: str
    package_version: str


class UpstreamMageBootstrapError(RuntimeError):
    """Fail-closed bootstrap failure carrying the named failure code."""

    def __init__(self, failure_code: str, message: str) -> None:
        self.failure_code = failure_code
        super().__init__(f"{failure_code}: {message}")


def git_blob_sha(data: bytes) -> str:
    """Compute the Git blob object SHA cheap and offline:
    SHA1("blob " + decimal_byte_length + NUL + raw_bytes)."""
    prefix = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + data).hexdigest()


def default_source_root() -> str:
    """``<PROJECT_ROOT>/vendor/mage_upstream``."""
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / "vendor" / "mage_upstream")


def _fail(code: str, message: str) -> "None":
    raise UpstreamMageBootstrapError(code, message)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        _fail(code, message)


def _read_manifest(source_root: Path) -> Dict[str, Any]:
    manifest_path = source_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        _fail(
            UPSTREAM_PROVENANCE_MANIFEST_MISSING,
            f"provenance manifest missing: {manifest_path}",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - classified below
        _fail(
            UPSTREAM_PROVENANCE_MISMATCH,
            f"provenance manifest unparsable: {manifest_path}: {exc}",
        )
    _require(
        isinstance(manifest, dict),
        UPSTREAM_PROVENANCE_MISMATCH,
        "provenance manifest root must be a JSON object",
    )
    return manifest


def verify_upstream_source(source_root: Optional[str] = None) -> Dict[str, Any]:
    """Static-gate provenance + file integrity verification (no import).

    Fail-closed: raises :class:`UpstreamMageBootstrapError` on any gate
    failure.  Returns a report dict on success.  Never imports ``mage_flow``.
    """
    root = Path(source_root or default_source_root()).resolve()

    _require(
        root.is_dir(),
        UPSTREAM_SOURCE_MISSING,
        f"upstream source root missing: {root}",
    )

    manifest = _read_manifest(root)

    _require(
        str(manifest.get("upstream_repository")) == UPSTREAM_REPOSITORY,
        UPSTREAM_PROVENANCE_MISMATCH,
        "repository URL does not match the authority pin",
    )
    _require(
        str(manifest.get("upstream_commit")) == UPSTREAM_COMMIT,
        UPSTREAM_PROVENANCE_MISMATCH,
        "upstream commit does not match the authority pin",
    )
    _require(
        str(manifest.get("upstream_mage_flow_tree")) == UPSTREAM_MAGE_FLOW_TREE,
        UPSTREAM_PROVENANCE_MISMATCH,
        "package-tree identity does not match the authority value",
    )

    files_entry = manifest.get("runtime_files")
    _require(
        isinstance(files_entry, dict),
        UPSTREAM_PROVENANCE_MISMATCH,
        "manifest 'runtime_files' must be an object",
    )
    _require(
        set(files_entry.keys()) == set(_RUNTIME_FILES),
        UPSTREAM_PROVENANCE_MISMATCH,
        "manifest runtime file set differs from the required runtime file set",
    )

    file_verdicts: Dict[str, Dict[str, Any]] = {}
    for rel in _RUNTIME_FILES:
        path = root / rel
        if not path.is_file():
            _fail(UPSTREAM_RUNTIME_FILE_MISSING, f"required runtime file missing: {path}")
        entry = files_entry[rel]
        expected_blob = entry.get("git_blob_sha")
        data = path.read_bytes()
        actual_blob = git_blob_sha(data)
        if not expected_blob or actual_blob != str(expected_blob):
            _fail(
                UPSTREAM_RUNTIME_FILE_HASH_MISMATCH,
                f"git blob SHA mismatch: {rel}: expected {expected_blob}, got {actual_blob}",
            )
        sha256_match = True
        expected_sha256 = entry.get("sha256")
        if expected_sha256:
            actual_sha256 = hashlib.sha256(data).hexdigest()
            sha256_match = str(expected_sha256) == actual_sha256
            if not sha256_match:
                _fail(
                    UPSTREAM_RUNTIME_FILE_HASH_MISMATCH,
                    f"sha256 mismatch: {rel}: expected {expected_sha256}, got {actual_sha256}",
                )
        file_verdicts[rel] = {
            "status": "PASS",
            "git_blob_sha": actual_blob,
            "sha256_match": sha256_match,
        }

    return {
        "status": "PASS",
        "source_root": str(root),
        "upstream_repository": str(manifest["upstream_repository"]),
        "upstream_commit": str(manifest["upstream_commit"]),
        "upstream_mage_flow_tree": str(manifest["upstream_mage_flow_tree"]),
        "package": str(manifest.get("package")),
        "package_version": str(manifest.get("package_version")),
        "manifest": manifest,
        "runtime_file_count": len(_RUNTIME_FILES),
        "runtime_file_hash_verdict": "PASS",
        "runtime_file_verdicts": file_verdicts,
    }


def _assert_symbols(module: Any, symbols: List[str], where: str) -> None:
    missing = [s for s in symbols if not hasattr(module, s)]
    if missing:
        _fail(
            UPSTREAM_SYMBOL_CONTRACT_MISMATCH,
            f"required symbol(s) missing in {where}: {missing}",
        )


def _existing_module_is_vendor(intended_root: Path) -> bool:
    module = sys.modules.get(PACKAGE_NAME)
    if module is None:
        return True
    path = getattr(module, "__file__", None)
    if path is None:
        return False
    try:
        return Path(path).resolve().is_relative_to(intended_root)
    except ValueError:
        return False


def _reuse_provenance(intended_root: Path) -> UpstreamMageProvenance:
    """Provenance for an already-imported vendor module (idempotent reuse)."""
    package_file = (intended_root / PACKAGE_NAME / "__init__.py").resolve()
    return UpstreamMageProvenance(
        source_root=str(intended_root),
        package_file=str(package_file),
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_commit=UPSTREAM_COMMIT,
        package_version=PACKAGE_VERSION,
    )


def bootstrap_upstream_mage(source_root: Optional[str] = None) -> UpstreamMageProvenance:
    """Provision the pinned upstream ``mage_flow`` from the vendor tree.

    Runs the full static gate sequence (1-7) through the import and symbol
    gates (8-14) and returns an immutable provenance record (15).  Fail-closed
    with the exact named failure codes from :data:`UPSTREAM_SOURCE_MISSING`
    through :data:`UPSTREAM_SYMBOL_CONTRACT_MISMATCH`.
    """
    verify_upstream_source(source_root)
    root = Path(source_root or default_source_root()).resolve()

    if PACKAGE_NAME in sys.modules and not _existing_module_is_vendor(root):
        existing = getattr(sys.modules[PACKAGE_NAME], "__file__", None)
        _fail(
            UPSTREAM_PREIMPORT_CONTAMINATION,
            f"{PACKAGE_NAME} already imported from outside the vendor root: {existing}",
        )

    vendor_path = str(root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    importlib.invalidate_caches()

    try:
        mage_flow = importlib.import_module(PACKAGE_NAME)
    except Exception as exc:  # noqa: BLE001 - classified as import failure
        _fail(UPSTREAM_IMPORT_FAILED, f"import mage_flow failed: {type(exc).__name__}: {exc}")

    package_file = (root / PACKAGE_NAME / "__init__.py").resolve()
    resolved_file = Path(getattr(mage_flow, "__file__", "") or "").resolve()
    _require(
        resolved_file == package_file,
        UPSTREAM_MODULE_PATH_MISMATCH,
        f"mage_flow.__file__ resolves to {resolved_file}, expected {package_file}",
    )

    _require(
        getattr(mage_flow, "__version__", None) == PACKAGE_VERSION,
        UPSTREAM_VERSION_MISMATCH,
        f"mage_flow.__version__ is {getattr(mage_flow, '__version__', None)!r}, "
        f"expected {PACKAGE_VERSION!r}",
    )

    _assert_symbols(mage_flow, list(_TOP_LEVEL_SYMBOLS), PACKAGE_NAME)

    for deep_module, symbols in _DEEP_SYMBOLS.items():
        try:
            module = importlib.import_module(deep_module)
        except Exception as exc:  # noqa: BLE001 - classified as symbol contract
            _fail(
                UPSTREAM_SYMBOL_CONTRACT_MISMATCH,
                f"deep module import failed for {deep_module}: {type(exc).__name__}: {exc}",
            )
        _assert_symbols(module, symbols, deep_module)

    return UpstreamMageProvenance(
        source_root=str(root),
        package_file=str(package_file),
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_commit=UPSTREAM_COMMIT,
        package_version=PACKAGE_VERSION,
    )