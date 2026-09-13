"""initial clean bootstrap environment (mage_t4x2/bootstrap_environment.py).

Implements the initial bootstrap dependency authority policy (static runbook
2026-09-17, sections 20-29, 37, 50):

  * ALWAYS create a fresh project-local bootstrap site
  * ALWAYS provision the exact bootstrap lock from the local wheelhouse into it
    (offline pip semantics: --no-index / --find-links / --target, no network)
  * ALWAYS prepend that local site before Mage import
  * ALWAYS verify the bootstrap dependency module origin resolves inside that
    local site
  * a strict meta-path import guard makes global ``loguru`` unusable inside the
    masked clean-room subprocess proof

Even when a globally installed ``loguru==0.7.3`` exists the initial authority clean
bootstrap gate relies only on the masked proof; global package state remains
informational only.

CPU-only. Importing this module never initializes CUDA.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bootstrap_dependencies import (
    BootstrapDependencyError,
    BootstrapPin,
    load_bootstrap_requirements,
    load_wheelhouse_manifest,
    reject_network_commands,
)

BOOTSTRAP_SITE_REL = ".runtime/bootstrap-site"
STALE_BOOTSTRAP_SITE = "STALE_BOOTSTRAP_SITE"
FRESH_TARGET_MARKER = ".BOOTSTRAP_FRESH_TARGET_CREATED"


class BootstrapEnvironmentError(RuntimeError):
    """initial bootstrap environment failure (fail-closed)."""


# ---------------------------------------------------------------------------
# strict loguru import guard (runbook sections 25-26)
# ---------------------------------------------------------------------------

class BootstrapLoguruGuardFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that intercepts ``loguru`` imports.

    Resolution is only allowed from the approved local bootstrap paths; any
    other origin (global site-packages, user site, another project) raises
    ModuleNotFoundError. With no approved paths, ALL ``loguru`` resolution is
    blocked — this is the pre-provision mask.
    """

    def __init__(self, allowed_paths: Sequence[Any] = ()) -> None:
        self.allowed = [str(Path(p).resolve()) for p in allowed_paths if p]

    def _resolve_from_base(self, fullname: str, base: str) -> Optional[Any]:
        rel = fullname.replace(".", "/")
        base_path = Path(base)
        py_file = base_path / f"{rel}.py"
        pkg_dir = base_path / rel
        if str(py_file.parent).startswith(str(base_path)) and py_file.is_file():
            spec = importlib.machinery.ModuleSpec(
                fullname,
                importlib.machinery.SourceFileLoader(fullname, str(py_file)),
                origin=str(py_file),
            )
            spec.has_location = True
            return spec
        if str(pkg_dir.parent).startswith(str(base_path)) and (pkg_dir / "__init__.py").is_file():
            init_py = pkg_dir / "__init__.py"
            spec = importlib.machinery.ModuleSpec(
                fullname,
                importlib.machinery.SourceFileLoader(fullname, str(init_py)),
                origin=str(init_py),
                is_package=True,
            )
            spec.has_location = True
            spec.submodule_search_locations = [str(pkg_dir)]
            return spec
        return None

    def find_spec(self, fullname: str, path: Optional[List[str]] = None,
                  target: Any = None) -> Optional[Any]:
        if fullname != "loguru" and not fullname.startswith("loguru."):
            return None
        for base in self.allowed:
            if not str(Path(base)).startswith("/"):
                continue
            spec = self._resolve_from_base(fullname, base)
            if spec is not None:
                return spec
        raise ModuleNotFoundError(
            f"No module named {fullname!r}: loguru must resolve from the "
            "approved local bootstrap site, not global site-packages"
        )


def install_loguru_mask(allowed_paths: Sequence[Any] = ()) -> BootstrapLoguruGuardFinder:
    """Install the strict loguru import guard at sys.meta_path[0].

    Any already-imported ``loguru``/``loguru.*`` modules are evicted so the
    guard is effective. Returns the guard for later uninstall.
    """
    guard = BootstrapLoguruGuardFinder(allowed_paths)
    sys.meta_path.insert(0, guard)
    for name in [m for m in list(sys.modules) if m == "loguru" or m.startswith("loguru.")]:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return guard


def uninstall_loguru_mask(guard: Optional[BootstrapLoguruGuardFinder] = None) -> None:
    """Remove the guard from sys.meta_path and invalidate import caches."""
    if guard is not None and guard in sys.meta_path:
        sys.meta_path.remove(guard)
    importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# bootstrap input authority
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bootstrap_inputs(
    requirements_path: Any,
    wheelhouse_dir: Any,
    wheelhouse_manifest: Any,
    pins: Optional[Sequence[BootstrapPin]] = None,
) -> Dict[str, Any]:
    """Verify bootstrap lock + wheelhouse manifest + wheelhouse bytes fail-closed.

    hardened corrective (2026-09-18): the real manifest schema declares an
    ``artifacts`` list, not a ``wheels`` mapping.  The lock SHA-256 is compared
    against the manifest ``requirements_sha256``; every required artifact is
    validated against actual on-disk bytes (presence, filename /
    distribution / version, byte size, SHA-256, python-tag/platform tag); every
    pinned requirement must have a matching artifact.  Unexpected schema
    ambiguity is rejected.  Any integrity mismatch is appended to the
    authoritative ``errors`` and forces top-level ``status=FAIL`` so that
    ``prepare_bootstrap_environment`` raises before pip/provision is ever
    reached.  Never raises for an input problem (the caller decides).
    """
    req = Path(requirements_path)
    whl = Path(wheelhouse_dir)
    manifest_path = Path(wheelhouse_manifest)
    errors: List[str] = []
    if not req.is_file():
        errors.append(f"requirements bootstrap lock absent: {req}")
    if not whl.is_dir():
        errors.append(f"wheelhouse absent: {whl}")
    if not manifest_path.is_file():
        errors.append(f"wheelhouse manifest absent: {manifest_path}")

    lock_sha = _sha256_file(req) if req.is_file() else None
    manifest_sha = _sha256_file(manifest_path) if manifest_path.is_file() else None
    wheels = sorted(whl.glob("*.whl")) if whl.is_dir() else []

    resolved: List[BootstrapPin] = []
    if pins is not None:
        resolved = list(pins)
    elif req.is_file():
        try:
            resolved = load_bootstrap_requirements(str(req))
        except BootstrapDependencyError as exc:
            errors.append(f"lock parse failed: {exc}")

    requirements_lock_status = "FAIL"
    if req.is_file() and resolved:
        if all(p.version and "=" in p.raw_line for p in resolved):
            requirements_lock_status = "PASS"
        else:
            errors.append("bootstrap lock contains unpinned (non-exact) requirement")

    wheelhouse_integrity_status = "FAIL"
    if manifest_path.is_file():
        try:
            mdoc = load_wheelhouse_manifest(str(manifest_path))
        except Exception as exc:  # pragma: no cover - defensive
            mdoc = {}
            errors.append(f"wheelhouse manifest unreadable: {exc}")

        if not isinstance(mdoc, dict):
            errors.append("wheelhouse manifest root must be a JSON object")
        else:
            artifacts = mdoc.get("artifacts")
            if not isinstance(artifacts, list):
                errors.append("wheelhouse manifest must declare an 'artifacts' list")
            else:
                integrity_pass = True

                wheels_key = mdoc.get("wheels")
                if isinstance(wheels_key, dict):
                    aliases = set(wheels_key.keys())
                    artifact_names = {
                        a.get("filename") for a in artifacts if isinstance(a, dict)
                    }
                    if aliases != artifact_names:
                        errors.append(
                            "wheelhouse manifest schema ambiguity: 'wheels' and "
                            "'artifacts' disagree"
                        )
                        integrity_pass = False

                expected_req_sha = mdoc.get("requirements_sha256", None)
                if not isinstance(expected_req_sha, str) or not expected_req_sha:
                    errors.append("wheelhouse manifest missing requirements_sha256")
                    integrity_pass = False
                elif lock_sha is None:
                    errors.append("requirements bootstrap lock absent")
                    integrity_pass = False
                elif lock_sha != expected_req_sha:
                    errors.append(
                        f"manifest requirements_sha256 mismatch: expected "
                        f"{expected_req_sha}, lock is {lock_sha}"
                    )
                    integrity_pass = False

                artifact_map: Dict[Any, Any] = {}
                for idx, art in enumerate(artifacts):
                    if not isinstance(art, dict):
                        errors.append(f"manifest artifact #{idx} is not an object")
                        integrity_pass = False
                        continue
                    fname = art.get("filename")
                    size = art.get("size")
                    sha = art.get("sha256")
                    dist = art.get("distribution")
                    version = art.get("version")
                    required_fields_ok = (
                        isinstance(fname, str)
                        and isinstance(size, int)
                        and isinstance(sha, str)
                        and isinstance(dist, str)
                        and isinstance(version, str)
                    )
                    if not required_fields_ok:
                        errors.append(
                            f"manifest artifact #{idx} missing filename/size/"
                            "sha256/distribution/version"
                        )
                        integrity_pass = False
                        continue
                    artifact_map[(dist.lower(), version)] = fname

                    wheel_path = whl / fname
                    if not wheel_path.is_file():
                        errors.append(f"wheel missing from manifest: {fname}")
                        integrity_pass = False
                        continue

                    expected_prefix = f"{dist}-{version}-"
                    if not fname.endswith(".whl") or not fname.startswith(
                        expected_prefix
                    ):
                        errors.append(
                            "wheel filename does not match distribution/version: "
                            f"{fname}"
                        )
                        integrity_pass = False
                    else:
                        tag_part = fname[len(expected_prefix):-4]
                        if not any(
                            t.startswith("py3") or t.startswith("cp3")
                            for t in tag_part.split("-")
                        ):
                            errors.append(
                                f"wheel filename python-tag/platform mismatch: {fname}"
                            )
                            integrity_pass = False

                    actual_size = wheel_path.stat().st_size
                    if actual_size != size:
                        errors.append(
                            f"wheel size mismatch: {fname}: expected {size}, got "
                            f"{actual_size}"
                        )
                        integrity_pass = False
                    actual_sha = _sha256_file(wheel_path)
                    if actual_sha != sha:
                        errors.append(
                            f"wheel sha256 mismatch: {fname}: expected {sha}, got "
                            f"{actual_sha}"
                        )
                        integrity_pass = False

                for pin in resolved:
                    if (pin.distribution.lower(), pin.version) not in artifact_map:
                        errors.append(
                            f"no manifest artifact for pinned requirement {pin}"
                        )
                        integrity_pass = False

                if integrity_pass:
                    wheelhouse_integrity_status = "PASS"
    else:
        errors.append("wheelhouse manifest unusable")

    if not wheels:
        errors.append("wheelhouse contains no wheels")

    return {
        "status": "PASS" if not errors else "FAIL",
        "requirements_path": str(req),
        "lock_sha256": lock_sha,
        "manifest_sha256": manifest_sha,
        "wheel_sha256": _sha256_file(wheels[0]) if len(wheels) == 1 else None,
        "wheel_count": len(wheels),
        "wheels": [w.name for w in wheels],
        "pins": resolved,
        "requirements_lock_status": requirements_lock_status,
        "wheelhouse_integrity_status": wheelhouse_integrity_status,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# fresh local target policy (runbook section 23)
# ---------------------------------------------------------------------------

def create_fresh_bootstrap_site(target: Any) -> Dict[str, Any]:
    """Create a fresh, empty local bootstrap site.

    hardened corrective (2026-09-18): if the target existed before this invocation
    it is stale and MUST fail closed with STALE_BOOTSTRAP_SITE.  A freshness
    marker only proves the CURRENT invocation created the site; it is never a
    reusable authorization token.  No marker may authorize reuse, and no
    hidden delete-and-recreate behavior is allowed.
    """
    tgt = Path(target)
    if tgt.exists():
        raise BootstrapEnvironmentError(
            f"{STALE_BOOTSTRAP_SITE}: target already exists (stale or "
            f"pre-existing; markers never authorize reuse): {tgt}"
        )
    tgt.mkdir(parents=True)
    (tgt / FRESH_TARGET_MARKER).write_text("created\n", encoding="utf-8")
    return {"status": "PASS", "target": str(tgt), "marker": FRESH_TARGET_MARKER}


# ---------------------------------------------------------------------------
# offline provisioning (runbook sections 22, 24)
# ---------------------------------------------------------------------------

def provision_bootstrap_site(
    pins: Sequence[BootstrapPin],
    requirements_path: Any,
    wheelhouse_dir: Any,
    target: Any,
) -> Dict[str, Any]:
    """Provision the exact bootstrap lock from the local wheelhouse.

    Uses the same interpreter (sys.executable) with local-only pip semantics
    (--no-index / --find-links / --target / -r lock). The resulting installed
    version is verified against the pins and origin residency is checked.
    """
    req = Path(requirements_path).resolve()
    whl = Path(wheelhouse_dir).resolve()
    tgt = Path(target).resolve()
    if not tgt.is_dir():
        raise BootstrapEnvironmentError("BOOTSTRAP_PROVISION_TARGET_MISSING: target must exist (fresh site)")
    argv = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--disable-pip-version-check",
        "--no-input",
        "--find-links",
        str(whl),
        "--target",
        str(tgt),
        "-r",
        str(req),
    ]
    violations = reject_network_commands(" ".join(argv))
    if violations:
        raise BootstrapEnvironmentError(f"BOOTSTRAP_PROVISION_NETWORK_REJECTED: {violations}")

    proc = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise BootstrapEnvironmentError(
            f"BOOTSTRAP_PROVISION_PIP_FAILED: rc={proc.returncode} "
            f"{proc.stderr.strip()[:500]}"
        )

    origin = verify_bootstrap_origins(pins=pins, target=tgt)
    if origin["status"] != "PASS":
        verrs = "; ".join(origin.get("errors", []))
        raise BootstrapEnvironmentError(f"BOOTSTRAP_PROVISION_VERSION_MISMATCH: version verification mismatch: {verrs}")

    return {
        "status": "PASS",
        "target": str(tgt),
        "pip_argv": argv,
        "pip_returncode": proc.returncode,
        "installed_version": next(
            (e["version"] for e in origin.get("entries", []) if e["distribution"] == "loguru"),
            None,
        ),
    }


# ---------------------------------------------------------------------------
# local origin verification (runbook sections 24, 37)
# ---------------------------------------------------------------------------

def _normalize_dist(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def verify_bootstrap_origins(
    pins: Sequence[BootstrapPin],
    target: Any,
) -> Dict[str, Any]:
    """Verify each pinned dependency is installed at the exact version AND
    resident under the local target (never global / user site / another dir)."""
    tgt = Path(target).resolve()
    entries: List[Dict[str, Any]] = []
    errors: List[str] = []
    for pin in pins:
        dist = _normalize_dist(pin.distribution)
        meta_dir = (tgt / f"{pin.distribution}-{pin.version}.dist-info")
        if not meta_dir.is_dir():
            candidates = sorted(
                p for p in tgt.glob("*.dist-info")
                if p.name.lower().startswith(f"{dist}-")
            )
            if candidates:
                metadata_file = candidates[0] / "METADATA"
                observed = None
                if metadata_file.is_file():
                    for line in metadata_file.read_text(encoding="utf-8").splitlines():
                        if line.startswith("Version:"):
                            observed = line.split(":", 1)[1].strip()
                            break
                resident = False
                pkg = tgt / pin.distribution.lower()
                if (pkg / "__init__.py").is_file():
                    resident = True
                entries.append(
                    {
                        "distribution": pin.distribution,
                        "expected": pin.version,
                        "version": observed,
                        "met": observed == pin.version,
                        "resident": resident,
                        "dist_info": candidates[0].name,
                    }
                )
            else:
                errors.append(
                    f"no metadata for {pin.distribution}; target not provisioned"
                )
            continue
        metadata_file = meta_dir / "METADATA"
        observed = None
        if metadata_file.is_file():
            for line in metadata_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version:"):
                    observed = line.split(":", 1)[1].strip()
                    break
        pkg = tgt / pin.distribution.lower()
        residential = (pkg / "__init__.py").is_file()
        if not residential:
            pkg = tgt / _normalize_dist(pin.distribution).replace("-", "_")
            residential = (pkg / "__init__.py").is_file()
        entries.append(
            {
                "distribution": pin.distribution,
                "expected": pin.version,
                "version": observed,
                "met": observed == pin.version,
                "resident": residential,
                "dist_info": meta_dir.name,
            }
        )

    for e in entries:
        if not e["met"]:
            errors.append(
                f"{e['distribution']}: expected {e['expected']}, got {e['version']}"
            )
        if not e["resident"]:
            errors.append(f"{e['distribution']}: not resident under {tgt}")

    return {
        "status": "PASS" if not errors and entries else "FAIL",
        "target": str(tgt),
        "entries": entries,
        "errors": errors,
    }


def activate_bootstrap_site(target: Any) -> Dict[str, Any]:
    """Prepend the local site to sys.path, invalidate import caches, and evict
    bootstrap dependency names from sys.modules if they were not imported by
    project code (bootstrap deps are authoritative only from the local site)."""
    tgt = Path(target).resolve()
    evicted: List[str] = []
    if str(tgt) not in sys.path:
        sys.path.insert(0, str(tgt))
    elif sys.path and sys.path[0] != str(tgt):
        sys.path.remove(str(tgt))
        sys.path.insert(0, str(tgt))
    for name in [m for m in list(sys.modules) if m == "loguru" or m.startswith("loguru.")]:
        if sys.modules.get(name) is not None:
            evicted.append(name)
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return {"status": "PASS", "path_front": str(tgt), "evicted": evicted}


# ---------------------------------------------------------------------------
# orchestrated prepare (runbook section 31 order)
# ---------------------------------------------------------------------------

def prepare_bootstrap_environment(
    project_root: Any,
    requirements_path: Optional[Any] = None,
    wheelhouse_dir: Optional[Any] = None,
    wheelhouse_manifest: Optional[Any] = None,
    target: Optional[Any] = None,
    pins: Optional[Sequence[BootstrapPin]] = None,
    activate: bool = True,
) -> Dict[str, Any]:
    """Run the full initial local bootstrap path and emit the four initial gates.

    Order (runbook 31): verify lock + wheelhouse manifest -> assert fresh initial
    bootstrap target -> offline install exact lock to local target -> activate
    local target -> verify exact version + local origin. Any failure raises
    fail-closed; the returned dict carries the four initial machine gates.
    """
    root = Path(project_root)
    req = Path(requirements_path) if requirements_path else root / "requirements-bootstrap.lock"
    whl = Path(wheelhouse_dir) if wheelhouse_dir else root / "vendor" / "bootstrap-wheelhouse"
    manifest = (
        Path(wheelhouse_manifest)
        if wheelhouse_manifest
        else root / "vendor" / "bootstrap-wheelhouse-manifest.json"
    )
    tgt = Path(target) if target else root / BOOTSTRAP_SITE_REL

    inputs = verify_bootstrap_inputs(
        requirements_path=req, wheelhouse_dir=whl, wheelhouse_manifest=manifest, pins=pins
    )
    if inputs["status"] != "PASS":
        raise BootstrapEnvironmentError(f"BOOTSTRAP_BOOTSTRAP_INPUTS_FAIL: {inputs['errors']}")
    resolved = inputs["pins"]

    create_fresh_bootstrap_site(tgt)  # STALE_BOOTSTRAP_SITE if pre-existing

    provision = provision_bootstrap_site(
        pins=resolved, requirements_path=req, wheelhouse_dir=whl, target=tgt
    )

    if activate:
        activate_bootstrap_site(tgt)

    version_gate = verify_bootstrap_origins(pins=resolved, target=tgt)
    return {
        "status": "PASS" if version_gate["status"] == "PASS" else "FAIL",
        "BOOTSTRAP_REQUIREMENTS_LOCK": inputs.get("requirements_lock_status", "FAIL"),
        "BOOTSTRAP_WHEELHOUSE_INTEGRITY": inputs.get("wheelhouse_integrity_status", "FAIL"),
        "BOOTSTRAP_LOCAL_SITE_FRESH": "PASS",
        "BOOTSTRAP_LOCAL_PROVISION": "PASS" if provision["status"] == "PASS" else "FAIL",
        "BOOTSTRAP_LOCAL_VERSION_VERIFY": version_gate["status"],
        "BOOTSTRAP_LOCAL_ORIGIN_VERIFY": version_gate["status"],
        "target": str(tgt),
        "pins": [{"distribution": p.distribution, "version": p.version} for p in resolved],
        "lock_sha256": inputs["lock_sha256"],
        "manifest_sha256": inputs["manifest_sha256"],
        "wheel_sha256": inputs["wheel_sha256"],
    }


# ---------------------------------------------------------------------------
# clean-room masking proof (runbook sections 25-29, 50)
# ---------------------------------------------------------------------------

_CLEAN_ROOM_PROOF_SNIPPET = r'''
import importlib, json, pathlib, sys
sys.path.insert(0, {project!r})
import mage_t4x2
from mage_t4x2 import upstream_bootstrap as ub
from mage_t4x2.bootstrap_environment import (
    activate_bootstrap_site,
    create_fresh_bootstrap_site,
    install_loguru_mask,
    provision_bootstrap_site,
    uninstall_loguru_mask,
    verify_bootstrap_inputs,
    verify_bootstrap_origins,
)

PROJECT = pathlib.Path({project!r})
WORK = pathlib.Path({work!r})
LOCK = PROJECT / "requirements-bootstrap.lock"
WHEELHOUSE = PROJECT / "vendor" / "bootstrap-wheelhouse"
MANIFEST = PROJECT / "vendor" / "bootstrap-wheelhouse-manifest.json"
TARGET = WORK / "fresh-bootstrap-site"

out = {{}}

# global loguru presence is informational only
out["global_loguru_detected"] = None
try:
    import loguru  # noqa: F401 (informational only)
    out["global_loguru_detected"] = getattr(loguru, "__version__", "?")
except Exception:
    out["global_loguru_detected"] = None

# step 1-4: mask global loguru; pre-provision import must FAIL as expected
guard = install_loguru_mask(())
try:
    try:
        import loguru  # noqa: F401
        out["PRE_PROVISION_LOGURU_IMPORT"] = "UNEXPECTED_PASS"
    except ModuleNotFoundError:
        out["PRE_PROVISION_LOGURU_IMPORT"] = "FAIL_AS_EXPECTED"
finally:
    uninstall_loguru_mask(guard)

inputs = verify_bootstrap_inputs(LOCK, WHEELHOUSE, MANIFEST)
if inputs["status"] != "PASS":
    out["LOCAL_OFFLINE_PROVISION"] = "FAIL"
    out["input_errors"] = inputs["errors"]
    out["subprocess_returncode"] = 0
else:
    create_fresh_bootstrap_site(TARGET)
    prov = provision_bootstrap_site(inputs["pins"], LOCK, WHEELHOUSE, TARGET)
    out["LOCAL_OFFLINE_PROVISION"] = "PASS" if prov["status"] == "PASS" else "FAIL"
    out["pip_argv"] = prov.get("pip_argv", [])
    out["pip_returncode"] = prov.get("pip_returncode")

    guard2 = install_loguru_mask((TARGET,))
    try:
        activate_bootstrap_site(TARGET)
        import loguru as local_loguru
        out["POST_PROVISION_LOGURU_IMPORT"] = "PASS"
        out["POST_PROVISION_LOGURU_VERSION"] = getattr(local_loguru, "__version__", "?")
        out["POST_PROVISION_LOGURU_ORIGIN"] = getattr(local_loguru, "__file__", "?")
    except Exception as exc:
        out["POST_PROVISION_LOGURU_IMPORT"] = "FAIL"
        out["local_import_error"] = str(exc)

    vg = verify_bootstrap_origins(inputs["pins"], TARGET)
    out["LOCAL_BOOTSTRAP_VERSION_VERIFY"] = vg["status"]
    out["LOCAL_BOOTSTRAP_ORIGIN_VERIFY"] = vg["status"]
    out["local_gate_entries"] = vg.get("entries", [])

    try:
        mage_boot = ub.bootstrap_upstream_mage()
        import mage_flow
        out["VENDORED_MAGE_CLEAN_IMPORT"] = "PASS"
        out["MAGE_ORIGIN"] = pathlib.Path(getattr(mage_flow, "__file__", "?"))
        mrel = pathlib.Path(out["MAGE_ORIGIN"]).resolve()
        mrel = mrel.relative_to(PROJECT.resolve()).as_posix()
        out["MAGE_ORIGIN"] = mrel
        out["mage_provenance_present"] = bool(mage_boot)
    except Exception as exc:
        out["VENDORED_MAGE_CLEAN_IMPORT"] = "FAIL"
        out["mage_import_error"] = str(exc)[:800]

    origin = out.get("POST_PROVISION_LOGURU_ORIGIN") or ""
    local_target = str(TARGET.resolve())
    origin_resolved = str(pathlib.Path(origin).resolve()) if origin not in ("?", "") else ""
    out["GLOBAL_LOGURU_USED"] = "YES" if origin_resolved and not origin_resolved.startswith(local_target) else "NO"
    out["subprocess_returncode"] = 0

out["target"] = str(TARGET)
pathlib.Path(WORK / "proof.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
'''


def run_clean_bootstrap_proof(
    project_root: Any,
    work_root: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the clean-room global-loguru masking proof in a fresh subprocess.

    Global ``loguru`` is actually made unusable inside the subprocess by the
    strict import guard; provisioning happens from the local wheelhouse into a
    fresh temporary target; Mage is bootstrap-imported from vendor; and the
    loaded ``loguru`` is proven to remain the local-target copy.

    Returns a JSON-serializable dict with the machine facts of section 27/29.
    """
    root = Path(project_root).resolve()
    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="initial-clean-proof-", dir="/tmp/opencode"))
    else:
        work_root = Path(work_root)
        work_root.mkdir(parents=True, exist_ok=True)

    snippet = _CLEAN_ROOM_PROOF_SNIPPET.format(project=str(root), work=str(work_root))
    proc = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(root),
    )
    proof_path = work_root / "proof.json"
    proof: Dict[str, Any] = {}
    if proof_path.is_file():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    else:
        proof = {"error": "proof subprocess produced no proof.json"}
    proof["GLOBAL_LOGURU_MASK_ACTIVE"] = (
        "PASS" if proof.get("PRE_PROVISION_LOGURU_IMPORT") == "FAIL_AS_EXPECTED" else "FAIL"
    )
    proof["subprocess_executable"] = sys.executable
    proof["subprocess_isolation_flags"] = ["-I", "-c"]
    proof["subprocess_returncode"] = proc.returncode
    proof["stderr"] = proc.stderr.strip()[-2000:]

    inputs = verify_bootstrap_inputs(
        root / "requirements-bootstrap.lock",
        root / "vendor" / "bootstrap-wheelhouse",
        root / "vendor" / "bootstrap-wheelhouse-manifest.json",
    )
    proof.setdefault("wheelhouse_path", str(root / "vendor" / "bootstrap-wheelhouse"))
    proof.setdefault("lock_sha256", inputs.get("lock_sha256"))
    proof.setdefault("manifest_sha256", inputs.get("manifest_sha256"))
    proof.setdefault("wheel_sha256", inputs.get("wheel_sha256"))
    proof["recorded_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return proof
