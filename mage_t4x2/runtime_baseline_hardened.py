"""hardened runtime baseline manifest authority (mage_t4x2/runtime_baseline_hardened.py).

Implements the hardened bootstrap fail-closed corrective baseline contract
(static runbook 2026-09-18):

  * successor runtime authority file set (same candidate set as initial, with
    the three hardened successor swaps: initial baseline module -> hardened baseline
    module, initial driver -> hardened driver, bootstrap environment kept in place
    with corrected fail-closed bytes)
  * authority/runtime-baseline-hardened.json build + self-consistency verification
  * qualification gating: the manifest may only be generated from qualified
    bytes (behavior + full regression + source invariants), never by direct
    edit before qualification
  * no wallet-style hard-coded hashes; every hash is computed from current bytes
  * initial/intermediate baselines remain historical and immutable; this successor baseline
    describes the corrected successor source

CPU-only. Importing this module never initializes CUDA.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mage_t4x2.runtime_baseline_initial import REQUIRED_RUNTIME_FILES as _INITIAL_FILES

# ---------------------------------------------------------------------------
# Successor runtime file set (single source of truth shared with the tests).
# Derived from the initial set so the shared candidate set never drifts; only the
# three hardened successor entries differ.
# ---------------------------------------------------------------------------

_HARDENED_SNAPSHOT_ROOTS: List[str] = []


def _derive_hardened_runtime_files() -> List[Dict[str, str]]:
    derived: List[Dict[str, str]] = []
    for entry in _INITIAL_FILES:
        rel = entry["path"]
        if rel == "mage_t4x2/runtime_baseline_initial.py":
            derived.append(
                {
                    "path": "mage_t4x2/runtime_baseline_hardened.py",
                    "role": "hardened runtime baseline manifest build + verify",
                }
            )
            continue
        if rel == "scripts/dual_t4_runtime_qualification_initial.py":
            derived.append(
                {
                    "path": "scripts/dual_t4_runtime_qualification_hardened.py",
                    "role": "hardened bootstrap fail-closed dual-T4 routing authority driver",
                }
            )
            continue
        if rel == "mage_t4x2/bootstrap_environment.py":
            derived.append(
                {
                    "path": "mage_t4x2/bootstrap_environment.py",
                    "role": "hardened fail-closed clean bootstrap environment (Defects A+B corrected)",
                }
            )
            continue
        derived.append(entry)
    return derived


REQUIRED_RUNTIME_FILES: List[Dict[str, str]] = _derive_hardened_runtime_files()

MANIFEST_NAME = "HARDENED_RUNTIME_BASELINE"
MANIFEST_STATUS = "QUALIFIED_STATIC"
MANIFEST_REASON = (
    "new hardened corrective baseline for the successor source; not byte-equivalent "
    "to the initial/intermediate baseline because Defects A+B were corrected fail-closed"
)
BOOTSTRAP_NOTE = "initial/intermediate baselines remain historical and are not the successor baseline"
MANIFEST_REL = "authority/runtime-baseline-hardened.json"

_REQUIRED_QUALIFICATION = (
    "behavior_qualification_pass",
    "full_regression_pass",
    "source_invariants_pass",
)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_project_root(project_root: Optional[Path] = None) -> Path:
    if project_root is not None:
        return Path(project_root)
    return Path(__file__).resolve().parent.parent


def _manifest_path_for(project_root: Path, manifest_path: Optional[str] = None) -> Path:
    if manifest_path is not None:
        p = Path(manifest_path)
        if not p.is_absolute():
            p = project_root / p
        return p
    return project_root / MANIFEST_REL


# ---------------------------------------------------------------------------
# manifest verification
# ---------------------------------------------------------------------------

def verify_hardened_runtime_baseline(
    project_root: Optional[Path] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Self-consistency check for authority/runtime-baseline-hardened.json.

    Verifies, strictly from current bytes: every required successor file
    exists, size matches, SHA-256 matches, no duplicate paths, manifest
    status == QUALIFIED_STATIC, and the reason explicitly describes a new
    hardened baseline. Never hard-codes a verdict.
    """
    root = _default_project_root(project_root)
    manifest = _manifest_path_for(root, manifest_path)
    errors: List[str] = []
    if not manifest.is_file():
        return {"status": "FAIL", "errors": [f"manifest absent: {manifest}"]}
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "FAIL", "errors": [f"manifest unreadable: {exc}"]}

    if doc.get("schema_version") != 1:
        errors.append("schema_version != 1")
    if doc.get("baseline_name") != MANIFEST_NAME:
        errors.append(f"baseline_name != {MANIFEST_NAME}")
    if doc.get("status") != MANIFEST_STATUS:
        errors.append(f"status != {MANIFEST_STATUS}")
    reason = (doc.get("reason") or "").lower()
    if "hardened" not in reason or "baseline" not in reason:
        errors.append("reason must explicitly describe a new hardened baseline")

    files = doc.get("files") or []
    paths = [str(f.get("path")) for f in files]
    dupes = {p for p in paths if paths.count(p) > 1}
    if dupes:
        errors.append(f"duplicate paths: {sorted(dupes)}")

    required_set = {e["path"] for e in REQUIRED_RUNTIME_FILES}
    manifest_set = set(paths)
    missing = sorted(required_set - manifest_set)
    if missing:
        errors.append(f"missing required files: {missing}")

    entries: List[Dict[str, Any]] = []
    for f in files:
        rel = str(f.get("path"))
        target = root / rel
        rec: Dict[str, Any] = {
            "path": rel,
            "role": f.get("role"),
            "expected_size": f.get("size"),
            "expected_sha256": f.get("sha256"),
        }
        if not target.is_file():
            rec["status"] = "MISSING"
            entries.append(rec)
            errors.append(f"file absent: {rel}")
            continue
        actual_size = target.stat().st_size
        actual_sha = _sha256(target)
        rec["size"] = actual_size
        rec["sha256"] = actual_sha
        if actual_size != f.get("size"):
            rec["status"] = "SIZE_MISMATCH"
            errors.append(f"size mismatch: {rel}")
        elif actual_sha != f.get("sha256"):
            rec["status"] = "HASH_MISMATCH"
            errors.append(f"sha256 mismatch: {rel}")
        else:
            rec["status"] = "MATCH"
        entries.append(rec)

    verification: Dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "manifest_path": str(manifest),
        "manifest_entry": {
            "schema_version": doc.get("schema_version"),
            "baseline_name": doc.get("baseline_name"),
            "status": doc.get("status"),
            "created_utc": doc.get("created_utc"),
            "reason": doc.get("reason"),
            "behavior_qualification": doc.get("behavior_qualification"),
            "historical_reference": doc.get("historical_reference"),
        },
        "entries": entries,
        "errors": errors,
        "duplicates": sorted(dupes),
        "missing_required": missing,
    }
    return verification


# ---------------------------------------------------------------------------
# manifest build (quality-gated)
# ---------------------------------------------------------------------------

def build_hardened_runtime_baseline(
    project_root: Optional[Path] = None,
    qualification_evidence: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate authority/runtime-baseline-hardened.json from qualified current bytes.

    Refuses to emit a manifest unless the required qualification evidence is
    present (behavior qualification PASS, full regression PASS, source
    invariants PASS). The manifest is then written and immediately
    self-verified; status reflects the real verification result.
    """
    root = _default_project_root(project_root)
    evidence = qualification_evidence or {}
    missing_evidence = [k for k in _REQUIRED_QUALIFICATION if not evidence.get(k)]
    if missing_evidence:
        raise RuntimeError(
            "HARDENED_BASELINE_REFUSE_UNQUALIFIED: qualification evidence missing: "
            f"{missing_evidence}"
        )

    behavior = evidence.get("behavior_qualification") or {}
    files: List[Dict[str, Any]] = []
    for entry in REQUIRED_RUNTIME_FILES:
        rel = entry["path"]
        target = root / rel
        if not target.is_file():
            raise RuntimeError(f"HARDENED_BASELINE_CANNOT_BUILD: file absent: {rel}")
        files.append(
            {
                "path": rel,
                "size": target.stat().st_size,
                "sha256": _sha256(target),
                "role": entry["role"],
            }
        )

    doc = {
        "schema_version": 1,
        "baseline_name": MANIFEST_NAME,
        "status": MANIFEST_STATUS,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": MANIFEST_REASON,
        "behavior_qualification": {
            "status": behavior.get("status", "QUALIFIED"),
            "duration_ms": behavior.get("duration_ms"),
            "tests_passed": behavior.get("tests_passed"),
            "checks": behavior.get("checks"),
            "reason": evidence.get("reason") or behavior.get("reason"),
        },
        "files": files,
        "historical_reference": {"initial_note": BOOTSTRAP_NOTE},
    }

    manifest = _manifest_path_for(root, manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    return verify_hardened_runtime_baseline(root, str(manifest))


# ---------------------------------------------------------------------------
# source manifest builder (authority/hardened-authority-source-manifest.json)
# ---------------------------------------------------------------------------

def build_hardened_runtime_authority_source_manifest(
    project_root: Optional[Path] = None,
    notebook_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate authority/hardened-authority-source-manifest.json after qualification.

    Hashes/sizes for the hardened runtime baseline manifest, the corrected
    bootstrap environment, the hardened baseline module, the hardened driver, the
    bootstrap dependency modules, requirements lock, wheelhouse manifest +
    wheels, upstream provenance JSON, the hardened runtime authority files, and the
    pristine hardened notebook. Returns None-with-no-write if a required source file
    is missing (closeout tooling reports the gap); otherwise writes and returns
    the verification record.
    """
    root = _default_project_root(project_root)
