"""R2D runtime rebaseline manifest authority (mage_t4x2/r2d_runtime_baseline.py).

Implements the R2D runtime rebaseline contract (static runbook 2026-09-17,
sections 13-19, 32, 41, 43, 59):

  * candidate runtime file set (runbook 8 + audited routing-path additions,
    plus the three R2D additions)
  * immutable pre-R2D snapshot comparison (candidate bytes frozen)
  * authority/r2d-runtime-baseline.json build + self-consistency verification
  * qualification gating: the manifest may only be generated from qualified
    bytes (behavior + full regression + source invariants), never by direct
    edit before qualification
  * no wallet-style hard-coded hashes; every hash is computed from current bytes

CPU-only. Importing this module never initializes CUDA.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Candidate runtime file set (28 pre-existing + 3 R2D additions).
# Order matters: this list is the single source of truth shared with the tests.
# ---------------------------------------------------------------------------

REQUIRED_RUNTIME_FILES: List[Dict[str, str]] = [
    {"path": "mage_t4x2/dual_device_forward.py", "role": "transformer cross-device and return bridge"},
    {"path": "mage_t4x2/device_bridges.py", "role": "VAE input bridge; caller/latent anchor"},
    {"path": "scripts/gpu_session.py", "role": "runtime session orchestration; load-once; cross-device config"},
    {"path": "mage_t4x2/evidence_reducers.py", "role": "GPU participation / transfer / block-integrity reducers"},
    {"path": "mage_t4x2/phase_acceptance.py", "role": "per-phase acceptance facts derived from evidence"},
    {"path": "mage_t4x2/stage_b_session.py", "role": "session state machine; load_once; G1/G2 runners"},
    {"path": "mage_t4x2/upstream_bootstrap.py", "role": "vendored mage_flow bootstrap provenance"},
    {"path": "mage_t4x2/sdpa_contract.py", "role": "SDPA freeze env/backend"},
    {"path": "mage_t4x2/constants.py", "role": "frozen project constants (devices, blocks, pins)"},
    {"path": "mage_t4x2/contracts.py", "role": "run contract model"},
    {"path": "mage_t4x2/bootstrap_dependencies.py", "role": "bootstrap dependency lock/wheelhouse provisioning"},
    {"path": "mage_t4x2/block_partition.py", "role": "block discovery feeding adapter + session (block integrity / cross-device)"},
    {"path": "mage_t4x2/device_plan.py", "role": "build_device_plan (device placement)"},
    {"path": "mage_t4x2/transformer_placement.py", "role": "pre/post block placement consistency (return placement)"},
    {"path": "mage_t4x2/spread_loader.py", "role": "L0 dual-T4 spread load (placement / load count)"},
    {"path": "mage_t4x2/memory_plan.py", "role": "split-block / memory plan derivation (split boundary / L0)"},
    {"path": "mage_t4x2/telemetry.py", "role": "telemetry recorder + parser (block-integrity evidence)"},
    {"path": "mage_t4x2/evidence.py", "role": "EvidenceRun write paths"},
    {"path": "mage_t4x2/dtype_audit.py", "role": "runtime dtype audit (acceptance)"},
    {"path": "mage_t4x2/environment.py", "role": "cuda inventory (G0 / device placement)"},
    {"path": "mage_t4x2/model_provenance.py", "role": "model identity / revision (bootstrap provenance)"},
    {"path": "mage_t4x2/model_inventory.py", "role": "component inventory (model-load)"},
    {"path": "mage_t4x2/image_validation.py", "role": "output validation (acceptance)"},
    {"path": "mage_t4x2/nvidia_bootstrap.py", "role": "NVIDIA libs preflight (G0)"},
    {"path": "mage_t4x2/source_pin.py", "role": "pinned source authority (upstream provenance)"},
    {"path": "mage_t4x2/hashing.py", "role": "acceptance hashing (replay policy)"},
    {"path": "mage_t4x2/dtype_policy.py", "role": "dtype policy (acceptance)"},
    {"path": "mage_t4x2/phase_backend.py", "role": "synthetic phase backend parity (acceptance / session)"},
    {"path": "mage_t4x2/r2d_runtime_baseline.py", "role": "R2D runtime baseline manifest build + verify"},
    {"path": "mage_t4x2/r2d_bootstrap_environment.py", "role": "R2D clean bootstrap environment"},
    {"path": "scripts/g2_routing_authority_r2d_fresh_kernel.py", "role": "R2D dual-T4 routing authority driver"},
]

MANIFEST_NAME = "R2D_RUNTIME_REBASELINE"
MANIFEST_STATUS = "QUALIFIED_STATIC"
MANIFEST_REASON = (
    "new R2D baseline for reconstructed runtime; not byte-equivalent to R2B"
)
R2B_NOTE = "historical hashes are not current baseline"
MANIFEST_REL = "authority/r2d-runtime-baseline.json"

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

def verify_r2d_runtime_baseline(
    project_root: Optional[Path] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Self-consistency check for authority/r2d-runtime-baseline.json.

    Verifies, strictly from current bytes: every required file exists, size
    matches, SHA-256 matches, no duplicate paths, no unexpected missing
    authority-critical module, manifest status == QUALIFIED_STATIC, and the
    reason explicitly describes a new R2D baseline.

    Returns a dict with ``status`` PASS/FAIL plus full detail. Never hard-codes
    a verdict.
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
    if "r2d" not in reason or "baseline" not in reason:
        errors.append("reason must explicitly describe a new R2D baseline")

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

    extras = sorted(manifest_set - {"MISSING"} - required_set) if "MISSING" not in manifest_set else []
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
        "extras": extras,
    }
    return verification


# ---------------------------------------------------------------------------
# manifest build (quality-gated)
# ---------------------------------------------------------------------------

def build_r2d_runtime_baseline(
    project_root: Optional[Path] = None,
    qualification_evidence: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate authority/r2d-runtime-baseline.json from qualified current bytes.

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
            "R2D_REFUSE_UNQUALIFIED: qualification evidence missing: "
            f"{missing_evidence}"
        )

    behavior = evidence.get("behavior_qualification") or {}
    files: List[Dict[str, Any]] = []
    for entry in REQUIRED_RUNTIME_FILES:
        rel = entry["path"]
        target = root / rel
        if not target.is_file():
            raise RuntimeError(f"R2D_CANNOT_BUILD: file absent: {rel}")
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
        "historical_reference": {"r2b_note": R2B_NOTE},
    }

    manifest = _manifest_path_for(root, manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    return verify_r2d_runtime_baseline(root, str(manifest))


# ---------------------------------------------------------------------------
# pre-R2D snapshot comparison (candidate bytes frozen)
# ---------------------------------------------------------------------------

def runtime_candidate_match_snapshot(
    snapshot_path: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare every pre-existing candidate runtime file's current bytes to the
    immutable pre-R2D snapshot.

    Used to prove R2D_RUNTIME_FILES_CHANGED_DURING_QUALIFICATION=NO.
    """
    root = _default_project_root(project_root)
    if snapshot_path is None:
        roots = sorted(
            root.glob(
                "evidence/static/g2-routing-authority-r2d-rebaseline-clean-bootstrap-*/pre-r2d-snapshot.json"
            )
        )
        if roots:
            snapshot_path = roots[-1]
    if snapshot_path is None or not Path(snapshot_path).is_file():
        return {"status": "FAIL", "errors": [f"snapshot absent: {snapshot_path}"]}

    doc = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    comparison: Dict[str, Any] = {}
