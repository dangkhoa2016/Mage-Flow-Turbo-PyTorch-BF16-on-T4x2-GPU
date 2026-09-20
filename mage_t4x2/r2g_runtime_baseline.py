"""R2G runtime rebaseline manifest authority (mage_t4x2/r2g_runtime_baseline.py).

Implements the R2G multistep block-integrity corrective cold-start runbook
(2026-09-18):

  * successor runtime authority file set (same candidate set as R2F, with the
    two R2G successor swaps: R2F baseline module -> R2G baseline module,
    R2F driver -> R2G driver)
  * authority/r2g-runtime-baseline.json build + self-consistency verification
  * qualification gating: the manifest may only be generated from qualified
    bytes (TDD chronology + new R2G tests + focused regression + full suite +
    compileall + source freeze + protected integrity + real attempt #2 replay),
    never by direct edit before qualification
  * no wallet-style hard-coded hashes; every hash is computed from current bytes
  * R2D/R2E/R2F baselines remain historical and immutable; this successor
    baseline describes the corrective successor source whose reducer/acceptance
    refused the genuine G1 multistep false-negative (root cause: global
    block-order derivation instead of per-invocation multi-step segmentation)

CPU-only. Importing this module never initializes CUDA.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mage_t4x2.r2f_runtime_baseline import REQUIRED_RUNTIME_FILES as _R2F_FILES

# ---------------------------------------------------------------------------
# Successor runtime file set (single source of truth shared with the tests).
# Derived from the R2F set so the shared candidate set never drifts; only the
# two R2G successor entries differ.
# ---------------------------------------------------------------------------

_R2G_SNAPSHOT_ROOTS: List[str] = []


def _derive_r2g_runtime_files() -> List[Dict[str, str]]:
    derived: List[Dict[str, str]] = []
    for entry in _R2F_FILES:
        rel = entry["path"]
        if rel == "mage_t4x2/r2f_runtime_baseline.py":
            derived.append(
                {
                    "path": "mage_t4x2/r2g_runtime_baseline.py",
                    "role": "R2G runtime baseline manifest build + verify",
                }
            )
            continue
        if rel == "scripts/g2_routing_authority_r2f_fresh_kernel.py":
            derived.append(
                {
                    "path": "scripts/g2_routing_authority_r2g_fresh_kernel.py",
                    "role": "R2G multistep block-integrity corrective dual-T4 routing authority driver",
                }
            )
            continue
        derived.append(entry)
    return derived


REQUIRED_RUNTIME_FILES: List[Dict[str, str]] = _derive_r2g_runtime_files()

MANIFEST_NAME = "R2G_RUNTIME_REBASELINE"
MANIFEST_STATUS = "QUALIFIED_STATIC"
MANIFEST_REASON = (
    "new R2G corrective baseline for the successor source that refuses the "
    "genuine G1 multistep block-integrity false negative (global block-order "
    "derivation replaced by per-invocation multi-step segmentation); not "
    "byte-equivalent to the R2D/R2E/R2F baselines because reducer/acceptance "
    "were corrected"
)
R2F_NOTE = (
    "R2D/R2E/R2F baselines remain historical and are not the successor baseline"
)
MANIFEST_REL = "authority/r2g-runtime-baseline.json"

# Qualification gate required by runbook section 31: the R2G manifest must
# refuse to build unless every gate below is PASS.
_REQUIRED_QUALIFICATION = (
    "tdd_chronology_pass",
    "new_r2g_tests_pass",
    "focused_regression_pass",
    "full_suite_pass",
    "compileall_pass",
    "source_freeze_pass",
    "protected_integrity_pass",
    "real_attempt2_replay_pass",
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

def verify_r2g_runtime_baseline(
    project_root: Optional[Path] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Self-consistency check for authority/r2g-runtime-baseline.json.

    Verifies, strictly from current bytes: every required successor file
    exists, size matches, SHA-256 matches, no duplicate paths, manifest
    status == QUALIFIED_STATIC, and the reason explicitly describes a new
    R2G baseline. Never hard-codes a verdict.
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
    if "r2g" not in reason or "baseline" not in reason:
        errors.append("reason must explicitly describe a new R2G baseline")

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
        "manifest_sha256": _sha256(manifest),
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

def build_r2g_runtime_baseline(
    project_root: Optional[Path] = None,
    qualification_evidence: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate authority/r2g-runtime-baseline.json from qualified current bytes.

    Refuses to emit a manifest unless the full R2G qualification gate is
    satisfied (TDD chronology PASS, new R2G tests PASS, focused regression
    PASS, full suite PASS, compileall PASS, source freeze PASS, protected
    integrity PASS, real attempt #2 replay PASS). The manifest is then written
    and immediately self-verified; status reflects the real verification result.
    """
    root = _default_project_root(project_root)
    evidence = qualification_evidence or {}
    missing_evidence = [k for k in _REQUIRED_QUALIFICATION if not evidence.get(k)]
    if missing_evidence:
        raise RuntimeError(
            "R2G_REFUSE_UNQUALIFIED: qualification evidence missing: "
            f"{missing_evidence}"
        )

    behavior = evidence.get("behavior_qualification") or {}
    files: List[Dict[str, Any]] = []
    for entry in REQUIRED_RUNTIME_FILES:
        rel = entry["path"]
        target = root / rel
        if not target.is_file():
            raise RuntimeError(f"R2G_CANNOT_BUILD: file absent: {rel}")
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
        "historical_reference": {"r2f_note": R2F_NOTE},
    }

    manifest = _manifest_path_for(root, manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    return verify_r2g_runtime_baseline(root, str(manifest))


# ---------------------------------------------------------------------------
# source manifest builder (authority/r2g-authority-source-manifest.json)
# ---------------------------------------------------------------------------

def build_r2g_authority_source_manifest(
    project_root: Optional[Path] = None,
    notebook_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate authority/r2g-authority-source-manifest.json after qualification.

    Hashes/sizes for the R2G runtime baseline manifest, the R2G baseline
    module, the R2G driver, the bootstrap dependency modules, requirements
    lock, wheelhouse manifest + wheels, upstream provenance JSON, the R2G
    runtime authority files, the R2G regression test file, and the pristine
    R2G notebook. Returns None-with-no-write if a required source file is
    missing (closeout tooling reports the gap); otherwise writes and returns
    the verification record.
    """
    root = _default_project_root(project_root)
    required_sources = [
        "authority/r2g-runtime-baseline.json",
        "mage_t4x2/r2g_runtime_baseline.py",
        "mage_t4x2/bootstrap_dependencies.py",
        "mage_t4x2/upstream_bootstrap.py",
        "scripts/g2_routing_authority_r2g_fresh_kernel.py",
        "requirements-bootstrap.lock",
        "vendor/bootstrap-wheelhouse-manifest.json",
        "vendor/mage_upstream/UPSTREAM_SOURCE_PROVENANCE.json",
        "tests/test_r2g_multistep_block_integrity_regression.py",
        "tests/fixtures/r2g-attempt2-g1-telemetry.jsonl",
        "tests/fixtures/r2g-attempt2-g1-fixture-provenance.json",
    ]
    if notebook_path:
        required_sources.append(notebook_path)

    missing = [r for r in required_sources if not (root / r).is_file()]
    wheelhouse = root / "vendor" / "bootstrap-wheelhouse"
    wheel_missing = not (wheelhouse.is_dir() and list(wheelhouse.glob("*.whl")))
    if missing or wheel_missing:
        return None

    files: List[Dict[str, Any]] = []
    for rel in required_sources:
        p = root / rel
        files.append({"path": rel, "size": p.stat().st_size, "sha256": _sha256(p)})
    for w in sorted(wheelhouse.glob("*.whl")):
        rel = w.relative_to(root).as_posix()
        files.append({"path": rel, "size": w.stat().st_size, "sha256": _sha256(w)})
    for entry in REQUIRED_RUNTIME_FILES:
        p = root / entry["path"]
        files.append(
            {
                "path": entry["path"],
                "size": p.stat().st_size,
                "sha256": _sha256(p),
                "role": entry["role"],
                "baseline": True,
            }
        )

    doc = {
        "schema_version": 1,
        "manifest_name": "R2G_AUTHORITY_SOURCE_MANIFEST",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    out = root / "authority" / "r2g-authority-source-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "manifest_path": str(out), "files": len(files)}
