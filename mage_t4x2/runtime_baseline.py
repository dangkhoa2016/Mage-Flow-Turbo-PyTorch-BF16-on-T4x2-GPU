"""qualified runtime rebaseline manifest authority (mage_t4x2/runtime_baseline.py).

Implements the qualified multistep block-integrity corrective cold-start runbook
(2026-09-18):

  * successor runtime authority file set (same candidate set as hardened, with the
    two qualified successor swaps: hardened baseline module -> qualified baseline module,
    hardened driver -> qualified driver)
  * authority/runtime-baseline.json build + self-consistency verification
  * qualification gating: the manifest may only be generated from qualified
    bytes (TDD chronology + new qualified tests + focused regression + full suite +
    compileall + source freeze + protected integrity + real attempt #2 replay),
    never by direct edit before qualification
  * no wallet-style hard-coded hashes; every hash is computed from current bytes
  * initial/intermediate/hardened baselines remain historical and immutable; this successor
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

from mage_t4x2.runtime_baseline_hardened import REQUIRED_RUNTIME_FILES as _HARDENED_FILES

# ---------------------------------------------------------------------------
# Successor runtime file set (single source of truth shared with the tests).
# Derived from the hardened set so the shared candidate set never drifts; only the
# two qualified successor entries differ.
# ---------------------------------------------------------------------------

_FINAL_SNAPSHOT_ROOTS: List[str] = []


def _derive_runtime_files() -> List[Dict[str, str]]:
    derived: List[Dict[str, str]] = []
    for entry in _HARDENED_FILES:
        rel = entry["path"]
        if rel == "mage_t4x2/runtime_baseline_hardened.py":
            derived.append(
                {
                    "path": "mage_t4x2/runtime_baseline.py",
                    "role": "qualified runtime baseline manifest build + verify",
                }
            )
            continue
        if rel == "scripts/dual_t4_runtime_qualification_hardened.py":
            derived.append(
                {
                    "path": "scripts/dual_t4_runtime_qualification.py",
                    "role": "qualified multistep block-integrity corrective dual-T4 routing authority driver",
                }
            )
            continue
        derived.append(entry)
    return derived


REQUIRED_RUNTIME_FILES: List[Dict[str, str]] = _derive_runtime_files()

MANIFEST_NAME = "QUALIFIED_RUNTIME_BASELINE"
MANIFEST_STATUS = "QUALIFIED_STATIC"
MANIFEST_REASON = (
    "new qualified corrective baseline for the successor source that refuses the "
    "genuine G1 multistep block-integrity false negative (global block-order "
    "derivation replaced by per-invocation multi-step segmentation); not "
    "byte-equivalent to the initial/intermediate/hardened baselines because reducer/acceptance "
    "were corrected"
)
HARDENED_NOTE = (
    "initial/intermediate/hardened baselines remain historical and are not the successor baseline"
)
MANIFEST_REL = "authority/runtime-baseline.json"

# Qualification gate required by runbook section 31: the qualified manifest must
# refuse to build unless every gate below is PASS.
_REQUIRED_QUALIFICATION = (
    "tdd_chronology_pass",
    "new_runtime_tests_pass",
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

def verify_runtime_baseline(
    project_root: Optional[Path] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Self-consistency check for authority/runtime-baseline.json.

    Verifies, strictly from current bytes: every required successor file
    exists, size matches, SHA-256 matches, no duplicate paths, manifest
    status == QUALIFIED_STATIC, and the reason explicitly describes a new
    qualified baseline. Never hard-codes a verdict.
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
    if "qualified" not in reason or "baseline" not in reason:
        errors.append("reason must explicitly describe a new qualified baseline")

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

def build_runtime_baseline(
    project_root: Optional[Path] = None,
    qualification_evidence: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate authority/runtime-baseline.json from qualified current bytes.

    Refuses to emit a manifest unless the full qualified qualification gate is
    satisfied (TDD chronology PASS, new qualified tests PASS, focused regression
    PASS, full suite PASS, compileall PASS, source freeze PASS, protected
    integrity PASS, real attempt #2 replay PASS). The manifest is then written
    and immediately self-verified; status reflects the real verification result.
    """
    root = _default_project_root(project_root)
    evidence = qualification_evidence or {}
    missing_evidence = [k for k in _REQUIRED_QUALIFICATION if not evidence.get(k)]
    if missing_evidence:
        raise RuntimeError(
            "RUNTIME_BASELINE_REFUSE_UNQUALIFIED: qualification evidence missing: "
            f"{missing_evidence}"
        )

    behavior = evidence.get("behavior_qualification") or {}
    files: List[Dict[str, Any]] = []
    for entry in REQUIRED_RUNTIME_FILES:
        rel = entry["path"]
        target = root / rel
        if not target.is_file():
            raise RuntimeError(f"RUNTIME_BASELINE_CANNOT_BUILD: file absent: {rel}")
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
        "historical_reference": {"hardened_note": HARDENED_NOTE},
    }

    manifest = _manifest_path_for(root, manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    return verify_runtime_baseline(root, str(manifest))


# ---------------------------------------------------------------------------
# source manifest builder (authority/qualified-authority-source-manifest.json)
# ---------------------------------------------------------------------------

def build_runtime_authority_source_manifest(
    project_root: Optional[Path] = None,
    notebook_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate authority/qualified-authority-source-manifest.json after qualification.

    Hashes/sizes for the qualified runtime baseline manifest, the qualified baseline
    module, the qualified driver, the bootstrap dependency modules, requirements
    lock, wheelhouse manifest + wheels, upstream provenance JSON, the qualified
    runtime authority files, the qualified regression test file, and the pristine
    qualified notebook. Returns None-with-no-write if a required source file is
    missing (closeout tooling reports the gap); otherwise writes and returns
    the verification record.
    """
    root = _default_project_root(project_root)
    required_sources = [
        "authority/runtime-baseline.json",
        "mage_t4x2/runtime_baseline.py",
        "mage_t4x2/bootstrap_dependencies.py",
        "mage_t4x2/upstream_bootstrap.py",
        "scripts/dual_t4_runtime_qualification.py",
        "requirements-bootstrap.lock",
        "vendor/bootstrap-wheelhouse-manifest.json",
        "vendor/mage_upstream/UPSTREAM_SOURCE_PROVENANCE.json",
        "tests/test_multistep_block_integrity_regression.py",
        "tests/fixtures/qualified-attempt2-g1-telemetry.jsonl",
        "tests/fixtures/qualified-attempt2-g1-fixture-provenance.json",
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
        "manifest_name": "RUNTIME_AUTHORITY_SOURCE_MANIFEST",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    out = root / "authority" / "qualified-authority-source-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "manifest_path": str(out), "files": len(files)}