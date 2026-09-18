from __future__ import annotations

import json
from pathlib import Path

from mage_t4x2.evidence import EvidenceRun
from mage_t4x2.hashing import sha256_file
from public_demo.runner import finalize_public_evidence


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _passing_summary() -> dict:
    return {
        "status": "PASS",
        "run_id": "public-demo-finalization-test",
        "bootstrap_target": "bootstrap-site",
        "first_failed_gate": None,
        "error": None,
        "precision": {"status": "PASS"},
        "routing": {"observed_transformer_invocations": 4},
        "placement": {"text_encoder": "cuda:0", "vae": "cuda:1"},
    }


def _runner_tail(out_dir: Path, summary: dict, verdict_lines: list) -> None:
    verdict_lines.append(f"PUBLIC_DEMO_FINAL_VERDICT={summary['status']}")
    (out_dir / "runtime.log").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")


def _verify_manifest(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "manifest.json")
    assert isinstance(manifest, dict) and manifest, "manifest.json must list hashes"
    for name, digest in manifest.items():
        target = run_dir / name
        assert target.is_file(), f"manifest entry is absent: {name}"
        assert sha256_file(str(target)) == digest, f"manifest hash mismatch: {name}"


def test_public_runner_fails_if_acceptance_write_fails(tmp_path, monkeypatch):
    def boom(self, report):
        raise RuntimeError("acceptance write failed: disk full")

    monkeypatch.setattr(EvidenceRun, "write_acceptance", boom)

    summary = _passing_summary()
    verdict_lines: list = []
    result = finalize_public_evidence(tmp_path, summary, verdict_lines)
    _runner_tail(tmp_path, summary, verdict_lines)

    persisted = _read_json(tmp_path / "summary.json")

    assert result["status"] == "FAIL"
    assert summary["status"] == "FAIL"
    assert summary["first_failed_gate"] == "EVIDENCE_ACCEPTANCE_WRITE"
    assert "RuntimeError" in summary["error"]
    assert persisted["status"] == "FAIL"
    assert persisted["first_failed_gate"] == "EVIDENCE_ACCEPTANCE_WRITE"
    assert "RuntimeError" in persisted["error"]
    assert (
        "PUBLIC_DEMO_FINAL_VERDICT=FAIL"
        in (tmp_path / "runtime.log").read_text(encoding="utf-8")
    )
    assert "EVIDENCE_ACCEPTANCE_WRITE=FAIL" in "\n".join(verdict_lines)

    acceptance = tmp_path / "acceptance.json"
    if acceptance.is_file():
        assert _read_json(acceptance)["status"] != "PASS"


def test_public_runner_fails_if_manifest_build_fails(tmp_path, monkeypatch):
    def boom(directory, names):
        raise RuntimeError("manifest build failed: permission denied")

    monkeypatch.setattr("mage_t4x2.hashing.build_manifest", boom)

    summary = _passing_summary()
    verdict_lines: list = []
    result = finalize_public_evidence(tmp_path, summary, verdict_lines)
    _runner_tail(tmp_path, summary, verdict_lines)

    persisted_summary = _read_json(tmp_path / "summary.json")
    persisted_acceptance = _read_json(tmp_path / "acceptance.json")

    assert result["status"] == "FAIL"
    assert summary["status"] == "FAIL"
    assert summary["first_failed_gate"] == "EVIDENCE_MANIFEST_BUILD"
    assert "RuntimeError" in summary["error"]
    assert (tmp_path / "acceptance.json").is_file()
    assert persisted_summary["status"] == "FAIL"
    assert persisted_summary["first_failed_gate"] == "EVIDENCE_MANIFEST_BUILD"
    assert persisted_summary["error"] == summary["error"]
    assert persisted_acceptance["status"] == "FAIL"
    assert persisted_acceptance["first_failed_gate"] == "EVIDENCE_MANIFEST_BUILD"
    assert (
        "PUBLIC_DEMO_FINAL_VERDICT=FAIL"
        in (tmp_path / "runtime.log").read_text(encoding="utf-8")
    )
    assert "EVIDENCE_MANIFEST_BUILD=FAIL" in "\n".join(verdict_lines)


def test_public_runner_preserves_primary_failure_when_finalization_fails(tmp_path, monkeypatch):
    def boom(self, report):
        raise RuntimeError("acceptance write failed: disk full")

    monkeypatch.setattr(EvidenceRun, "write_acceptance", boom)

    summary = {
        "status": "FAIL",
        "run_id": "public-demo-finalization-test",
        "first_failed_gate": "ROUTING_CARDINALITY",
        "error": "ROUTING_CARDINALITY: observed transfer count mismatch",
    }
    verdict_lines: list = []
    finalize_public_evidence(tmp_path, summary, verdict_lines)

    persisted = _read_json(tmp_path / "summary.json")

    assert summary["status"] == "FAIL"
    assert summary["first_failed_gate"] == "ROUTING_CARDINALITY"
    assert summary["error"] == "ROUTING_CARDINALITY: observed transfer count mismatch"
    assert persisted["status"] == "FAIL"
    assert persisted["first_failed_gate"] == "ROUTING_CARDINALITY"
    assert persisted["error"] == "ROUTING_CARDINALITY: observed transfer count mismatch"

    acceptance = tmp_path / "acceptance.json"
    if acceptance.is_file():
        assert _read_json(acceptance)["status"] == "FAIL"
        assert _read_json(acceptance)["first_failed_gate"] == "ROUTING_CARDINALITY"


def test_public_runner_pass_is_preserved_when_evidence_finalization_succeeds(tmp_path):
    summary = _passing_summary()
    verdict_lines: list = []
    result = finalize_public_evidence(tmp_path, summary, verdict_lines)
    _runner_tail(tmp_path, summary, verdict_lines)

    summary_json = _read_json(tmp_path / "summary.json")
    acceptance_json = _read_json(tmp_path / "acceptance.json")

    assert result["status"] == "PASS"
    assert summary["status"] == "PASS"
    assert summary["first_failed_gate"] is None
    assert (tmp_path / "acceptance.json").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "manifest.sha256").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert summary_json["status"] == "PASS"
    assert acceptance_json["status"] == "PASS"
    assert (
        "PUBLIC_DEMO_FINAL_VERDICT=PASS"
        in (tmp_path / "runtime.log").read_text(encoding="utf-8")
    )


def test_public_runner_success_manifest_hashes_match_persisted_documents(tmp_path):
    summary = _passing_summary()
    verdict_lines: list = []
    finalize_public_evidence(tmp_path, summary, verdict_lines)

    for name, digest in _read_json(tmp_path / "manifest.json").items():
        assert name not in ("manifest.json", "manifest.sha256")
        target = tmp_path / name
        assert target.is_file()
        assert sha256_file(str(target)) == digest
    _verify_manifest(tmp_path)


def test_public_runner_persisted_evidence_agrees_across_all_surfaces_when_manifest_fails(
    tmp_path, monkeypatch
):
    def boom(directory, names):
        raise RuntimeError("manifest build failed: permission denied")

    monkeypatch.setattr("mage_t4x2.hashing.build_manifest", boom)

    summary = _passing_summary()
    verdict_lines: list = []
    result = finalize_public_evidence(tmp_path, summary, verdict_lines)
    _runner_tail(tmp_path, summary, verdict_lines)

    persisted_summary = _read_json(tmp_path / "summary.json")
    persisted_acceptance = _read_json(tmp_path / "acceptance.json")
    runtime_log = (tmp_path / "runtime.log").read_text(encoding="utf-8")

    assert result["status"] == "FAIL"
    assert summary["status"] == "FAIL"
    assert persisted_summary["status"] == "FAIL"
    assert persisted_acceptance["status"] == "FAIL"
    assert "PUBLIC_DEMO_FINAL_VERDICT=FAIL" in runtime_log
    assert persisted_summary["first_failed_gate"] == "EVIDENCE_MANIFEST_BUILD"
    assert persisted_acceptance["first_failed_gate"] == "EVIDENCE_MANIFEST_BUILD"
    assert not (tmp_path / "manifest.json").exists()