from pathlib import Path

import pytest

from public_demo.live_execution import (
    PublicLiveGateFailure,
    require_public_dual_t4_preflight,
)


def test_public_runner_uses_git_pypi_bootstrap_not_publication_archives():
    source = Path("public_demo/runner.py").read_text()
    assert "prepare_public_bootstrap" in source
    assert "publication ZIP" not in source
    assert "identity JSON" not in source
    assert "candidate-sdpa-fixed" not in source


def test_public_runner_signature_is_canonical_only():
    source = Path("public_demo/runner.py").read_text()
    signature = source.split("def run_public_demo(")[1].split(") -> Dict[str, Any]:")[0]
    assert "model_path" not in signature
    assert "prompt" not in signature
    assert "seed" not in signature
    assert "steps" not in signature
    assert "cfg_scale" not in signature
    assert "width" not in signature
    assert "height" not in signature
    assert "num_blocks" not in signature
    assert "split_block" not in signature


def test_public_runner_uses_canonical_contract_single_source_of_truth():
    source = Path("public_demo/runner.py").read_text()
    assert "from .contract import" in source
    assert "CANONICAL.validate()" in source
    assert "require_canonical_public_demo_inputs()" in source
    assert 'record_gate("CANONICAL_PUBLIC_CONTRACT"' in source


def test_public_runner_requires_owner_qualified_model_path():
    source = Path("public_demo/runner.py").read_text()
    assert "canonical_model = str(CANONICAL.model_path)" in source
    assert "require_public_demo_model_path" in source
    assert "require_public_demo_model_source" in source
    assert "fallback_id=None" in source
    assert "C.MODEL_ID.split" not in source


def test_public_runner_contract_fields_are_canonical_not_generic():
    source = Path("public_demo/runner.py").read_text()
    contract = source.split("contract = RunContract(")[1]
    assert "CANONICAL.model_id" in contract
    assert "CANONICAL.seed" in contract
    assert "CANONICAL.steps" in contract
    assert "CANONICAL.cfg_scale" in contract
    assert "CANONICAL.num_transformer_blocks" in contract
    assert "CANONICAL.split_block" in contract


def test_sdpa_guards_surround_model_execution():
    source = Path("public_demo/live_execution.py").read_text()
    load = source.index("load_runtime_state_dual_t4_spread")
    post = source.index("SDPA_POST_LOAD")
    placement = source.index("apply_dual_t4_mixed")
    pre = source.index("SDPA_PRE_INFERENCE")
    run = source.index("gs.run_t2i")
    assert load < post < placement < pre < run


def test_failed_runner_never_leaves_first_failed_gate_empty():
    source = Path("public_demo/runner.py").read_text()
    assert 'summary["first_failed_gate"] = "PUBLIC_BOOTSTRAP"' in source
    assert 'summary["first_failed_gate"] = "CANONICAL_PUBLIC_CONTRACT"' in source
    assert 'summary["first_failed_gate"] = "UNHANDLED_EXCEPTION"' in source


def test_runner_catches_precise_live_gate_failures():
    source = Path("public_demo/runner.py").read_text()
    assert "except PublicLiveGateFailure as exc:" in source
    assert 'summary["first_failed_gate"] = exc.gate' in source
    assert 'record_gate(exc.gate, False, str(exc))' in source


def test_runner_wraps_model_source_and_path_failures_as_model_path_gate():
    source = Path("public_demo/runner.py").read_text()
    assert "except PublicDemoContractError as exc:" in source
    assert 'raise PublicDemoGateFailure("MODEL_PATH", str(exc)) from exc' in source


def test_live_execution_uses_precise_public_gate_failures():
    source = Path("public_demo/live_execution.py").read_text()
    assert 'PublicLiveGateFailure("L0_MODEL_LOAD"' in source
    assert 'PublicLiveGateFailure("SDPA_POST_LOAD"' in source
    assert 'PublicLiveGateFailure("PLACEMENT"' in source
    assert 'PublicLiveGateFailure("SDPA_PRE_INFERENCE"' in source
    assert "require_public_dual_t4_preflight" in source
    assert "preflight_gpu(require_t4x2=True)" not in source


def test_public_runner_finalizes_evidence_fail_closed():
    source = Path("public_demo/runner.py").read_text()
    assert "def finalize_public_evidence(" in source
    assert "finalize_public_evidence(out_dir, summary, verdict_lines)" in source
    assert "EVIDENCE_ACCEPTANCE_WRITE" in source
    assert "EVIDENCE_MANIFEST_BUILD" in source


def test_public_runner_no_silent_success_on_evidence_finalization():
    source = Path("public_demo/runner.py").read_text()
    assert "write_acceptance(summary)" in source
    assert "build_manifest(str(out_dir), manifest_names())" in source
    assert source.count("except Exception:\n        pass") == 0


def test_public_runner_persists_corrected_status_after_finalization_failure():
    source = Path("public_demo/runner.py").read_text()
    assert "def persist_corrected_summary()" in source
    assert 'write_json_doc(str(out_dir / "summary.json"), summary)' in source
    assert "def reconcile_acceptance()" in source
    assert "if not acceptance_written" in source
    assert "if not manifest_built" in source


def test_public_runner_final_verdict_uses_corrected_status():
    source = Path("public_demo/runner.py").read_text()
    finalize_index = source.index("finalize_public_evidence(out_dir, summary, verdict_lines)")
    verdict_index = source.index("PUBLIC_DEMO_FINAL_VERDICT")
    log_index = source.index('"runtime.log").write_text')
    assert finalize_index < verdict_index < log_index


def test_public_hardware_preflight_false_result_is_g0_failure():
    calls = []
    gates = []

    def fake_preflight(*, require_t4x2):
        calls.append(require_t4x2)
        return {
            "device_count": 1,
            "inventory": [{"name": "Tesla T4"}],
            "t4x2_ok": False,
        }

    def record_gate(name, ok, detail):
        gates.append((name, ok, detail))

    with pytest.raises(PublicLiveGateFailure) as exc:
        require_public_dual_t4_preflight(fake_preflight, record_gate)

    assert exc.value.gate == "G0_HARDWARE"
    assert calls == [False]
    assert gates == [("G0_HARDWARE", False, "gpu_count=1")]


def test_public_hardware_preflight_exception_is_g0_failure():
    def fake_preflight(*, require_t4x2):
        assert require_t4x2 is False
        raise RuntimeError("NVIDIA_LIB_MISSING: libcuda not found")

    with pytest.raises(PublicLiveGateFailure) as exc:
        require_public_dual_t4_preflight(fake_preflight, lambda *_: None)

    assert exc.value.gate == "G0_HARDWARE"
    assert "NVIDIA_LIB_MISSING" in str(exc.value)


def test_public_hardware_preflight_success_records_pass():
    gates = []
    expected = {
        "device_count": 2,
        "inventory": [
            {"name": "Tesla T4"},
            {"name": "Tesla T4"},
        ],
        "t4x2_ok": True,
    }

    def fake_preflight(*, require_t4x2):
        assert require_t4x2 is False
        return expected

    result = require_public_dual_t4_preflight(
        fake_preflight,
        lambda name, ok, detail: gates.append((name, ok, detail)),
    )

    assert result is expected
    assert gates == [("G0_HARDWARE", True, "gpu_count=2")]