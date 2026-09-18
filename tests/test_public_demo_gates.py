from pathlib import Path

import pytest

from public_demo.gates import (
    component_dtypes,
    count_vae_input_transfer_0_to_1,
    cpu_fallback_observed,
    derive_model_load_count,
    model_load_count_ok,
    placement_from_plan,
    routing_cardinality_ok,
    routing_integrity_ok,
)
from public_demo.live_execution import build_model_load_evidence_doc


def test_direction_specific_vae_count():
    records = [
        {"event": "vae_input_transfer", "from": "cuda:0", "to": "cuda:1"},
        {"event": "vae_input_transfer", "from": "cuda:1", "to": "cuda:0"},
        {"event": "vae_input_transfer", "from": "cpu", "to": "cuda:1"},
    ]
    assert count_vae_input_transfer_0_to_1(records) == 1


def test_model_load_count_is_derived_from_load_events():
    assert derive_model_load_count([]) == 0
    assert derive_model_load_count([object()]) == 1
    assert derive_model_load_count([object(), object()]) == 2


@pytest.mark.parametrize("count,ok", [(0, False), (1, True), (2, False)])
def test_model_load_gate_fails_closed_unless_exactly_one(count, ok):
    assert model_load_count_ok(count) is ok


def test_live_execution_derives_load_count_not_hard_codes():
    source = Path("public_demo/live_execution.py").read_text()
    assert '"model_load_count": 1' not in source
    assert "derive_model_load_count" in source
    assert "load_events" in source
    assert "L0_LOAD_ONCE" in source


def test_build_model_load_evidence_doc_derives_count_and_schema():
    assert build_model_load_evidence_doc([]) == {
        "schema_version": 1,
        "events": [],
        "model_load_count": 0,
    }
    events = [{"event": "model_load", "index": 1, "status": "STARTED"}]
    doc = build_model_load_evidence_doc(events)
    assert doc["model_load_count"] == 1
    assert doc["events"][0]["status"] == "STARTED"
    assert build_model_load_evidence_doc([{}, {}])["model_load_count"] == 2


def test_live_execution_persists_durable_model_load_evidence():
    source = Path("public_demo/live_execution.py").read_text()
    assert "model-load-events.json" in source
    assert "model_load_evidence_path" in source
    assert '"model_load_count": 1' not in source


def test_routing_cardinality_requires_exact_four_step_shape():
    routing = {
        "expected_transformer_invocations": 4,
        "observed_transformer_invocations": 4,
        "transfer_0_to_1_count": 4,
        "transformer_return_1_to_0_count": 4,
        "vae_input_transfer_0_to_1_count": 1,
        "single_t2i_instance": True,
    }
    assert routing_cardinality_ok(routing, 4)
    routing["transfer_0_to_1_count"] = 3
    assert not routing_cardinality_ok(routing, 4)


def test_routing_integrity_requires_both_gpus_and_block_integrity():
    routing = {
        "block_order_valid": True,
        "no_skipped_blocks": True,
        "no_duplicated_blocks_within_invocation": True,
        "gpu0_participation": True,
        "gpu1_participation": True,
    }
    assert routing_integrity_ok(routing)
    routing["gpu1_participation"] = False
    assert not routing_integrity_ok(routing)


def test_component_dtypes_prefers_floating_histogram():
    report = {
        "text_encoder": {"status": "PASS", "parameter_dtype_histogram": {"torch.bfloat16": 10}},
        "transformer": {"status": "PASS", "parameter_dtype_histogram": {"torch.bfloat16": 20}},
        "vae": {"status": "PASS", "parameter_dtype_histogram": {"torch.bfloat16": 5}},
    }
    assert component_dtypes(report) == {
        "text_encoder": "torch.bfloat16",
        "transformer": "torch.bfloat16",
        "vae": "torch.bfloat16",
    }


def test_cpu_fallback_detects_any_cpu_endpoint():
    assert not cpu_fallback_observed([{"device": "cuda:0"}, {"from": "cuda:0", "to": "cuda:1"}])
    assert cpu_fallback_observed([{"from": "cpu", "to": "cuda:1"}])


def test_placement_is_derived_from_live_device_plan():
    plan = {
        "text_encoder": "cuda:0",
        "vae": "cuda:1",
        "transformer": {
            "pre": {"img_in": "cuda:0"},
            "blocks": {"0": "cuda:0", "1": "cuda:1"},
            "post": {"norm_out": "cuda:1"},
        },
    }
    placement = placement_from_plan(plan)
    assert placement["block 0"] == "cuda:0"
    assert placement["block 1"] == "cuda:1"
    assert placement["post.norm_out"] == "cuda:1"
