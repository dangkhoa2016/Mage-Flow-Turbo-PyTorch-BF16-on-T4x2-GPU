from public_demo.gates import (
    component_dtypes,
    count_vae_input_transfer_0_to_1,
    cpu_fallback_observed,
    placement_from_plan,
    routing_cardinality_ok,
    routing_integrity_ok,
)


def test_direction_specific_vae_count():
    records = [
        {"event": "vae_input_transfer", "from": "cuda:0", "to": "cuda:1"},
        {"event": "vae_input_transfer", "from": "cuda:1", "to": "cuda:0"},
        {"event": "vae_input_transfer", "from": "cpu", "to": "cuda:1"},
    ]
    assert count_vae_input_transfer_0_to_1(records) == 1


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
