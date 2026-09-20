from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict

from .gates import (
    component_dtypes,
    count_vae_input_transfer_0_to_1,
    cpu_fallback_observed,
    placement_from_plan,
)


def execute_live_inference(
    *,
    out_dir: Path,
    run: Any,
    contract: Any,
    model_path: str,
    source_authority: Dict[str, Any],
    run_id: str,
    num_blocks: int,
    split_block: int,
    record_gate: Callable[[str, bool, str], None],
) -> Dict[str, Any]:
    from mage_t4x2.dtype_audit import audit_runtime_dtypes
    from mage_t4x2.evidence_reducers import (
        adjudicate_multistep_block_integrity,
        derive_gpu_participation,
        derive_single_t2i_instance,
    )
    from mage_t4x2.hashing import sha256_file
    from mage_t4x2.sdpa_contract import assert_sdpa_frozen
    from scripts import gpu_session as gs

    pre = gs.preflight_gpu(require_t4x2=True)
    gpu_inventory = pre["inventory"]
    record_gate("G0_HARDWARE", pre["t4x2_ok"], f"gpu_count={pre['device_count']}")
    if not pre["t4x2_ok"]:
        raise RuntimeError("G0_HARDWARE: exactly two Tesla T4 GPUs are required")

    state = gs.load_runtime_state_dual_t4_spread(model_path, contract, source_authority)
    run.write_dtype("before", audit_runtime_dtypes(state))
    record_gate("L0_LOAD_ONCE", True, "model loaded once")

    post_load_sdpa = assert_sdpa_frozen()
    backend = (post_load_sdpa.get("backend") or {}).get("backend")
    record_gate("SDPA_POST_LOAD", post_load_sdpa.get("status") == "PASS", f"backend={backend}")
    if post_load_sdpa.get("status") != "PASS":
        raise RuntimeError(f"SDPA_POST_LOAD: {post_load_sdpa}")

    plan_result = gs.apply_dual_t4_mixed(state, num_blocks=num_blocks, split_block=split_block)
    device_plan = plan_result["device_plan"]
    record_gate("PLACEMENT", plan_result["ok"], "dual T4 explicit placement applied")
    if not plan_result["ok"]:
        raise RuntimeError(f"PLACEMENT: {plan_result}")

    pre_inference_sdpa = assert_sdpa_frozen()
    backend = (pre_inference_sdpa.get("backend") or {}).get("backend")
    record_gate(
        "SDPA_PRE_INFERENCE",
        pre_inference_sdpa.get("status") == "PASS",
        f"backend={backend}",
    )
    if pre_inference_sdpa.get("status") != "PASS":
        raise RuntimeError(f"SDPA_PRE_INFERENCE: {pre_inference_sdpa}")

    inference_id = f"t2i-{run_id}"
    telemetry_path = str(out_dir / "telemetry.jsonl")
    with gs.create_telemetry(
        telemetry_path,
        run_id=run_id,
        phase="public_demo",
        inference_id=inference_id,
    ) as telemetry:
        adapter = gs.attach_dual_adapter(
            state,
            device_plan,
            telemetry=telemetry,
            run_id=run_id,
            phase="public_demo",
            inference_id=inference_id,
        )
        try:
            result = gs.run_t2i(
                state,
                contract,
                run,
                telemetry,
                run_id,
                inference_id=inference_id,
                phase="public_demo",
            )
        finally:
            gs.detach_dual_adapter(state, adapter, reason="public_demo_complete")

    run.write_dtype("after", result["dtype_after"])
    run.copy_output_image(result["output_png"], "output.png")
    records = gs.collect_telemetry(telemetry_path)
    adjudication = adjudicate_multistep_block_integrity(
        records,
        run_id=run_id,
        inference_id=inference_id,
        num_blocks=num_blocks,
        expected_invocations=int(contract.steps),
    )
    gpu0 = derive_gpu_participation(records, "cuda:0", inference_id=inference_id)
    gpu1 = derive_gpu_participation(records, "cuda:1", inference_id=inference_id)
    single = derive_single_t2i_instance(records)

    boundary = [
        item for item in records
        if item.get("event") == "cross_device_transfer"
        and item.get("from") == "cuda:0"
        and item.get("to") == "cuda:1"
    ]
    returns = [
        item for item in records
        if item.get("event") == "transformer_output_return_transfer"
        and item.get("from") == "cuda:1"
        and item.get("to") == "cuda:0"
    ]

    dtype_labels = component_dtypes(result["dtype_after"])
    all_bf16 = all(
        result["dtype_after"].get(name, {}).get("status") == "PASS"
        for name in ("text_encoder", "transformer", "vae")
    )
    unexpected = sorted({
        *sum(
            (
                result["dtype_after"].get(name, {}).get("unexpected_floating_dtypes", [])
                for name in ("text_encoder", "transformer", "vae")
            ),
            [],
        )
    })
    validation = result["validation"]
    output_path = result["output_png"]
    output_sha = sha256_file(output_path) if os.path.isfile(output_path) else None
    memory_plan = getattr(state, "memory_plan", {}) or {}

    return {
        "inference_id": inference_id,
        "hardware": {
            "device_count": pre["device_count"],
            "gpu0": gpu_inventory[0]["name"] if len(gpu_inventory) > 0 else "n/a",
            "gpu1": gpu_inventory[1]["name"] if len(gpu_inventory) > 1 else "n/a",
        },
        "runtime": {
            "model_load_count": 1,
            "split_block": int(device_plan.get("split_block", memory_plan.get("split_block", split_block))),
            "num_blocks": int(device_plan.get("num_transformer_blocks", memory_plan.get("num_transformer_blocks", num_blocks))),
            "attention_impl": "sdpa",
        },
        "routing": {
            "expected_transformer_invocations": int(contract.steps),
            "observed_transformer_invocations": int(adjudication.get("observed_invocations", 0)),
            "transfer_0_to_1_count": len(boundary),
            "transformer_return_1_to_0_count": len(returns),
            "vae_input_transfer_0_to_1_count": count_vae_input_transfer_0_to_1(records),
            "gpu0_participation": gpu0["status"] == "PASS",
            "gpu1_participation": gpu1["status"] == "PASS",
            "single_t2i_instance": single["status"] == "PASS",
            "no_skipped_blocks": bool(adjudication.get("NO_SKIPPED_BLOCKS")),
            "no_duplicated_blocks_within_invocation": bool(adjudication.get("NO_DUPLICATED_BLOCKS")),
            "block_order_valid": bool(adjudication.get("BLOCK_ORDER_VALID")),
        },
        "precision": {
            "text_encoder": dtype_labels.get("text_encoder", "unknown"),
            "transformer": dtype_labels.get("transformer", "unknown"),
            "vae": dtype_labels.get("vae", "unknown"),
            "unexpected_floating_dtypes": unexpected,
            "status": "PASS" if all_bf16 else "FAIL",
        },
        "safety": {"cpu_fallback_observed": cpu_fallback_observed(records)},
        "output": {
            "exists": bool(validation.get("exists")),
            "width": int(validation.get("width") or 0),
            "height": int(validation.get("height") or 0),
            "mode": validation.get("mode"),
            "nan": bool(result.get("has_nan")),
            "inf": bool(result.get("has_inf")),
            "sha256": output_sha,
            "path": str(output_path),
            "is_512x512": bool(validation.get("is_512x512")),
            "rgb_valid": bool(validation.get("rgb_valid")),
        },
        "placement": placement_from_plan(device_plan),
        "invocation_block_sequences": adjudication.get("invocation_block_sequences", []),
    }
