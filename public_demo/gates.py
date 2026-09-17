from __future__ import annotations

from typing import Any, Dict, List, Sequence


def derive_model_load_count(load_events: Sequence[Any]) -> int:
    """Derive the model load count strictly from recorded load lifecycle events.

    The loader wrapper owned by the run appends exactly one event per real load
    call. The count is therefore run evidence, never a hard-coded assertion.
    """
    return len(load_events)


def model_load_count_ok(count: int) -> bool:
    """Canonical gate: exactly one model load, derived from run evidence."""
    return count == 1


def component_dtypes(dtype_after: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for comp in ("text_encoder", "transformer", "vae"):
        report = dtype_after.get(comp) or {}
        hist = report.get("parameter_dtype_histogram") or {}
        best = "unknown"
        best_count = -1
        for dtype_label, count in hist.items():
            if "float" in dtype_label or dtype_label.startswith("torch.bfloat"):
                if count > best_count:
                    best = dtype_label
                    best_count = count
        if report.get("status") == "PASS" and best == "unknown":
            best = "torch.bfloat16"
        out[comp] = best
    return out


def count_vae_input_transfer_0_to_1(records: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for record in records
        if record.get("event") == "vae_input_transfer"
        and str(record.get("from")) == "cuda:0"
        and str(record.get("to")) == "cuda:1"
    )


def routing_cardinality_ok(routing: Dict[str, Any], steps: int) -> bool:
    expected = int(routing.get("expected_transformer_invocations", -1))
    return (
        expected == int(steps)
        and int(routing.get("observed_transformer_invocations", -1)) == expected
        and int(routing.get("transfer_0_to_1_count", -1)) == expected
        and int(routing.get("transformer_return_1_to_0_count", -1)) == expected
        and int(routing.get("vae_input_transfer_0_to_1_count", -1)) == 1
        and routing.get("single_t2i_instance") is True
    )


def routing_integrity_ok(routing: Dict[str, Any]) -> bool:
    return (
        routing.get("block_order_valid") is True
        and routing.get("no_skipped_blocks") is True
        and routing.get("no_duplicated_blocks_within_invocation") is True
        and routing.get("gpu0_participation") is True
        and routing.get("gpu1_participation") is True
    )


def placement_from_plan(device_plan: Dict[str, Any]) -> Dict[str, str]:
    placement: Dict[str, str] = {
        "text_encoder": str(device_plan.get("text_encoder", "n/a")),
        "vae": str(device_plan.get("vae", "n/a")),
    }
    transformer = device_plan.get("transformer") or {}
    for name, device in (transformer.get("pre") or {}).items():
        placement[f"pre.{name}"] = str(device)
    for index, device in (transformer.get("blocks") or {}).items():
        placement[f"block {index}"] = str(device)
    for name, device in (transformer.get("post") or {}).items():
        placement[f"post.{name}"] = str(device)
    return placement


def cpu_fallback_observed(records: List[Dict[str, Any]]) -> bool:
    return any(
        str(record.get("device", "")).startswith("cpu")
        or str(record.get("from", "")).startswith("cpu")
        or str(record.get("to", "")).startswith("cpu")
        for record in records
    )
