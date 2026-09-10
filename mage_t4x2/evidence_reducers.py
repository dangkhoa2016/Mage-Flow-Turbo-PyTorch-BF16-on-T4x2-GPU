"""Evidence-derived GPU participation and transfer reducers.

Pure functions that consume telemetry/evidence records and return authority
facts. Booleans here are ALWAYS derived from actual evidence — never from config
intent. During CPU Stage A the observed GPU values remain ``NOT_RUN``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

NOT_RUN = "NOT_RUN"
PASS = "PASS"
FAIL = "FAIL"


def _records_by_inference(records: Iterable[Dict[str, Any]], inference_id: str) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("inference_id") == inference_id]


def derive_gpu_participation(
    telemetry_records: List[Dict[str, Any]],
    device: str,
    inference_id: Optional[str] = None,
) -> Dict[str, Any]:
    """PASS only if runtime telemetry contains actual block_forward events on
    the requested device for the same run/inference."""
    pool = telemetry_records
    if inference_id is not None:
        pool = _records_by_inference(telemetry_records, inference_id)
    events = [
        r for r in pool
        if r.get("event") == "block_forward"
        and str(r.get("device")) == str(device)
    ]
    status = PASS if events else FAIL
    return {
        "device": device,
        "status": status,
        "inference_id": inference_id,
        "event_count": len(events),
        "sample_blocks": [r.get("block") for r in events[:5]],
    }


def derive_cross_gpu_transfer(
    telemetry_records: List[Dict[str, Any]],
    inference_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """PASS only if at least one cross_device_transfer event has from != to,
    from the same run and (if given) the same inference."""
    pool = telemetry_records
    if inference_id is not None:
        pool = _records_by_inference(telemetry_records, inference_id)
    if run_id is not None:
        pool = [r for r in pool if r.get("run_id") == run_id]
    transfers = [
        r for r in pool
        if r.get("event") == "cross_device_transfer"
        and r.get("from") is not None
        and r.get("to") is not None
        and str(r.get("from")) != str(r.get("to"))
    ]
    status = PASS if transfers else FAIL
    return {
        "status": status,
        "inference_id": inference_id,
        "run_id": run_id,
        "event_count": len(transfers),
        "sample_boundaries": [r.get("block_boundary") for r in transfers[:5]],
    }


def derive_t4x2_hardware(gpu_inventory: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """PASS only when the runtime inventory shows exactly two Tesla T4 GPUs."""
    inventory = gpu_inventory or []
    names = [str(d.get("name", "")) for d in inventory]
    caps = [str(d.get("compute_capability", "")) for d in inventory]
    status = (
        PASS if (
            len(inventory) == 2
            and all("T4" in n for n in names)
            and all(c == "7.5" for c in caps)
        ) else FAIL
    )
    return {
        "status": status,
        "device_count": len(inventory),
        "names": names,
        "compute_capabilities": caps,
    }


def derive_dtype_status(dtype_audit: Dict[str, Any], component: str) -> Dict[str, Any]:
    """PASS only when the dtype audit report shows PASS for the component."""
    if not dtype_audit:
        return {"component": component, "status": NOT_RUN, "reason": "no audit"}
    comp = dtype_audit.get(component) or dtype_audit.get("components", {}).get(component)
    if comp is None:
        return {"component": component, "status": FAIL, "reason": "component missing from audit"}
    status = PASS if str(comp.get("status")) == "PASS" else FAIL
    return {
        "component": component,
        "status": status,
        "parameter_count": comp.get("float_parameter_count"),
        "float_buffer_count": comp.get("float_buffer_count"),
        "histogram": comp.get("dtype_histogram"),
        "unexpected": comp.get("unexpected_dtype_list"),
    }


def derive_model_identity(
    source_authority: Dict[str, Any],
    attached_model_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """MODEL_REVISION_MATCH / MODEL_ID_MATCH derived from evidence, not path name."""
    expected_id = (source_authority or {}).get("model_identifier")
    expected_rev = (source_authority or {}).get("model_revision")
    manifest_id = (attached_model_manifest or {}).get("model_identifier")
    manifest_rev = (attached_model_manifest or {}).get("model_revision")
    id_ok = bool(manifest_id) and expected_id is not None and manifest_id == expected_id
    rev_ok = bool(manifest_rev) and expected_rev is not None and manifest_rev == expected_rev
    return {
        "MODEL_ID_MATCH": id_ok,
        "MODEL_REVISION_MATCH": rev_ok,
        "expected_model_identifier": expected_id,
        "manifest_model_identifier": manifest_id,
        "expected_model_revision": expected_rev,
        "manifest_model_revision": manifest_rev,
        "status": PASS if (id_ok and rev_ok) else FAIL,
    }


def derive_single_t2i_instance(telemetry_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Derive SINGLE_T2I_INSTANCE from unique inference ids in telemetry."""
    ids = {
        str(r.get("inference_id"))
        for r in telemetry_records
        if r.get("inference_id")
    }
    status = PASS if len(ids) == 1 else (FAIL if len(ids) > 1 else NOT_RUN)
    return {
        "status": status,
        "inference_ids": sorted(ids),
        "count": len(ids),
    }


def derive_block_order_validity(telemetry_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """BLOCK_ORDER_VALID: block_forward events exactly [0, N), each once, ascending."""
    events = [
        r for r in telemetry_records
        if r.get("event") == "block_forward" and r.get("block") is not None
    ]
    idxs = [int(r["block"]) for r in events]
    if not idxs:
        return {"status": FAIL, "reason": "no block events"}
    expected = list(range(min(idxs), max(idxs) + 1))
    order_ok = idxs == sorted(idxs)
    coverage_ok = sorted(set(idxs)) == expected
    no_duplicates = len(set(idxs)) == len(idxs)
    status = PASS if order_ok and coverage_ok and no_duplicates else FAIL
    return {
        "status": status,
        "order_ok": order_ok,
        "coverage_ok": coverage_ok,
        "duplicate_free": no_duplicates,
        "block_count": len(idxs),
        "block_indices": idxs,
        "expected": expected,
    }


def derive_replay_cross_inference(
    telemetry_records: List[Dict[str, Any]],
    expected_inference_id: str,
) -> Dict[str, Any]:
    """Cross-inference contamination check: GPU0 evidence from one inference must
    not be combined with GPU1 evidence from another to claim success."""
    gpu0_ids = {
        r.get("inference_id")
        for r in telemetry_records
        if r.get("event") == "block_forward" and str(r.get("device")) == "cuda:0"
    }
    gpu1_ids = {
        r.get("inference_id")
        for r in telemetry_records
        if r.get("event") == "block_forward" and str(r.get("device")) == "cuda:1"
    }
    transfer_ids = {
        r.get("inference_id")
        for r in telemetry_records
        if r.get("event") == "cross_device_transfer"
    }
    union = gpu0_ids | gpu1_ids | transfer_ids
    clean = union == {expected_inference_id} if union else False
    return {
        "status": PASS if clean else FAIL,
        "expected_inference_id": expected_inference_id,
        "gpu0_inference_ids": sorted(gpu0_ids),
        "gpu1_inference_ids": sorted(gpu1_ids),
        "transfer_inference_ids": sorted(transfer_ids),
        "single_shared_inference": clean,
    }


def adjudicate_multistep_block_integrity(
    records: Iterable[Dict[str, Any]],
    run_id: Optional[str] = None,
    inference_id: Optional[str] = None,
    num_blocks: Optional[int] = None,
    expected_invocations: Optional[int] = None,
) -> Dict[str, Any]:
    """Adjudicate block integrity across multiple transformer invocations.

    Groups block_forward events into invocations delimited by
    ``transformer_output_return_transfer`` events, then validates each
    invocation independently before evaluating aggregate invariants.
    """
    # ---- filter to the requested scope ----
    def _in_scope(r: Dict[str, Any]) -> bool:
        if run_id is not None and r.get("run_id") != run_id:
            return False
        if inference_id is not None and r.get("inference_id") != inference_id:
            return False
        return True

    scoped = [r for r in records if _in_scope(r)]

    # ---- split into invocations by return-transfer boundaries ----
    invocations: List[List[int]] = []
    current: List[int] = []
    return_boundary_count = 0
    empty_boundary_count = 0

    for r in scoped:
        event = r.get("event")
        if event == "block_forward":
            current.append(r.get("block"))
        elif event == "transformer_output_return_transfer":
            return_boundary_count += 1
            if current:
                invocations.append(current)
                current = []
            else:
                empty_boundary_count += 1

    trailing_unclosed_blocks = current
    observed_invocations = len(invocations)

    # ---- per-invocation diagnostics ----
    full_set = set(range(num_blocks)) if num_blocks is not None else set()
    canonical_seq = list(range(num_blocks)) if num_blocks is not None else []

    per_invocation_order_valid = [seq == canonical_seq for seq in invocations]
    per_invocation_no_duplicates = [len(set(seq)) == len(seq) for seq in invocations]
    per_invocation_no_skips = [set(seq) == full_set for seq in invocations]

    all_canonical = all(per_invocation_order_valid) if invocations else False
    all_dedup = all(per_invocation_no_duplicates) if invocations else False
    all_coverage = all(per_invocation_no_skips) if invocations else False

    has_data = len(invocations) > 0
    no_trailing = not trailing_unclosed_blocks
    expected_positive = expected_invocations is not None and int(expected_invocations) > 0
    blocks_positive = num_blocks is not None and int(num_blocks) > 0

    return {
        "BLOCK_ORDER_VALID": (
            has_data
            and expected_positive
            and blocks_positive
            and all_canonical
            and observed_invocations == expected_invocations
            and return_boundary_count == expected_invocations
            and empty_boundary_count == 0
            and no_trailing
        ),
        "NO_DUPLICATED_BLOCKS": has_data and all_dedup,
        "NO_SKIPPED_BLOCKS": (
            has_data
            and blocks_positive
            and expected_positive
            and all_coverage
            and observed_invocations == expected_invocations
        ),
        "expected_invocations": expected_invocations,
        "observed_invocations": observed_invocations,
        "return_boundary_count": return_boundary_count,
        "empty_boundary_count": empty_boundary_count,
        "trailing_unclosed_blocks": trailing_unclosed_blocks,
        "invocation_block_sequences": invocations,
        "per_invocation_order_valid": per_invocation_order_valid,
        "per_invocation_no_duplicates": per_invocation_no_duplicates,
        "per_invocation_no_skips": per_invocation_no_skips,
    }
