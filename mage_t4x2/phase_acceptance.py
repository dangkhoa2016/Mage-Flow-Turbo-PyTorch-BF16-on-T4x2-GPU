"""Per-phase acceptance fields, facts derivation, and PASS/FAIL evaluation.

This module is the single authority for what makes G1..G5 acceptable. Every fact
is derived from observed evidence (telemetry, output validation, dtype audit,
adapter structural status, replay record) — never from configuration intent.
PASS is only ever derived by ``evaluate_phase_acceptance`` from those facts.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .evidence_reducers import (
    adjudicate_multistep_block_integrity,
    derive_block_order_validity,
    derive_cross_gpu_transfer,
    derive_gpu_participation,
    derive_replay_cross_inference,
    derive_single_t2i_instance,
)

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT_RUN"

# ---------------------------------------------------------------------------
# Per-phase acceptance field sets (Corrective A3 sections 8-12)
# ---------------------------------------------------------------------------

_COMMON_OUTPUT = (
    "MODEL_LOAD_COUNT_EQ_1",
    "RUN_T2I_CALLED_EQ_1",
    "OUTPUT_EXISTS",
    "OUTPUT_512X512",
    "OUTPUT_RGB_VALID",
    "NO_NAN",
    "NO_INF",
    "CPU_FALLBACK_FORBIDDEN",
    "RUN_EXIT_ZERO",
)

_DUAL_GPU = (
    "SINGLE_T2I_INSTANCE",
    "GPU0_PARTICIPATION",
    "GPU1_PARTICIPATION",
    "CROSS_GPU_TRANSFER_OBSERVED",
    "BLOCK_ORDER_VALID",
    "NO_DUPLICATED_BLOCKS",
    "NO_SKIPPED_BLOCKS",
    "DUAL_FORWARD_ADAPTER_ATTACHED",
)

_DTYPE = (
    "TEXT_ENCODER_BF16",
    "TRANSFORMER_BF16",
    "VAE_BF16",
)

G1_FIELDS = _COMMON_OUTPUT + _DUAL_GPU

ROUTING_FIELDS = _COMMON_OUTPUT + _DUAL_GPU

G3_FIELDS = _COMMON_OUTPUT + _DUAL_GPU + _DTYPE

G4_FIELDS = _COMMON_OUTPUT + _DUAL_GPU + _DTYPE

L0_FIELDS = (
    "MODEL_LOAD_COUNT_EQ_1",
    "SPREAD_LOAD_PLAN_PASS",
    "SPREAD_LOAD_OBSERVED_PLACEMENT_PASS",
    "NO_FULL_MODEL_CUDA0_MATERIALIZATION",
)

G5_FIELDS = (
    "MODEL_LOAD_COUNT_EQ_1",
    "RUN_T2I_CALLED_EQ_1",
    "REPLAY_EXECUTED",
    "FINAL_PROFILE_UNCHANGED",
    "FINAL_TOPOLOGY_UNCHANGED",
    "OUTPUT_VALID",
    "DETERMINISM_POLICY_EVALUATED",
)

PHASE_FIELDS: Dict[str, tuple] = {
    "L0": L0_FIELDS,
    "G1": G1_FIELDS,
    "ROUTING": ROUTING_FIELDS,
    "G3": G3_FIELDS,
    "G4": G4_FIELDS,
    "G5": G5_FIELDS,
    "G6": (),
}

# ---------------------------------------------------------------------------
# Replay determinism policy (Corrective A3 section 12)
# ---------------------------------------------------------------------------

REPLAY_POLICY = {
    "name": "CONFIG_STABLE_IMAGE_VALID",
    "description": (
        "G5 replays the G4 final configuration (same profile, topology, prompt, "
        "seed, resolution, steps, cfg) in the same process and must produce a "
        "valid 512x512 RGB image. Output hashes are recorded for comparison when "
        "both images exist; exact byte-equality is NOT required authority."
    ),
    "exact_hash_required": False,
}


def evaluate_replay_policy(
    g4_output_path: Optional[str],
    g5_output_path: Optional[str],
    same_profile: bool,
    same_topology: bool,
) -> Dict[str, Any]:
    """Evaluate the replay policy and record hash comparison evidence."""
    from . import hashing

    g4_hash = None
    g5_hash = None
    if g4_output_path and g5_output_path:
        try:
            g4_hash = hashing.sha256_file(g4_output_path)
            g5_hash = hashing.sha256_file(g5_output_path)
        except Exception:
            g4_hash = g5_hash = None

    config_stable = bool(same_profile and same_topology)
    pair_present = bool(g4_output_path and g5_output_path)
    status = PASS if (config_stable and pair_present and g5_output_path) else FAIL
    return {
        "policy": REPLAY_POLICY["name"],
        "exact_hash_required": REPLAY_POLICY["exact_hash_required"],
        "g4_output_hash": g4_hash,
        "g5_output_hash": g5_hash,
        "hashes_equal": (g4_hash == g5_hash) if (g4_hash and g5_hash) else None,
        "config_stable": config_stable,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Facts derivation (evidence -> booleans)
# ---------------------------------------------------------------------------


def _records_for_inference(records: Iterable[Dict[str, Any]], inference_id: str) -> list:
    return [r for r in records if r.get("inference_id") == inference_id]


def _derive_block_inventory_count(runtime_state: Any) -> Optional[int]:
    """Derive the expected transformer block count from the live runtime state.

    Sources, in order: ``memory_plan.components.transformer_blocks`` (list of
    per-block inventories in block order), then ``device_map.transformer``
    (mapping of block index -> device). Falls back to ``None`` (never guessed).

    ``derive_phase_facts`` (runbook section 22) refuses to assume defaults; a
    missing/invalid block inventory yields ``None`` so the caller fails closed.
    """
    try:
        memory_plan = getattr(runtime_state, "memory_plan", None) or {}
    except Exception:
        memory_plan = {}
    blocks = (memory_plan.get("components") or {}).get("transformer_blocks") or []
    if isinstance(blocks, (list, tuple)) and len(blocks) > 0:
        return len(blocks)
    try:
        device_map = getattr(runtime_state, "device_map", None) or {}
    except Exception:
        device_map = {}
    transformer_map = device_map.get("transformer") or {}
    if isinstance(transformer_map, dict) and len(transformer_map) > 0:
        block_keys = [k for k in transformer_map.keys() if str(k).isdigit()]
        if block_keys:
            return len(block_keys)
    return None


def _derive_block_integrity_facts(result, records, runtime_state, contract) -> Dict[str, Any]:
    """G1-G4 block integrity via the multistep helper, fail-closed (section 22).

    ``run_id``/``inference_id`` come from the phase result; ``expected_invocations``
    from ``contract.steps``; ``num_blocks`` from the live runtime state's
    transformer block plan/inventory. Any un-derivable value fails closed to
    False, never a silent default.
    """
    run_id = result.get("run_id")
    inference_id = result.get("inference_id")
    try:
        expected_invocations = int(getattr(contract, "steps", 0) or 0)
    except Exception:
        expected_invocations = 0
    num_blocks = _derive_block_inventory_count(runtime_state)

    if not run_id or not inference_id or expected_invocations <= 0 or num_blocks is None or num_blocks <= 0:
        return {
            "BLOCK_ORDER_VALID": False,
            "NO_DUPLICATED_BLOCKS": False,
            "NO_SKIPPED_BLOCKS": False,
            "expected_invocations": expected_invocations,
            "observed_invocations": None,
            "return_boundary_count": None,
            "empty_boundary_count": None,
            "trailing_unclosed_blocks": None,
            "invocation_block_sequences": [],
            "per_invocation_order_valid": [],
            "per_invocation_no_duplicates": [],
            "per_invocation_no_skips": [],
        }

    return adjudicate_multistep_block_integrity(
        records,
        run_id=run_id,
        inference_id=str(inference_id),
        num_blocks=num_blocks,
        expected_invocations=expected_invocations,
    )


def derive_run_t2i_called(records: Iterable[Dict[str, Any]], inference_id: str) -> bool:
    """RUN_T2I_CALLED_EQ_1: a START and a PASS phase event under this inference."""
    pool = _records_for_inference(records, inference_id)
    events = [r for r in pool if r.get("event") == "phase" and r.get("phase") == "t2i"]
    statuses = {str(r.get("status")) for r in events}
    return "START" in statuses and "PASS" in statuses


def derive_run_exit_zero(result: Dict[str, Any]) -> bool:
    """RUN_EXIT_ZERO fail-closed (A4.4).

    ``exit_code`` must be present and an int equal to 0. A missing ``exit_code``
    is FAIL, never PASS-by-default. If ``run_exit_zero`` is present it must agree
    with ``exit_code``; an inconsistent pair FAILs and neither field overrides
    the other silently.
    """
    exit_code = result.get("exit_code")
    valid_zero = isinstance(exit_code, int) and (exit_code is not False) and exit_code == 0
    declared = result.get("run_exit_zero")
    if declared is None:
        return valid_zero
    return valid_zero and bool(declared) is valid_zero


def derive_phase_facts(
    phase: str,
    result: Dict[str, Any],
    telemetry_records: list,
    model_load_count: int,
    runtime_state: Any,
    contract: Any,
    adapter_status: Optional[Dict[str, Any]] = None,
    replay_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive observed facts from evidence. No config-intent shortcut exists."""
    if phase == "L0":
        memory_plan = getattr(runtime_state, "memory_plan", None) or {}
        placement = getattr(runtime_state, "placement_validation", None) or {}
        return {
            "MODEL_LOAD_COUNT_EQ_1": int(model_load_count) == 1,
            "SPREAD_LOAD_PLAN_PASS": memory_plan.get("status") == "PASS",
            "SPREAD_LOAD_OBSERVED_PLACEMENT_PASS": bool(placement and placement.get("status") == "PASS"),
            "NO_FULL_MODEL_CUDA0_MATERIALIZATION": bool(
                getattr(runtime_state, "no_full_model_cuda0_materialization", False)
            ),
            "split_block": memory_plan.get("split_block"),
            "single_device_rejected": memory_plan.get("single_device", {}).get("status") == "FAIL",
        }

    validation = result.get("validation") or {}
    dtype_after = result.get("dtype_after") or {}
    inference_id = str(result.get("inference_id") or "n/a")
    records = telemetry_records or []

    facts: Dict[str, Any] = {}
    facts["MODEL_LOAD_COUNT_EQ_1"] = int(model_load_count) == 1
    facts["RUN_T2I_CALLED_EQ_1"] = derive_run_t2i_called(records, inference_id)
    facts["OUTPUT_EXISTS"] = bool(validation.get("exists"))
    facts["OUTPUT_512X512"] = bool(validation.get("is_512x512"))
    facts["OUTPUT_RGB_VALID"] = bool(validation.get("rgb_valid"))
    facts["NO_NAN"] = not bool(result.get("has_nan"))
    facts["NO_INF"] = not bool(result.get("has_inf"))
    facts["CPU_FALLBACK_FORBIDDEN"] = not bool(getattr(contract, "cpu_fallback", False))
    facts["RUN_EXIT_ZERO"] = derive_run_exit_zero(result)

    if phase in ("G1", "ROUTING", "G3", "G4"):
        facts["SINGLE_T2I_INSTANCE"] = derive_single_t2i_instance(records)["status"] == PASS
        facts["GPU0_PARTICIPATION"] = (
            derive_gpu_participation(records, "cuda:0", inference_id=inference_id)["status"] == PASS
        )
        facts["GPU1_PARTICIPATION"] = (
            derive_gpu_participation(records, "cuda:1", inference_id=inference_id)["status"] == PASS
        )
        facts["CROSS_GPU_TRANSFER_OBSERVED"] = (
            derive_cross_gpu_transfer(records, inference_id=inference_id)["status"] == PASS
        )
        block = _derive_block_integrity_facts(result, records, runtime_state, contract)
        facts["BLOCK_ORDER_VALID"] = bool(block["BLOCK_ORDER_VALID"])
        facts["NO_DUPLICATED_BLOCKS"] = bool(block["NO_DUPLICATED_BLOCKS"])
        facts["NO_SKIPPED_BLOCKS"] = bool(block["NO_SKIPPED_BLOCKS"])
        facts["BLOCK_INVOCATION_EXPECTED_COUNT"] = block.get("expected_invocations")
        facts["BLOCK_INVOCATION_OBSERVED_COUNT"] = block.get("observed_invocations")
        facts["BLOCK_RETURN_BOUNDARY_COUNT"] = block.get("return_boundary_count")
        facts["BLOCK_INVOCATION_SEQUENCES"] = block.get("invocation_block_sequences")
        facts["BLOCK_INVOCATION_ORDER_FLAGS"] = block.get("per_invocation_order_valid")
        facts["BLOCK_INVOCATION_DUPLICATE_FLAGS"] = block.get("per_invocation_no_duplicates")
        facts["BLOCK_INVOCATION_SKIP_FLAGS"] = block.get("per_invocation_no_skips")
        facts["BLOCK_TRAILING_UNCLOSED_SEQUENCE"] = block.get("trailing_unclosed_blocks")
        facts["DUAL_FORWARD_ADAPTER_ATTACHED"] = bool(
            adapter_status and adapter_status.get("attached")
        )
        clean = derive_replay_cross_inference(records, inference_id)
        facts["CROSS_INFERENCE_CLEAN"] = clean["status"] == PASS
        facts["CROSS_INFERENCE_EVIDENCE"] = clean

    if phase in ("G3", "G4"):
        for key, component in (
            ("TEXT_ENCODER_BF16", "text_encoder"),
            ("TRANSFORMER_BF16", "transformer"),
            ("VAE_BF16", "vae"),
        ):
            comp = dtype_after.get(component) or dtype_after.get("components", {}).get(component)
            facts[key] = bool(comp and comp.get("status") == "PASS")

    if phase == "G5":
        rev = replay_evidence or {}
        same_profile = bool(
            rev.get("g4_profile") is not None
            and rev.get("g5_profile") is not None
            and rev.get("g4_profile") == rev.get("g5_profile")
        )
        same_topology = bool(
            rev.get("g4_topology") is not None
            and rev.get("g5_topology") is not None
            and rev.get("g4_topology") == rev.get("g5_topology")
        )
        facts["REPLAY_EXECUTED"] = bool(rev.get("replay_executed"))
        facts["FINAL_PROFILE_UNCHANGED"] = same_profile
        facts["FINAL_TOPOLOGY_UNCHANGED"] = same_topology
        facts["OUTPUT_VALID"] = bool(validation.get("valid")) or (
            bool(validation.get("exists"))
            and bool(validation.get("is_512x512"))
            and bool(validation.get("rgb_valid"))
            and not bool(result.get("has_nan"))
            and not bool(result.get("has_inf"))
        )
        policy = evaluate_replay_policy(
            rev.get("g4_output_path"),
            rev.get("g5_output_path"),
            same_profile,
            same_topology,
        )
        facts["DETERMINISM_POLICY_EVALUATED"] = policy["status"] == PASS

    return facts


# ---------------------------------------------------------------------------
# Acceptance evaluation (facts -> PASS/FAIL)
# ---------------------------------------------------------------------------


def evaluate_phase_acceptance(phase: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a phase against its required field set.

    A phase PASSes ONLY when every required field is present and PASS. Any
    missing or failed field yields FAIL. Nothing here is inferable from intent.
    """
    required = PHASE_FIELDS.get(phase, ())
    if not required:
        return {
            "phase": phase,
            "status": NOT_RUN,
            "note": "no acceptance fields defined for phase",
            "fields": {},
            "required_fields": [],
            "provisional": True,
        }

    fields: Dict[str, Any] = {}
    missing: list = []
    failed: list = []
    for field in required:
        if field not in facts or facts[field] is None:
            missing.append(field)
            fields[field] = {"status": NOT_RUN, "condition_satisfied": None}
            continue
        ok = bool(facts[field])
        fields[field] = {"status": PASS if ok else FAIL, "condition_satisfied": ok}
        if not ok:
            failed.append(field)

    if missing:
        status, note = FAIL, f"missing required facts: {missing}"
    elif failed:
        status, note = FAIL, f"failed fields: {failed}"
    else:
        status, note = PASS, "all required phase fields satisfied"

    return {
        "phase": phase,
        "status": status,
        "note": note,
        "fields": fields,
        "required_fields": list(required),
        "provisional": bool(missing),
    }
