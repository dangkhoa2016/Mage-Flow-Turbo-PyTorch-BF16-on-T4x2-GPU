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
