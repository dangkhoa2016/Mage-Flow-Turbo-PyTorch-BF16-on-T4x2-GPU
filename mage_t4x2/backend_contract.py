"""Real-backend authority interface freeze (Corrective A4.1).

Every callable that ``StageBSession`` dispatches against the phase backend
(real ``scripts.gpu_session`` or synthetic ``SyntheticPhaseBackend``) is
enumerated here with its frozen signature and return schema. This module is
CPU-safe (no torch imports) and can be imported by parity tests and the A4
audit scripts without initializing CUDA.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

# One canonical list shared by the session, the parity tests and the audits.
# ``run_to_facts`` is StageBSession-dispatch compatible (it is mapped in
# ``StageBSession._backend``), so it is frozen here too.
#
# Corrective G0_LOAD_OOM: the two single-T4 gates (full-model materialization
# onto one T4) were RETIRED from the authority contract — the physically
# impossible single-T4 full-model materialization is forbidden. Only the
# dual-T4 gates remain.
REAL_BACKEND_CALLABLES: Tuple[str, ...] = (
    "load_runtime_state",
    "preflight_gpu",
    "apply_dual_t4_mixed",
    "apply_dual_t4_all_bf16",
    "create_evidence_run",
    "create_telemetry",
    "attach_dual_adapter",
    "detach_dual_adapter",
    "run_t2i",
    "collect_telemetry",
    "derive_phase_facts",
    "evaluate_phase_acceptance",
    "resolve_model_path",
    "run_to_facts",
)

REQUIRED_RESULT_KEYS: Tuple[str, ...] = (
    "output_png",
    "validation",
    "dtype_after",
    "has_nan",
    "has_inf",
    "inference_id",
    "run_id",
    "phase",
    "exit_code",
)

REQUIRED_DTYPE_COMPONENTS: Tuple[str, ...] = ("text_encoder", "transformer", "vae")

REQUIRED_DTYPE_COMPONENT_KEYS: Tuple[str, ...] = (
    "status",
    "parameter_dtype_histogram",
    "buffer_dtype_histogram",
    "unexpected_floating_dtypes",
)

# Authority dtype keys consumed by the shared G3/G4 acceptance reducer.
REQUIRED_DTYPE_AUTHORITY_KEYS: Tuple[str, ...] = ("text_encoder", "transformer", "vae", "overall_status")


def entry_points(module: Any) -> Dict[str, Optional[Callable]]:
    """Return ``{callable_name: fn}`` for every frozen callable reachable on the
    module (attribute or mapping style ``module[name]``)."""
    found: Dict[str, Optional[Callable]] = {}
    for name in REAL_BACKEND_CALLABLES:
        fn = None
        try:
            fn = getattr(module, name)
        except AttributeError:
            if hasattr(module, "__getitem__"):
                try:
                    fn = module[name]
                except (KeyError, TypeError):
                    fn = None
        found[name] = fn
    return found


def signature_of(fn: Optional[Callable]) -> Dict[str, Any]:
    if fn is None:
        return {"present": False}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {"present": True, "params": [], "return": "unknown", "error": "signature unavailable"}
    return {
        "present": True,
        "params": [p.name for p in sig.parameters.values()],
        "has_var_kwargs": any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()),
        "return": str(sig.return_annotation) if sig.return_annotation is not inspect.Signature.empty else "unannotated",
    }


def required_invocation_shapes() -> Dict[str, Dict[str, Any]]:
    """Canonical keyword shapes StageBSession uses for each dispatch (A4.1)."""
    return {
        "load_runtime_state": {"positional": ["model_path", "contract", "source_authority"], "kwargs": []},
        "preflight_gpu": {"positional": [], "kwargs": ["require_t4x2"]},
        "apply_dual_t4_mixed": {"positional": ["state", "num_blocks"], "kwargs": ["split_block"]},
        "apply_dual_t4_all_bf16": {"positional": ["state", "num_blocks"], "kwargs": ["split_block"]},
        "create_evidence_run": {"positional": ["run_dir"], "kwargs": ["source_authority"]},
        "create_telemetry": {"positional": ["path", "run_id"], "kwargs": ["phase", "inference_id"]},
        "attach_dual_adapter": {"positional": ["state", "device_plan"], "kwargs": ["telemetry", "run_id", "phase", "inference_id"]},
        "detach_dual_adapter": {"positional": ["state"], "kwargs": ["adapter", "reason"]},
        "run_t2i": {"positional": ["state", "contract", "run", "telemetry", "run_id"], "kwargs": ["inference_id", "phase"]},
        "collect_telemetry": {"positional": ["path"], "kwargs": []},
        "derive_phase_facts": {"positional": ["phase", "result", "telemetry_records", "model_load_count", "runtime_state", "contract"], "kwargs": ["adapter_status", "replay_evidence"]},
        "evaluate_phase_acceptance": {"positional": ["phase", "facts"], "kwargs": []},
        "resolve_model_path": {"positional": ["contract"], "kwargs": ["override"]},
        "run_to_facts": {"positional": ["result", "contract", "topo", "dual"], "kwargs": ["telemetry_records", "gpu_inventory"]},
    }


def call_shape_match(
    fn: Optional[Callable],
    required_positional: List[str],
    required_kwargs: List[str],
) -> bool:
    """Prove StageBSession invocation compatibility (positional/keyword absorb)."""
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    names = list(sig.parameters.keys())
    for req in required_positional:
        if req not in names and not any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        ):
            return False
    if required_kwargs and not has_var_kwargs:
        for req in required_kwargs:
            if req not in names:
                return False
    return True
