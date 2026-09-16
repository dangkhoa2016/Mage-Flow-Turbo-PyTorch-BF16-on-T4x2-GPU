"""Failure taxonomy and classification for the qualification project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FAILURE_TAXONOMY: List[Dict[str, str]] = [
    {"code": "SOURCE_DRIFT", "description": "pinned upstream commit no longer matches loaded source"},
    {"code": "DEPENDENCY_DRIFT", "description": "frozen dependency authority no longer matches runtime"},
    {"code": "MODEL_LOAD_FAILURE", "description": "model files incomplete or failed to materialize"},
    {"code": "UNSUPPORTED_BF16_OP", "description": "a CUDA kernel refuses bf16 execution on T4"},
    {"code": "DTYPE_MISMATCH", "description": "tensor dtype inconsistent with the precision contract"},
    {"code": "DEVICE_MISMATCH", "description": "tensor on an unexpected device"},
    {"code": "CUDA_OOM", "description": "out-of-memory on a CUDA device"},
    {"code": "CROSS_DEVICE_ROUTING_FAILURE", "description": "activation routing between GPUs failed"},
    {"code": "NO_GPU0_ACTIVITY", "description": "no observed computation on cuda:0"},
    {"code": "NO_GPU1_ACTIVITY", "description": "no observed computation on cuda:1"},
    {"code": "CPU_FALLBACK", "description": "authority run silently moved work to CPU"},
    {"code": "INVALID_IMAGE", "description": "output missing, wrong geometry, or invalid channel layout"},
    {"code": "NAN_INF", "description": "output or intermediate tensor contains NaN/Inf"},
    {"code": "NONDETERMINISM", "description": "replay did not match the deterministic policy"},
    {"code": "MULTIPLE_MODEL_LOADS", "description": "authority session loaded the model more than once"},
    {"code": "DUAL_FORWARD_ADAPTER_MISSING", "description": "blocks split but the dual-forward adapter is not attached"},
    {"code": "INFERENCE_NOT_EXECUTED", "description": "phase runner never executed the backend inference call"},
    {"code": "INFERENCE_RUNTIME_FAILURE", "description": "backend inference raised or returned an unusable result"},
    {"code": "PHASE_ACCEPTANCE_FAILED", "description": "observed evidence did not satisfy the phase acceptance fields"},
    {"code": "ADAPTER_TELEMETRY_MISMATCH", "description": "adapter telemetry ownership does not match the current phase/inference"},
    {"code": "ADAPTER_DOUBLE_ATTACH", "description": "more than one live adapter patch on the same transformer"},
    {"code": "TRANSFORMER_POST_DEVICE_MISMATCH", "description": "post-block module (norm_out/proj_out) not on the final block device"},
    {"code": "MISSING_PHASE_EVIDENCE", "description": "required phase evidence files are absent or malformed"},
    {"code": "REPLAY_NOT_EXECUTED", "description": "G5 replay did not perform a real final-configuration inference"},
    {"code": "REPLAY_PROFILE_DRIFT", "description": "G5 precision profile drifted from the G4 final profile"},
    {"code": "REPLAY_TOPOLOGY_DRIFT", "description": "G5 topology drifted from the G4 final topology"},
    {"code": "TELEMETRY_FAILURE", "description": "telemetry missing, malformed, or contradicted by evidence"},
    {"code": "MEMORY_PLAN_FAIL", "description": "no conservative dual-T4 spread plan exists; concrete GPU budgets cannot be met"},
    {"code": "FULL_MODEL_SINGLE_DEVICE_FORBIDDEN", "description": "whole-model materialization on a single device is forbidden by the corrective"},
    {"code": "SPREAD_LOAD_PLACEMENT_MISMATCH", "description": "observed parameter/buffer placement differs from the approved spread plan"},
    {"code": "SDPA_NOT_FROZEN", "description": "attention backend/implementation is not frozen to SDPA for the T4/sm75 run"},
    {"code": "NVIDIA_LIB_MISSING", "description": "NVIDIA driver/user-mode libraries are not present on LD_LIBRARY_PATH"},
    {"code": "UNCLASSIFIED", "description": "failure did not match any known taxonomy entry"},
]


def taxonomy_list() -> List[Dict[str, str]]:
    return [dict(entry) for entry in FAILURE_TAXONOMY]


def classify(exc: Optional[BaseException], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify an exception into a taxonomy entry."""
    context = context or {}
    message = str(exc) if exc else context.get("message", "")
    desc = message.lower()

    table = [
        ("MODEL_LOAD_FAILURE", lambda: "safetensors" in desc or "from_pretrained" in desc or "load_state_dict" in desc),
        ("MEMORY_PLAN_FAIL", lambda: "memory_plan_fail" in desc or "spread plan" in desc and "no conservative" in desc),
        ("FULL_MODEL_SINGLE_DEVICE_FORBIDDEN", lambda: "single device" in desc or "materializ" in desc and "single" in desc),
        ("SPREAD_LOAD_PLACEMENT_MISMATCH", lambda: "spread_load" in desc or "placement" in desc and "plan" in desc),
        ("SDPA_NOT_FROZEN", lambda: "sdpa" in desc or "attention backend" in desc),
        ("NVIDIA_LIB_MISSING", lambda: "nvidia" in desc or "libcuda" in desc or "ld_library_path" in desc),
        ("CUDA_OOM", lambda: "out of memory" in desc or "cuda out of memory" in desc),
        ("UNSUPPORTED_BF16_OP", lambda: "not implemented for 'bfloat16'" in desc or "bfloat16" in desc and "not implemented" in desc),
        ("DTYPE_MISMATCH", lambda: "dtype" in desc and ("expected" in desc or "mismatch" in desc)),
        ("DEVICE_MISMATCH", lambda: "device" in desc and ("expected" in desc or "mismatch" in desc or "are on different devices" in desc)),
        ("TRANSFORMER_POST_DEVICE_MISMATCH", lambda: "transformer_post_device_mismatch" in desc),
        ("CROSS_DEVICE_ROUTING_FAILURE", lambda: "routing" in desc or "transfer" in desc and "device" in desc),
        ("INVALID_IMAGE", lambda: "image" in desc and ("invalid" in desc or "size" in desc)),
        ("NAN_INF", lambda: "nan" in desc or "inf" in desc),
        ("SOURCE_DRIFT", lambda: "commit" in desc or "sha" in desc),
        ("DEPENDENCY_DRIFT", lambda: "version" in desc and ("mismatch" in desc or "pinned" in desc)),
        ("CPU_FALLBACK", lambda: "cpu fallback" in desc or "offload" in desc),
        ("NONDETERMINISM", lambda: "deterministic" in desc or "replay" in desc),
        ("TELEMETRY_FAILURE", lambda: "telemetry" in desc or "jsonl" in desc),
    ]
    for code, pred in table:
        if pred():
            return {"category": code, "details": message, "code": code}
    return {"category": "UNCLASSIFIED", "details": message, "code": "UNCLASSIFIED"}


def write_taxonomy(path: str) -> str:
    data = {
        "taxonomy": taxonomy_list(),
        "classifier": "mage_t4x2.failures.classify",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
