"""Acceptance engine: machine-readable PASS / FAIL / NOT_RUN verdicts.

GPU-specific fields must be ``NOT_RUN`` during CPU Stage A. A phase must not be
marked PASS merely because the process exited with code 0 — evidence must
satisfy each acceptance field directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ACCEPTANCE_FIELDS: List[str] = [
    # CPU-verifiable
    "PYTORCH_RUNTIME",
    "UPSTREAM_MAGE_PINNED",
    "MODEL_ID_MATCH",
    "MODEL_REVISION_MATCH",
    "CPU_FALLBACK_FORBIDDEN",
    # GPU-verifiable
    "T4_COUNT_EQ_2",
    "TEXT_ENCODER_BF16",
    "TRANSFORMER_BF16",
    "VAE_BF16",
    "SINGLE_T2I_INSTANCE",
    "GPU0_PARTICIPATION",
    "GPU1_PARTICIPATION",
    "CROSS_GPU_TRANSFER_OBSERVED",
    "OUTPUT_EXISTS",
    "OUTPUT_512X512",
    "OUTPUT_RGB_VALID",
    "NO_NAN",
    "NO_INF",
    "RUN_EXIT_ZERO",
]

CPU_FIELDS: set = {
    "PYTORCH_RUNTIME",
    "UPSTREAM_MAGE_PINNED",
    "MODEL_ID_MATCH",
    "MODEL_REVISION_MATCH",
    "CPU_FALLBACK_FORBIDDEN",
}

GPU_FIELDS: set = {f for f in ACCEPTANCE_FIELDS if f not in CPU_FIELDS}


def evaluate_acceptance(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate facts (bool condition-satisfied per field) into a report."""
    fields: Dict[str, Any] = {}
    for field in ACCEPTANCE_FIELDS:
        if field not in facts:
            fields[field] = {"status": "NOT_RUN", "condition_satisfied": None}
            continue
        value = bool(facts[field])
        fields[field] = {
            "status": "PASS" if value else "FAIL",
            "condition_satisfied": value,
        }

    failed = [f for f in ACCEPTANCE_FIELDS if fields[f]["status"] == "FAIL"]
    passed_cpu = [f for f in sorted(CPU_FIELDS) if fields[f]["status"] == "PASS"]
    ran_gpu = [f for f in sorted(GPU_FIELDS) if fields[f]["status"] in ("PASS", "FAIL")]

    if failed:
        overall = ("FAIL", f"failed fields: {failed}")
    elif ran_gpu:
        missing_gpu = [f for f in sorted(GPU_FIELDS) if fields[f]["status"] == "NOT_RUN"]
        if passed_cpu and not missing_gpu:
            overall = ("PASS", "all CPU and GPU acceptance fields satisfied")
        else:
            overall = (
                "PARTIAL_PASS",
                f"CPU fields satisfied: {passed_cpu}; GPU fields pending: {missing_gpu}",
            )
    elif passed_cpu:
        overall = (
            "CPU_STAGE_PASS_GPU_PENDING",
            "CPU Stage A fields satisfied; GPU phases not yet executed.",
        )
    else:
        overall = ("NOT_RUN", "no acceptance evidence supplied")

    return {
        "fields": fields,
        "overall_status": overall[0],
        "overall_note": overall[1],
    }


def facts_for_cpu_stage(source_ok: bool, model_id_ok: bool, model_rev_ok: bool, cpu_fallback_ok: bool) -> Dict[str, Any]:
    return {
        "PYTORCH_RUNTIME": True,
        "UPSTREAM_MAGE_PINNED": source_ok,
        "MODEL_ID_MATCH": model_id_ok,
        "MODEL_REVISION_MATCH": model_rev_ok,
        "CPU_FALLBACK_FORBIDDEN": cpu_fallback_ok,
    }


def write_acceptance_json(report: Dict[str, Any], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
