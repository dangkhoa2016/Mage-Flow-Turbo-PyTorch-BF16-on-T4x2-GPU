"""Public dual-T4 production showcase lane (Lane B).

Lane A (``public_demo.runner``) protects the canonical qualification contract:
one independent model load and one canonical T2I trajectory. Lane B, implemented
here, is a separate, sustained multi-image demo: it performs its own single
model load and keeps that instance hot across a deterministic multi-prompt,
multi-seed, multi-resolution workload (512 / 768 / 1024), recording provable
timing, GPU-memory (allocated and reserved), routing and output evidence per
image. Outputs are independent PNG files plus a derived labeled contact-sheet
visual index; the contact sheet itself is never a direct model output.

The two lanes never share a single model load, and this module never claims
otherwise. Lane B never rewrites Lane A evidence.

Everything that does not require CUDA is a pure/CPU-safe function so static CI
can lock the showcase contract without a GPU. ``run_public_showcase`` is the
only entry that touches PyTorch, and it eagerly requires exactly two CUDA
devices before any model load is attempted.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mage_t4x2.hashing import sha256_file
from mage_t4x2.image_validation import validate_image
from public_demo.contract import CANONICAL

# --------------------------------------------------------------------------- #
# Deterministic showcase workload (the case table) and lane contract.
# --------------------------------------------------------------------------- #

MIN_SHOWCASE_CASES = 12
MAX_SHOWCASE_CASES = 24
TARGET_SHOWCASE_INFERENCE_SECONDS = 300.0
SHOWCASE_STEPS = 4
SHOWCASE_CFG = 1.0
SHOWCASE_RESOLUTIONS: Tuple[int, ...] = (512, 768, 1024)
SHOWCASE_ARTIFACTS_REL = "artifacts/public-showcase"
SHOWCASE_FINAL_VERDICT_KEY = "PUBLIC_SHOWCASE_FINAL_VERDICT"

_MANDATORY_CASES_PER_RESOLUTION = 4  # 3 resolutions * 4 = 12 mandatory cases
_EXTENSION_ROUNDS = 4  # 4 rounds * 3 resolutions = 12 extension cases
_BASE_SEED = 1001


class PublicShowcaseGateFailure(RuntimeError):
    """Raised when a Lane B showcase gate fails precisely by name."""

    def __init__(self, gate: str, message: str) -> None:
        self.gate = gate
        super().__init__(f"{gate}: {message}")


# Immutable deterministic prompt catalogue: 12 mandatory cases then 12
# extension cases. No runtime prompt generation, no external prompt API, no
# date/time or UUID-derived prompt content. Every entry is a fixed string.
SHOWCASE_PROMPT_SPECS: Tuple[Dict[str, str], ...] = (
    {
        "category": "wildlife-winter",
        "prompt": (
            "a red fox standing in a snowy pine forest at golden hour, detailed fur, "
            "soft atmospheric light, natural wildlife photography"
        ),
    },
    {
        "category": "editorial-portrait",
        "prompt": (
            "a fictional adult traveler in a simple linen jacket beside a large window, "
            "soft daylight, neutral background, editorial portrait photography"
        ),
    },
    {
        "category": "brutalist-architecture",
        "prompt": (
            "a monumental brutalist concrete museum beside a reflecting pool, overcast sky, "
            "precise architectural photography, strong geometric composition"
        ),
    },
    {
        "category": "alpine-landscape",
        "prompt": (
            "an alpine lake surrounded by rugged mountains at sunrise, low mist over the "
            "water, crisp reflections, cinematic landscape photography"
        ),
    },
    {
        "category": "macro-nature",
        "prompt": (
            "a dragonfly covered with morning dew resting on a green fern, extreme macro "
            "photography, shallow depth of field, fine wing detail"
        ),
    },
    {
        "category": "desert-vehicle",
        "prompt": (
            "a classic unbranded roadster driving along an empty desert highway, warm "
            "late-afternoon sunlight, dust in the distance, cinematic automotive photo"
        ),
    },
    {
        "category": "rainy-city-night",
        "prompt": (
            "a narrow modern city street at night after rain, colorful neon reflections "
            "on wet pavement, pedestrians with umbrellas, cinematic urban photography"
        ),
    },
    {
        "category": "scandinavian-interior",
        "prompt": (
            "a calm Scandinavian reading room with pale wood furniture, a wool chair, "
            "large windows and soft morning light, realistic interior photography"
        ),
    },
    {
        "category": "fantasy-environment",
        "prompt": (
            "floating green islands above a vast cloud layer with thin waterfalls "
            "descending into the mist, dramatic sunlight, detailed fantasy environment"
        ),
    },
    {
        "category": "tropical-coast",
        "prompt": (
            "an aerial view of a turquoise tropical cove with pale sand, dark volcanic "
            "rocks and small waves, clear midday light, realistic coastal photography"
        ),
    },
    {
        "category": "ceramic-still-life",
        "prompt": (
            "a handmade ceramic teapot and two cups on a natural linen tablecloth, soft "
            "side light, muted tones, refined product still-life photography"
        ),
    },
    {
        "category": "railway-canyon",
        "prompt": (
            "a long freight train crossing a steel bridge through a dramatic red-rock "
            "canyon, late-day sunlight, wide cinematic landscape composition"
        ),
    },
    {
        "category": "artisan-food",
        "prompt": (
            "fresh artisan bread, figs and herbs on a rustic kitchen table, warm window "
            "light, realistic food photography, rich natural texture"
        ),
    },
    {
        "category": "night-observatory",
        "prompt": (
            "a remote astronomical observatory on a dark mountain ridge beneath a dense "
            "Milky Way sky, long-exposure astrophotography style"
        ),
    },
    {
        "category": "underwater-wildlife",
        "prompt": (
            "a manta ray gliding above a colorful coral reef in clear tropical water, "
            "sunbeams below the surface, realistic underwater photography"
        ),
    },
    {
        "category": "terraced-agriculture",
        "prompt": (
            "layered green rice terraces winding across misty hills at dawn, small paths "
            "and farm huts, atmospheric documentary landscape photography"
        ),
    },
    {
        "category": "minimal-fashion",
        "prompt": (
            "a fictional adult model wearing a minimalist monochrome outfit in a clean "
            "studio, soft directional lighting, contemporary fashion editorial"
        ),
    },
    {
        "category": "bird-action",
        "prompt": (
            "a kingfisher diving toward a clear river with water droplets frozen in "
            "motion, telephoto wildlife photography, sharp natural detail"
        ),
    },
    {
        "category": "scifi-rover",
        "prompt": (
            "an autonomous exploration rover crossing a dark basalt plain on a distant "
            "planet beneath a pale sky, realistic science-fiction concept photography"
        ),
    },
    {
        "category": "winter-village",
        "prompt": (
            "a small mountain village covered in fresh snow during blue hour, warm lights "
            "inside wooden houses, quiet atmospheric winter photography"
        ),
    },
    {
        "category": "botanical-greenhouse",
        "prompt": (
            "a glass greenhouse filled with large tropical plants, filtered sunlight, "
            "humid air and detailed leaves, realistic botanical photography"
        ),
    },
    {
        "category": "mountain-cycling",
        "prompt": (
            "a road cyclist climbing a dramatic high mountain pass, distant switchbacks "
            "and deep valleys, bright clear weather, sports photography"
        ),
    },
    {
        "category": "storm-lighthouse",
        "prompt": (
            "a solitary lighthouse on a rocky coast during a powerful storm, large waves "
            "and dark clouds, dramatic realistic seascape photography"
        ),
    },
    {
        "category": "glassblowing-craft",
        "prompt": (
            "an artisan shaping glowing molten glass inside a traditional workshop, "
            "orange furnace light, sparks, detailed documentary photography"
        ),
    },
)


def _build_showcase_case_table() -> Tuple[Dict[str, object], ...]:
    """Build the immutable 24-row showcase case table deterministically.

    Each of the 24 rows pairs one immutable prompt spec with one resolution:
    mandatory cases (s01..s12) are four repetitions of each base resolution in
    resolution order (512, 768, 1024), extension cases (s13..s24) round-robin
    the three resolutions four times. Seeds advance monotonically 1001..1024.
    Rows use the runbook ``id/category/prompt/seed/width/height/role`` schema.
    """
    cases: List[Dict[str, object]] = []
    mandatory_pattern = [
        resolution
        for resolution in SHOWCASE_RESOLUTIONS
        for _ in range(_MANDATORY_CASES_PER_RESOLUTION)
    ]
    extension_pattern = list(SHOWCASE_RESOLUTIONS) * _EXTENSION_ROUNDS
    resolution_pattern = mandatory_pattern + extension_pattern
    for offset, (resolution, prompt_spec) in enumerate(
        zip(resolution_pattern, SHOWCASE_PROMPT_SPECS),
        start=1,
    ):
        case_id = f"s{offset:02d}"
        role = "mandatory" if offset <= MIN_SHOWCASE_CASES else "extension"
        cases.append(
            {
                "id": case_id,
                "category": str(prompt_spec["category"]),
                "prompt": str(prompt_spec["prompt"]),
                "seed": _BASE_SEED + offset - 1,
                "width": int(resolution),
                "height": int(resolution),
                "role": role,
            }
        )
    return tuple(cases)


SHOWCASE_CASES: Tuple[Dict[str, object], ...] = _build_showcase_case_table()


def showcase_case_ids() -> List[str]:
    return [str(case["id"]) for case in SHOWCASE_CASES]


def showcase_case_table_errors() -> List[str]:
    """Return every invariant violation in the showcase case table, else [].

    The table is the deterministic workload contract: exactly 24 rows, ids
    ``s01..s24``, four mandatory copies of each base resolution, four extension
    resolution-order rounds, monotonic seeds 1001..1024 and a non-empty prompt
    plus category on every row, with all 24 normalized prompts and all 12
    mandatory categories distinct. The old single-prefix ``(sNN)`` scheme is
    structurally forbidden here (no prompt may merely append its case-id tag).
    """
    errors: List[str] = []
    ids = showcase_case_ids()
    if len(ids) != MAX_SHOWCASE_CASES:
        errors.append(f"expected {MAX_SHOWCASE_CASES} cases, got {len(ids)}")
    if len(set(ids)) != len(ids):
        errors.append("case ids must be unique")
    expected_ids = [f"s{i:02d}" for i in range(1, MAX_SHOWCASE_CASES + 1)]
    if ids != expected_ids:
        errors.append("case ids must be the deterministic sequence s01..s24")

    if len(SHOWCASE_PROMPT_SPECS) != MAX_SHOWCASE_CASES:
        errors.append(
            f"expected {MAX_SHOWCASE_CASES} immutable prompt specs, "
            f"got {len(SHOWCASE_PROMPT_SPECS)}"
        )
    if any(not str(spec.get("prompt") or "").strip() for spec in SHOWCASE_PROMPT_SPECS):
        errors.append("every prompt spec must carry a non-empty prompt")
    if any(
        not str(spec.get("category") or "").strip() for spec in SHOWCASE_PROMPT_SPECS
    ):
        errors.append("every prompt spec must carry a non-empty category")
    if len(SHOWCASE_CASES) != len(SHOWCASE_PROMPT_SPECS):
        errors.append(
            "the case table must pair every immutable prompt spec with one case"
        )

    widths = [int(case["width"]) for case in SHOWCASE_CASES]
    heights = [int(case["height"]) for case in SHOWCASE_CASES]
    seeds = [int(case["seed"]) for case in SHOWCASE_CASES]
    if any(w not in SHOWCASE_RESOLUTIONS for w in widths):
        errors.append("width must be one of 512 / 768 / 1024")
    if widths != heights:
        errors.append("every case must be square (width == height)")
    expected_seeds = list(range(_BASE_SEED, _BASE_SEED + MAX_SHOWCASE_CASES))
    if seeds != expected_seeds:
        errors.append("seeds must advance monotonically from 1001 to 1024")

    mandatory = [c for c in SHOWCASE_CASES if c["role"] == "mandatory"]
    if len(mandatory) != MIN_SHOWCASE_CASES:
        errors.append(f"expected {MIN_SHOWCASE_CASES} mandatory cases")
    elif [int(c["width"]) for c in mandatory] != [
        512,
        512,
        512,
        512,
        768,
        768,
        768,
        768,
        1024,
        1024,
        1024,
        1024,
    ]:
        errors.append("mandatory cases must be 4x512, 4x768, 4x1024 in order")

    extension = [c for c in SHOWCASE_CASES if c["role"] == "extension"]
    if len(extension) != MAX_SHOWCASE_CASES - MIN_SHOWCASE_CASES:
        errors.append(f"expected {MAX_SHOWCASE_CASES - MIN_SHOWCASE_CASES} extension cases")
    elif [int(c["width"]) for c in extension] != list(SHOWCASE_RESOLUTIONS) * _EXTENSION_ROUNDS:
        errors.append("extension cases must round-robin 512, 768, 1024 four times")

    normalized_prompts = [
        " ".join(str(case["prompt"]).lower().split())
        for case in SHOWCASE_CASES
    ]
    if len(normalized_prompts) != len(set(normalized_prompts)):
        errors.append(
            "all showcase prompts must be semantically distinct fixed strings"
        )
    mandatory_categories = [
        str(case["category"]) for case in SHOWCASE_CASES[:MIN_SHOWCASE_CASES]
    ]
    if len(set(mandatory_categories)) != MIN_SHOWCASE_CASES:
        errors.append(f"all {MIN_SHOWCASE_CASES} mandatory categories must be distinct")

    for index, case in enumerate(SHOWCASE_CASES, start=1):
        if str(case["id"]) != f"s{index:02d}":
            errors.append(f"case index {index} has wrong id {case['id']!r}")
        if not isinstance(case["prompt"], str) or not case["prompt"]:
            errors.append(f"case {case['id']} must carry a non-empty prompt")
        if not isinstance(case["category"], str) or not case["category"]:
            errors.append(f"case {case['id']} must carry a non-empty category")
        if str(case["prompt"]).endswith(f"({case['id']})"):
            errors.append(
                f"case {case['id']} prompt must not be a common prefix plus its case-id suffix"
            )
        if int(case["seed"]) <= 0:
            errors.append(f"case {case['id']} must carry a positive seed")
    return errors


def showcase_output_name(case_id: str, resolution: int, seed: int) -> str:
    """Deterministic per-image output filename, seed-qualified.

    The runbook pins the output path spelling ``outputs/<case>-<res>x<res>
    -seed<seed>.png``; omitting the seed from the filename would let two cases
    collide or make the recorded timing/digest ambiguous.
    """
    return f"{case_id}-{int(resolution)}x{int(resolution)}-seed{int(seed)}.png"


def _case_by_id(case_id: str) -> Dict[str, object]:
    for case in SHOWCASE_CASES:
        if case["id"] == case_id:
            return case
    raise KeyError(f"no such showcase case: {case_id}")


def should_execute_next_showcase_case(
    executed_count: int,
    cumulative_inference_seconds: float,
) -> bool:
    """Decide whether one more showcase case should run, after real timings.

    Pure decision over already-executed bookkeeping: the first 12 mandatory
    cases always run; strictly above that, execution continues only while the
    cumulative real inference time is below the target duration and the 24-case
    hard maximum has not been reached. It never reads wall clock, model-load
    duration, setup duration or current time, so it is testable on CPU.
    """
    if executed_count < MIN_SHOWCASE_CASES:
        return True
    if executed_count >= MAX_SHOWCASE_CASES:
        return False
    return cumulative_inference_seconds < TARGET_SHOWCASE_INFERENCE_SECONDS


def build_case_output_validation(
    cases: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Expected output descriptors for a schedule: filename + requested size."""
    result: Dict[str, Dict[str, object]] = {}
    for case in cases:
        case_id = str(case["id"])
        resolution = int(case["width"])
        seed = int(case["seed"])
        result[case_id] = {
            "output_filename": showcase_output_name(case_id, resolution, seed),
            "expected_resolution": resolution,
            "width": resolution,
            "height": resolution,
            "seed": seed,
            "steps": SHOWCASE_STEPS,
            "cfg": SHOWCASE_CFG,
        }
    return result


def build_showcase_case_catalog(
    cases: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Full immutable plan covering every planned showcase case."""
    output_specs = build_case_output_validation(cases)
    catalog: Dict[str, Dict[str, object]] = {}

    for case in cases:
        case_id = str(case["id"])
        catalog[case_id] = {
            "case_id": case_id,
            "category": str(case["category"]),
            "prompt": str(case["prompt"]),
            "role": str(case["role"]),
            **output_specs[case_id],
        }

    return catalog


def sync_showcase_execution_scope(
    summary: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> None:
    """Set executed/id/expected-output evidence from completed cases only.

    The full plan catalog is never reduced here; only the execution-scoped
    fields (``executed_case_ids``, compatibility ``scheduled_case_ids`` and
    ``expected_outputs``) describe what this run actually completed.
    """
    executed_case_ids = [
        str(result["case_id"])
        for result in results
        if result.get("case_id")
    ]

    executed_cases = [
        _case_by_id(case_id)
        for case_id in executed_case_ids
    ]

    summary["executed_case_ids"] = executed_case_ids

    # Compatibility field retained only if existing consumers/docs expect it.
    summary["scheduled_case_ids"] = list(executed_case_ids)

    summary["expected_outputs"] = build_case_output_validation(
        executed_cases
    )


# --------------------------------------------------------------------------- #
# Routing facts (shared reducer semantics with Lane A).
# --------------------------------------------------------------------------- #


def build_showcase_routing_facts(
    telemetry_records: Iterable[Dict[str, Any]],
    *,
    run_id: str,
    inference_ids: Sequence[str],
    num_blocks: int,
    expected_invocations: int = SHOWCASE_STEPS,
) -> Dict[str, Any]:
    """Per-inference routing facts derived only from captured telemetry.

    Uses the same adjudicators as Lane A so both lanes share one interpretation
    of block order, participation and single-instance evidence. Facts are
    strictly evidence-derived; config intent never appears here.
    """
    from mage_t4x2.evidence_reducers import (
        adjudicate_multistep_block_integrity,
        derive_gpu_participation,
        derive_single_t2i_instance,
    )
    from public_demo.gates import cpu_fallback_observed

    records = list(telemetry_records)
    per_inference: Dict[str, Dict[str, Any]] = {}

    for inference_id in inference_ids:
        scoped = [r for r in records if r.get("inference_id") == inference_id]
        integrity = adjudicate_multistep_block_integrity(
            scoped,
            run_id=run_id,
            inference_id=inference_id,
            num_blocks=num_blocks,
            expected_invocations=expected_invocations,
        )
        single = derive_single_t2i_instance(scoped)
        gpu0 = derive_gpu_participation(scoped, "cuda:0", inference_id=inference_id)
        gpu1 = derive_gpu_participation(scoped, "cuda:1", inference_id=inference_id)
        per_inference[inference_id] = {
            "inference_id": inference_id,
            "observed_invocations": int(integrity.get("observed_invocations", 0)),
            "block_order_valid": bool(integrity.get("BLOCK_ORDER_VALID")),
            "no_skipped_blocks": bool(integrity.get("NO_SKIPPED_BLOCKS")),
            "no_duplicated_blocks_within_invocation": bool(integrity.get("NO_DUPLICATED_BLOCKS")),
            "gpu0_participation": gpu0["status"] == "PASS",
            "gpu1_participation": gpu1["status"] == "PASS",
            "single_t2i_instance": single["status"] == "PASS",
            "cpu_fallback_observed": cpu_fallback_observed(scoped),
            "block_sequence": (
                integrity["invocation_block_sequences"][-1]
                if integrity["invocation_block_sequences"]
                else []
            ),
        }

    return {
        "run_id": run_id,
        "inference_ids": list(inference_ids),
        "num_blocks": num_blocks,
        "expected_invocations_per_case": expected_invocations,
        "per_inference": per_inference,
    }


def derive_case_route_ok(facts: Dict[str, Any]) -> bool:
    """Every executed case must satisfy the shared routing contract."""
    per = (facts or {}).get("per_inference", {})
    if not per:
        return False
    for entry in per.values():
        if not all(
            [
                entry["block_order_valid"],
                entry["no_skipped_blocks"],
                entry["no_duplicated_blocks_within_invocation"],
                entry["gpu0_participation"],
                entry["gpu1_participation"],
                entry["single_t2i_instance"],
                entry["cpu_fallback_observed"] is False,
                int(entry["observed_invocations"]) == SHOWCASE_STEPS,
            ]
        ):
            return False
    return True


# --------------------------------------------------------------------------- #
# Evidence merger, persistence and digest verification.
# --------------------------------------------------------------------------- #


def _case_row_passed(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("case_id")
        and row.get("block_order_valid")
        and row.get("no_skipped_blocks")
        and row.get("no_duplicated_blocks_within_invocation")
        and row.get("gpu0_participation")
        and row.get("gpu1_participation")
        and row.get("single_t2i_instance")
        and row.get("cpu_fallback_observed") is False
        and int(row.get("observed_invocations") or 0) == SHOWCASE_STEPS
        and row.get("validation_ok")
        and row.get("adapter_detached")
        and row.get("export_ok")
    )


def _serializable_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "inference_id",
        "case_id",
        "category",
        "prompt",
        "resolution",
        "seed",
        "elapsed_seconds",
        "cumulative_seconds",
        "output_rel",
        "output_sha256",
        "peak_memory_gpu0_bytes",
        "peak_memory_reserved_gpu0_bytes",
        "peak_memory_gpu1_bytes",
        "peak_memory_reserved_gpu1_bytes",
    )
    out: Dict[str, Any] = {key: row[key] for key in keys if key in row}
    out["observed_invocations"] = row.get("observed_invocations")
    out["block_order_valid"] = row.get("block_order_valid")
    out["no_skipped_blocks"] = row.get("no_skipped_blocks")
    out["no_duplicated_blocks_within_invocation"] = row.get("no_duplicated_blocks_within_invocation")
    out["gpu0_participation"] = row.get("gpu0_participation")
    out["gpu1_participation"] = row.get("gpu1_participation")
    out["single_t2i_instance"] = row.get("single_t2i_instance")
    out["cpu_fallback_observed"] = row.get("cpu_fallback_observed")
    out["validation_ok"] = row.get("validation_ok")
    out["adapter_detached"] = row.get("adapter_detached")
    out["export_ok"] = row.get("export_ok")
    return out


def finalize_showcase_evidence(
    results: Sequence[Dict[str, Any]],
    routing_facts: Dict[str, Any],
    summary_base: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge per-case results with routing facts into the showcase summary.

    Pure merger: performs no I/O. The verdict is NOT_RUN before any case is
    executed, FAIL the moment any executed case fails a gate, and PASS only
    when every executed case passed inline and the minimum case tally is met.
    This function can never fabricate a PASS.
    """
    result_map = {r["inference_id"]: r for r in results}
    per_inference = (routing_facts or {}).get("per_inference", {})
    rows: List[Dict[str, Any]] = []
    for inference_id, facts in per_inference.items():
        remark = result_map.get(inference_id, {})
        row = dict(facts)
        row["case_id"] = remark.get("case_id")
        row["category"] = remark.get("category")
        row["prompt"] = remark.get("prompt")
        row["resolution"] = remark.get("resolution")
        row["seed"] = remark.get("seed")
        row["elapsed_seconds"] = remark.get("elapsed_seconds")
        row["cumulative_seconds"] = remark.get("cumulative_seconds")
        row["output_rel"] = remark.get("output_rel")
        row["output_sha256"] = remark.get("output_sha256")
        row["peak_memory_gpu0_bytes"] = remark.get("peak_memory_gpu0_bytes")
        row["peak_memory_reserved_gpu0_bytes"] = remark.get("peak_memory_reserved_gpu0_bytes")
        row["peak_memory_gpu1_bytes"] = remark.get("peak_memory_gpu1_bytes")
        row["peak_memory_reserved_gpu1_bytes"] = remark.get("peak_memory_reserved_gpu1_bytes")
        row["validation_ok"] = bool(remark.get("validation_ok"))
        row["adapter_detached"] = bool(remark.get("adapter_detached"))
        row["export_ok"] = bool(remark.get("export_ok"))
        rows.append(_serializable_case_row(row))

    executed = len(rows)
    passed = sum(1 for row in rows if _case_row_passed(row))

    summary: Dict[str, Any] = dict(summary_base)
    summary["cases_executed"] = executed
    summary["cases_passed"] = passed
    summary["cases_required_minimum"] = MIN_SHOWCASE_CASES
    summary["case_results"] = rows
    if executed == 0:
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "NOT_RUN"
    elif executed >= MIN_SHOWCASE_CASES and passed == executed:
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "PASS"
    else:
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "FAIL"
    return summary


def verify_digest_matches(evidence_dir: str, required: Dict[str, str]) -> List[str]:
    """Verify recorded SHA-256 digests for every required artifact name.

    Returns every mismatch/absence as a string in a list; [] means pass.
    """
    mismatches: List[str] = []
    directory = Path(evidence_dir)
    for rel, expected in required.items():
        path = directory / rel
        if not path.is_file():
            mismatches.append(f"missing artifact: {rel}")
            continue
        actual = sha256_file(str(path))
        if actual.lower() != str(expected).lower():
            mismatches.append(f"digest mismatch for {rel}: expected={expected} actual={actual}")
    return mismatches


def build_showcase_aggregate(
    results: Sequence[Dict[str, Any]],
    wall_seconds: float,
    load_seconds: float,
) -> Dict[str, Any]:
    """CPU-safe wall/inference/memory aggregate for the showcase summary.

    All values derive from captured case rows; per-resolution timing uses only
    ``statistics`` from the standard library.
    """
    rows = [r for r in results if isinstance(r, dict)]

    def _peak(key: str) -> Optional[int]:
        values = [int(r[key]) for r in rows if r.get(key) is not None]
        return max(values) if values else None

    total_inference = sum(
        float(r["elapsed_seconds"]) for r in rows if r.get("elapsed_seconds") is not None
    )
    wall = float(wall_seconds or 0.0)
    per_resolution: Dict[str, Dict[str, object]] = {}
    for resolution in SHOWCASE_RESOLUTIONS:
        times = [
            float(r["elapsed_seconds"])
            for r in rows
            if int(r.get("resolution")) == resolution and r.get("elapsed_seconds") is not None
        ]
        entry: Dict[str, object] = {"count": len(times)}
        if times:
            entry["median_seconds"] = round(statistics.median(times), 6)
            entry["mean_seconds"] = round(statistics.mean(times), 6)
            entry["min_seconds"] = round(min(times), 6)
            entry["max_seconds"] = round(max(times), 6)
        else:
            entry["median_seconds"] = None
            entry["mean_seconds"] = None
            entry["min_seconds"] = None
            entry["max_seconds"] = None
        per_resolution[f"{int(resolution)}x{int(resolution)}"] = entry

    return {
        "showcase_model_load_seconds": round(float(load_seconds or 0.0), 6),
        "showcase_total_inference_seconds": round(total_inference, 6),
        "showcase_wall_seconds": round(wall, 6),
        "total_images": len(rows),
        "images_per_minute": round(len(rows) * 60.0 / wall, 4) if wall > 0 else None,
        "global_gpu0_peak_allocated": _peak("peak_memory_gpu0_bytes"),
        "global_gpu0_peak_reserved": _peak("peak_memory_reserved_gpu0_bytes"),
        "global_gpu1_peak_allocated": _peak("peak_memory_gpu1_bytes"),
        "global_gpu1_peak_reserved": _peak("peak_memory_reserved_gpu1_bytes"),
        "per_resolution": per_resolution,
        "observed_on": "this Kaggle Tesla T4×2 run",
    }


# --------------------------------------------------------------------------- #
# Gallery (Pillow contact sheet, CPU-safe).
# --------------------------------------------------------------------------- #


def build_gallery(
    image_paths: Sequence[str],
    out_path: str,
    cols: int = 6,
    annotations: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compose a derived labeled contact-sheet visual-index PNG.

    This contact sheet is created afterward as a visual index over the already
    independent per-case output PNGs. It is a derived audit/summary artifact,
    not a direct model inference output. Each tile shows the image thumbnail
    plus a one-line label carrying the case id, resolution, seed and elapsed
    seconds (runbook gallery contract).
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise PublicShowcaseGateFailure("GALLERY", "no case outputs to compose")

    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(p).convert("RGB") for p in paths]
    labels = list(annotations) if annotations is not None else [p.name for p in paths]
    if len(labels) < len(images):
        labels.extend([""] * (len(images) - len(labels)))

    thumb_w, thumb_h = 256, 256
    label_h = 18
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (18, 18, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    border = (110, 110, 120)
    text = (210, 210, 216)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        thumb = image.resize((thumb_w, thumb_h))
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=border, width=1)
        draw.text((x + 4, y + thumb_h + 1), str(labels[index])[:56], font=font, fill=text)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")
    return {
        "path": str(out),
        "images": [str(p) for p in paths],
        "count": len(images),
        "columns": cols,
        "labels": list(labels),
        "sha256": sha256_file(str(out)),
    }


# --------------------------------------------------------------------------- #
# Telemetry helpers (per-case JSONL files under telemetry/).
# --------------------------------------------------------------------------- #


def showcase_telemetry_path(telemetry_dir: Path, case_id: str) -> Path:
    return telemetry_dir / f"{case_id}.jsonl"


def read_telemetry_records(telemetry_file: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in telemetry_file.open(encoding="utf-8"):
        if line.strip():
            records.append(json.loads(line))
    return records


def read_showcase_telemetry(telemetry_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(telemetry_dir.glob("*.jsonl")):
        records.extend(read_telemetry_records(path))
    return records


# --------------------------------------------------------------------------- #
# Live runtime (dual-T4 only).
# --------------------------------------------------------------------------- #


def summarize_showcase_model_load_events(
    events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Separate physical load attempts from lifecycle event records.

    A single successful model load records exactly two lifecycle events
    (``STARTED`` then ``PASS``), so the number of events must never be
    confused with the number of load attempts. ``model_load_count`` counts
    attempts only (how many times ``STARTED`` was recorded), while
    ``model_load_event_count`` counts all lifecycle records. A single clean
    successful load is ``["STARTED", "PASS"]`` => model_load_count=1,
    event_count=2, single_successful_model_load=True.
    """
    statuses = [str(event.get("status")) for event in events]

    started_count = statuses.count("STARTED")
    pass_count = statuses.count("PASS")
    fail_count = statuses.count("FAIL")
    terminal_count = pass_count + fail_count

    return {
        "model_load_count": started_count,
        "model_load_event_count": len(events),
        "started_count": started_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "terminal_count": terminal_count,
        "statuses": statuses,
        "single_successful_model_load": (
            statuses == ["STARTED", "PASS"]
            and started_count == 1
            and pass_count == 1
            and fail_count == 0
            and terminal_count == 1
        ),
    }


def failed_showcase_gate_lines(
    verdict_lines: Sequence[str],
) -> List[str]:
    """Return every required ``[GATE] ...=FAIL`` verdict line, else []."""
    return [
        line
        for line in verdict_lines
        if line.startswith("[GATE] ")
        and "=FAIL" in line
    ]


def _record_load_event(load_events: List[Dict[str, Any]], status: str, error: str = "") -> None:
    event: Dict[str, Any] = {
        "event": "showcase_model_load",
        "stage": "lane_b",
        "status": status,
        "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if error:
        event["error"] = error
    load_events.append(event)


def run_public_showcase(
    project_root: Optional[str] = None,
    output_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the Lane B production showcase on Tesla T4 x2.

    Canonical-only surface: no prompt/seed/steps/cfg/resolution/case overrides
    are accepted, mirroring the Lane A runner. Performs one independent model
    load (never sharing Lane A) and executes a deterministic multi-case
    schedule whose evidence lands under
    ``<output_root>/artifacts/public-showcase/<run_id>/``.
    """
    import torch

    from mage_t4x2 import constants as C
    from mage_t4x2.contracts import RunContract
    from mage_t4x2.dtype_audit import audit_runtime_dtypes
    from mage_t4x2.environment import environment_summary
    from mage_t4x2.evidence import EvidenceRun, generate_run_id, write_json_doc
    from mage_t4x2.model_provenance import ModelPathResolver
    from mage_t4x2.runtime_baseline import verify_runtime_baseline
    from mage_t4x2.sdpa_contract import assert_sdpa_frozen, freeze_sdpa_env
    from mage_t4x2.source_pin import (
        current_dependency_authority,
        pinned_source_authority,
        render_dependency_authority_text,
    )
    from mage_t4x2.telemetry import TelemetryRecorder
    from public_demo.contract import require_public_demo_model_path, require_public_demo_model_source
    from scripts import gpu_session as gs

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
    if not (root / "mage_t4x2").is_dir():
        raise RuntimeError(f"project root is invalid: {root}")

    required_model_files = [
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ]

    run_id = run_id or generate_run_id("public-showcase")
    out_dir = Path(output_root or root / SHOWCASE_ARTIFACTS_REL) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = out_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir = out_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "status": "NOT_RUN",
        "lane": "B",
        "run_id": run_id,
        "model_id": CANONICAL.model_id,
        "output_root": str(out_dir),
        "showcase_steps": SHOWCASE_STEPS,
        "showcase_cfg": SHOWCASE_CFG,
        "target_showcase_inference_seconds": TARGET_SHOWCASE_INFERENCE_SECONDS,
        "minimum_showcase_cases": MIN_SHOWCASE_CASES,
        "maximum_showcase_cases": MAX_SHOWCASE_CASES,
        "stop_reason": "failure",
        "executed_case_ids": [],
        "scheduled_case_ids": [],
        "expected_outputs": {},
        "case_catalog": build_showcase_case_catalog(SHOWCASE_CASES),
        "first_failed_gate": None,
        "error": None,
    }
    verdict_lines: List[str] = []
    load_events: List[Dict[str, Any]] = []
    load_evidence_path = out_dir / "model-load-events.json"
    results: List[Dict[str, Any]] = []
    inference_ids: List[str] = []
    routing_facts: Dict[str, Any] = {}
    state = None
    start_clock = time.perf_counter()

    def record_gate(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        verdict_lines.append(f"[GATE] {name}={status}" + (f" {detail}" if detail else ""))

    def persist_load_events() -> None:
        evidence = summarize_showcase_model_load_events(load_events)
        write_json_doc(
            str(load_evidence_path),
            {
                "model_load_count": evidence["model_load_count"],
                "model_load_event_count": evidence["model_load_event_count"],
                "started_count": evidence["started_count"],
                "pass_count": evidence["pass_count"],
                "fail_count": evidence["fail_count"],
                "terminal_count": evidence["terminal_count"],
                "single_successful_model_load": evidence["single_successful_model_load"],
                "statuses": evidence["statuses"],
                "events": load_events,
            },
        )

    def load_once(selected: str, contract: Any, source_authority: Dict[str, Any]) -> Any:
        _record_load_event(load_events, "STARTED")
        persist_load_events()
        try:
            loaded = gs.load_runtime_state_dual_t4_spread(selected, contract, source_authority)
        except Exception as exc:
            _record_load_event(load_events, "FAIL", error=f"{type(exc).__name__}: {exc}")
            persist_load_events()
            raise PublicShowcaseGateFailure(
                "SHOWCASE_MODEL_LOAD", f"{type(exc).__name__}: {exc}"
            ) from exc
        _record_load_event(load_events, "PASS")
        persist_load_events()
        return loaded

    try:
        table_errors = showcase_case_table_errors()
        if table_errors:
            raise PublicShowcaseGateFailure("SHOWCASE_TABLE", "; ".join(table_errors))

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise PublicShowcaseGateFailure(
                "SHOWCASE_T4X2_REQUIRED",
                "Lane B requires two Tesla T4 GPUs; no model load is attempted",
            )
        pre = gs.preflight_gpu(require_t4x2=False)
        if not pre.get("t4x2_ok"):
            raise PublicShowcaseGateFailure(
                "SHOWCASE_T4X2_PREFLIGHT", "exactly two Tesla T4 GPUs are required"
            )
        record_gate("SHOWCASE_T4X2_PREFLIGHT", True, f"gpu_count={pre.get('device_count')}")

        baseline = verify_runtime_baseline(root)
        if baseline["status"] != "PASS":
            raise PublicShowcaseGateFailure(
                "SHOWCASE_RUNTIME_BASELINE", str(baseline.get("errors"))
            )
        record_gate(
            "SHOWCASE_RUNTIME_BASELINE", True, f"rows={len(baseline.get('entries', []))}"
        )

        freeze_sdpa_env()
        sdpa = assert_sdpa_frozen()
        if sdpa.get("status") != "PASS":
            raise PublicShowcaseGateFailure("SHOWCASE_SDPA_FROZEN", str(sdpa))
        record_gate(
            "SHOWCASE_SDPA_FROZEN", True, f"vf_hf_attn_impl={sdpa.get('vf_hf_attn_impl')}"
        )

        canonical_model = str(CANONICAL.model_path)
        resolver = ModelPathResolver(
            candidates=[canonical_model], required_rel_files=required_model_files
        )
        resolution_doc = resolver.resolve(fallback_id=None)
        selected = resolution_doc["selected_path"]
        try:
            require_public_demo_model_source(resolution_doc)
            if not selected or not resolution_doc["required_files_present"]:
                raise PublicShowcaseGateFailure(
                    "SHOWCASE_MODEL_PATH", "no valid owner-qualified local model path found"
                )
            require_public_demo_model_path(str(selected))
        except PublicShowcaseGateFailure:
            raise
        except Exception as exc:
            raise PublicShowcaseGateFailure("SHOWCASE_MODEL_PATH", str(exc)) from exc
        record_gate("SHOWCASE_MODEL_PATH", True, f"selected={selected}")

        contract = RunContract(
            model=CANONICAL.model_id,
            framework=C.FRAMEWORK,
            task=C.TASK,
            resolution=(CANONICAL.height, CANONICAL.width),
            seed=CANONICAL.seed,
            steps=CANONICAL.steps,
            cfg=CANONICAL.cfg_scale,
            accelerator=C.ACCELERATOR,
            cpu_fallback=C.CPU_FALLBACK,
            stable_diffusion_cpp=C.STABLE_DIFFUSION_CPP_ALLOWED,
            sd_cli=C.SD_CLI_ALLOWED,
        )
        contract_errors = contract.validate()
        if contract_errors:
            raise PublicShowcaseGateFailure("SHOWCASE_CONTRACT", "; ".join(contract_errors))
        record_gate("SHOWCASE_CONTRACT", True, f"seed={CANONICAL.seed} steps={CANONICAL.steps} cfg={CANONICAL.cfg_scale}")

        source_authority = pinned_source_authority().to_dict()
        run = EvidenceRun(str(out_dir), source_authority=source_authority)
        run.write_run_contract(contract.to_dict())
        run.write_environment(environment_summary())
        run.write_source_authority(source_authority)
        run.write_dependency_authority_text(render_dependency_authority_text(current_dependency_authority()))
        run.write_gpu_inventory(pre.get("inventory") or [])

        load_start = time.perf_counter()
        state = load_once(selected, contract, source_authority)
        load_seconds = time.perf_counter() - load_start
        load_evidence = summarize_showcase_model_load_events(load_events)
        if not load_evidence["single_successful_model_load"]:
            raise PublicShowcaseGateFailure(
                "SHOWCASE_MODEL_LOAD",
                (
                    f"model_load_count={load_evidence['model_load_count']} "
                    f"model_load_event_count={load_evidence['model_load_event_count']} "
                    f"statuses={load_evidence['statuses']}"
                ),
            )
        record_gate(
            "SHOWCASE_MODEL_LOAD",
            True,
            "model_load_count=1 lifecycle_events=2",
        )

        dtype_after_load = audit_runtime_dtypes(state)
        run.write_dtype("before", dtype_after_load)
        if dtype_after_load.get("overall_status") != "PASS":
            raise PublicShowcaseGateFailure("SHOWCASE_BF16_AFTER_LOAD", str(dtype_after_load))
        record_gate("SHOWCASE_BF16_AFTER_LOAD", True)

        plan_result = gs.apply_dual_t4_mixed(
            state, CANONICAL.num_transformer_blocks, split_block=CANONICAL.split_block
        )
        if not plan_result.get("ok"):
            raise PublicShowcaseGateFailure("SHOWCASE_DUAL_T4_PLACEMENT", str(plan_result))
        record_gate("SHOWCASE_DUAL_T4_PLACEMENT", True)
        device_plan = plan_result["device_plan"]
        run.write_device_map(device_plan.get("device_map_json") or {})
        plan_result.pop("prior", None)

        post_placement_sdpa = assert_sdpa_frozen()
        if post_placement_sdpa.get("status") != "PASS":
            raise PublicShowcaseGateFailure("SHOWCASE_SDPA_POST_PLACEMENT", str(post_placement_sdpa))
        record_gate("SHOWCASE_SDPA_POST_PLACEMENT", True)

        cumulative_inference_seconds = 0.0
        stop_reason = "failure"

        for case in SHOWCASE_CASES:
            if not should_execute_next_showcase_case(
                len(results), cumulative_inference_seconds
            ):
                break
            case_id = str(case["id"])
            category = str(case["category"])
            resolution = int(case["width"])
            prompt = str(case["prompt"])
            seed = int(case["seed"])
            inference_id = f"showcase-{run_id}-{case_id}"
            inference_ids.append(inference_id)
            output_rel = showcase_output_name(case_id, resolution, seed)
            output_png = outputs_dir / output_rel

            with TelemetryRecorder(
                str(showcase_telemetry_path(telemetry_dir, case_id)),
                run_id=run_id,
                phase="showcase",
                inference_id=inference_id,
            ) as recorder:
                memory_before0 = int(torch.cuda.memory_allocated("cuda:0"))
                memory_before1 = int(torch.cuda.memory_allocated("cuda:1"))
                torch.cuda.reset_peak_memory_stats("cuda:0")
                torch.cuda.reset_peak_memory_stats("cuda:1")
                torch.cuda.synchronize()
                case_start = time.perf_counter()

                adapter = None
                try:
                    adapter = gs.attach_dual_adapter(
                        state,
                        device_plan,
                        telemetry=recorder,
                        run_id=run_id,
                        phase="showcase",
                        inference_id=inference_id,
                    )
                    images = state.pipeline.generate(
                        [prompt],
                        steps=SHOWCASE_STEPS,
                        cfg=SHOWCASE_CFG,
                        heights=[resolution],
                        widths=[resolution],
                        seeds=[seed],
                    )
                finally:
                    gs.detach_dual_adapter(state, adapter, reason="showcase_case_end")

                torch.cuda.synchronize()
                case_elapsed = time.perf_counter() - case_start
                adapters_detached = (
                    getattr(state, "dual_forward_adapter", None) is None
                    and getattr(state, "vae_input_bridge", None) is None
                )

                image = images[0]
                output_png.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_png, format="PNG")
                validation = validate_image(str(output_png), expected_size=(resolution, resolution))
                validation_ok = bool(
                    validation.get("valid")
                    and validation.get("width") == resolution
                    and validation.get("height") == resolution
                    and validation.get("rgb_valid")
                    and not validation.get("has_nan")
                    and not validation.get("has_inf")
                )
                recorder.metric("case_elapsed_seconds", round(case_elapsed, 6))
                recorder.event(
                    "phase",
                    phase="case",
                    status="PASS",
                    case_id=case_id,
                    category=category,
                    resolution=resolution,
                    seed=seed,
                )

            cumulative_inference_seconds += case_elapsed
            results.append(
                {
                    "inference_id": inference_id,
                    "case_id": case_id,
                    "category": category,
                    "prompt": prompt,
                    "resolution": resolution,
                    "seed": seed,
                    "elapsed_seconds": round(case_elapsed, 6),
                    "cumulative_seconds": round(cumulative_inference_seconds, 6),
                    "output_rel": output_rel,
                    "output_sha256": sha256_file(str(output_png)),
                    "validation_ok": validation_ok,
                    "adapter_detached": bool(adapters_detached),
                    "export_ok": True,
                    "peak_memory_gpu0_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
                    "peak_memory_reserved_gpu0_bytes": int(torch.cuda.max_memory_reserved("cuda:0")),
                    "peak_memory_gpu1_bytes": int(torch.cuda.max_memory_allocated("cuda:1")),
                    "peak_memory_reserved_gpu1_bytes": int(torch.cuda.max_memory_reserved("cuda:1")),
                    "memory_before_gpu0_bytes": memory_before0,
                    "memory_before_gpu1_bytes": memory_before1,
                }
            )

        if len(results) >= MAX_SHOWCASE_CASES:
            stop_reason = "max_cases_reached"
        elif cumulative_inference_seconds >= TARGET_SHOWCASE_INFERENCE_SECONDS:
            stop_reason = "target_inference_seconds_reached"

        routing_facts = build_showcase_routing_facts(
            read_showcase_telemetry(telemetry_dir),
            run_id=run_id,
            inference_ids=inference_ids,
            num_blocks=CANONICAL.num_transformer_blocks,
            expected_invocations=SHOWCASE_STEPS,
        )
        if not derive_case_route_ok(routing_facts):
            raise PublicShowcaseGateFailure("SHOWCASE_ROUTING", str(routing_facts))
        record_gate("SHOWCASE_ROUTING", True, f"cases={len(results)}")

        if len(results) < MIN_SHOWCASE_CASES:
            raise PublicShowcaseGateFailure(
                "SHOWCASE_MIN_CASES",
                f"expected at least {MIN_SHOWCASE_CASES} executed cases, got {len(results)}",
            )
        record_gate("SHOWCASE_MIN_CASES", True, f"executed={len(results)}")

        if not all(r["validation_ok"] and r["adapter_detached"] for r in results):
            raise PublicShowcaseGateFailure(
                "SHOWCASE_OUTPUT_VALIDATION", "one or more case outputs invalid"
            )
        record_gate("SHOWCASE_OUTPUT_VALIDATION", True)

        run.write_dtype("after", audit_runtime_dtypes(state))
        summary = finalize_showcase_evidence(results, routing_facts, summary)
        summary["aggregate"] = build_showcase_aggregate(
            summary["case_results"],
            wall_seconds=time.perf_counter() - start_clock,
            load_seconds=load_seconds,
        )
        summary["stop_reason"] = stop_reason
        summary["model_load_count"] = load_evidence["model_load_count"]
        summary["model_load_event_count"] = load_evidence["model_load_event_count"]
        summary["single_successful_model_load"] = load_evidence["single_successful_model_load"]

        annotations = [
            f"{r['case_id']} {int(r['resolution'])}x{int(r['resolution'])} "
            f"seed{int(r['seed'])} {float(r['elapsed_seconds']):.1f}s"
            for r in results
        ]
        gallery_info = build_gallery(
            [str(outputs_dir / r["output_rel"]) for r in results],
            str(out_dir / "gallery.png"),
            annotations=annotations,
        )
        record_gate("SHOWCASE_GALLERY", True, f"images={gallery_info['count']}")
        summary["gallery"] = gallery_info

        required_digests = {r["output_rel"]: r["output_sha256"] for r in results}
        digest_mismatches = verify_digest_matches(str(outputs_dir), required_digests)
        if digest_mismatches:
            raise PublicShowcaseGateFailure(
                "SHOWCASE_DIGEST_VERIFY", "; ".join(digest_mismatches)
            )
        record_gate("SHOWCASE_DIGEST_VERIFY", True)

        failed = failed_showcase_gate_lines(verdict_lines)
        if failed:
            raise PublicShowcaseGateFailure(
                "SHOWCASE_REQUIRED_GATE_INVARIANT",
                "; ".join(failed),
            )

        summary["status"] = "PASS"
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "PASS"

    except PublicShowcaseGateFailure as exc:
        summary["status"] = "FAIL"
        summary["stop_reason"] = "failure"
        summary["first_failed_gate"] = exc.gate
        summary["error"] = str(exc)
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "FAIL"
        record_gate(exc.gate, False, str(exc))
        if len(results) >= MIN_SHOWCASE_CASES:
            routing_facts = build_showcase_routing_facts(
                read_showcase_telemetry(telemetry_dir),
                run_id=run_id,
                inference_ids=inference_ids,
                num_blocks=CANONICAL.num_transformer_blocks,
            )
            summary = finalize_showcase_evidence(results, routing_facts, summary)
            summary["status"] = "FAIL"
            summary[SHOWCASE_FINAL_VERDICT_KEY] = "FAIL"
    except Exception as exc:
        summary["status"] = "FAIL"
        summary["stop_reason"] = "failure"
        if not summary.get("first_failed_gate"):
            summary["first_failed_gate"] = "SHOWCASE_UNHANDLED_EXCEPTION"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "FAIL"
        record_gate(summary["first_failed_gate"], False, summary["error"])
    finally:
        if state is not None:
            try:
                gs.detach_dual_adapter(state, reason="showcase_teardown")
            except Exception:
                pass
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sync_showcase_execution_scope(summary, results)
    persist_showcase_evidence(out_dir, summary, verdict_lines)
    return {
        "status": summary["status"],
        "run_id": run_id,
        "output_dir": str(out_dir),
        "summary_path": str(out_dir / "showcase-summary.json"),
        "verdict_lines": verdict_lines,
        "first_failed_gate": summary.get("first_failed_gate"),
        "error": summary.get("error"),
    }


def persist_showcase_evidence(
    run_dir: Path,
    summary: Dict[str, Any],
    verdict_lines: List[str],
) -> Dict[str, Any]:
    """Fail-closed evidence persistence for the Lane B showcase.

    Writes ``showcase-summary.json`` before building the manifest (so the
    summary itself is manifest-covered), then builds the evidence manifest. If
    the manifest build fails, the status is flipped to FAIL, the corrected
    summary is persisted immediately, and the runtime log records the final
    corrected ``PUBLIC_SHOWCASE_FINAL_VERDICT`` so no surviving document keeps
    a stale PASS snapshot.
    """
    from mage_t4x2.evidence import write_json_doc
    from mage_t4x2.hashing import build_manifest

    if summary.get("status") != "PASS":
        summary["stop_reason"] = "failure"

    write_json_doc(str(run_dir / "showcase-summary.json"), summary)
    case_results = summary.get("case_results")
    if case_results:
        write_json_doc(str(run_dir / "case-results.json"), case_results)
    else:
        write_json_doc(str(run_dir / "case-results.json"), [])

    def manifest_names() -> List[str]:
        names = sorted(p.name for p in run_dir.iterdir() if p.is_file())
        return [n for n in names if n not in ("manifest.json", "manifest.sha256")]

    try:
        build_manifest(str(run_dir), manifest_names())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        verdict_lines.append(f"[GATE] SHOWCASE_MANIFEST_BUILD=FAIL {error}")
        summary["status"] = "FAIL"
        summary["stop_reason"] = "failure"
        summary["first_failed_gate"] = "SHOWCASE_MANIFEST_BUILD"
        summary["error"] = error
        summary[SHOWCASE_FINAL_VERDICT_KEY] = "FAIL"
        write_json_doc(str(run_dir / "showcase-summary.json"), summary)
    else:
        verdict_lines.append("[GATE] SHOWCASE_MANIFEST_BUILD=PASS")

    verdict_lines.append(
        f"PUBLIC_SHOWCASE_FINAL_VERDICT={summary.get(SHOWCASE_FINAL_VERDICT_KEY, summary.get('status', 'FAIL'))}"
    )
    (run_dir / "showcase-runtime.log").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    return summary