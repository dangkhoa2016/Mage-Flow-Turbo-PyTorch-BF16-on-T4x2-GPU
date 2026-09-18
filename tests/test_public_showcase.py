import ast
import inspect
import json
from pathlib import Path

import pytest

from public_demo import showcase as showcase
from public_demo.showcase import PublicShowcaseGateFailure


# --------------------------------------------------------------------------- #
# Source-level contract of the Lane B showcase module.
# --------------------------------------------------------------------------- #


def test_showcase_signature_is_canonical_only():
    source = Path("public_demo/showcase.py").read_text()
    signature = source.split("def run_public_showcase(")[1].split(") -> Dict[str, Any]:")[0]
    assert "model_path" not in signature
    assert "prompt" not in signature
    assert "seed" not in signature
    assert "steps" not in signature
    assert "cfg_scale" not in signature
    assert "width" not in signature
    assert "height" not in signature
    assert "num_blocks" not in signature
    assert "split_block" not in signature


def test_showcase_imports_no_torch_or_pil_at_module_top():
    source = Path("public_demo/showcase.py").read_text()
    header = source.split("\nclass ", 1)[0]
    for line in header.splitlines():
        if line.startswith(("import ", "from ")):
            assert "torch" not in line
            assert "PIL" not in line


def test_showcase_module_declares_independent_lane_b_load():
    source = Path("public_demo/showcase.py").read_text()
    assert "never share a single model load" in showcase.__doc__
    assert 'load_runtime_state_dual_t4_spread' in source
    assert "gs.run_t2i" not in source
    assert "state.pipeline.generate" in source


def test_showcase_uses_owner_qualified_model_path_and_canonical_contract():
    source = Path("public_demo/showcase.py").read_text()
    assert "require_public_demo_model_path" in source
    assert "require_public_demo_model_source" in source
    assert "fallback_id=None" in source
    assert "contract.validate()" in source
    assert "CANONICAL.model_id" in source
    assert "CANONICAL.seed" in source
    assert "CANONICAL.steps" in source
    assert "CANONICAL.cfg_scale" in source
    assert "CANONICAL.num_transformer_blocks" in source


def test_showcase_writes_dual_t4_evidence_lifecycle():
    source = Path("public_demo/showcase.py").read_text()
    for marker in (
        "EvidenceRun(",
        "write_run_contract",
        "write_environment",
        "write_source_authority",
        "write_dependency_authority_text",
        "write_gpu_inventory",
        "write_dtype(\"before\"",
        "write_dtype(\"after\"",
        "write_device_map",
    ):
        assert marker in source


# --------------------------------------------------------------------------- #
# Deterministic case table.
# --------------------------------------------------------------------------- #


def test_showcase_case_table_contract():
    assert showcase.SHOWCASE_CASES
    assert len(showcase.SHOWCASE_CASES) == showcase.MAX_SHOWCASE_CASES == 24
    assert showcase.MIN_SHOWCASE_CASES == 12
    assert showcase.SHOWCASE_STEPS == 4
    assert showcase.SHOWCASE_CFG == 1.0
    assert showcase.SHOWCASE_RESOLUTIONS == (512, 768, 1024)
    assert showcase.showcase_case_table_errors() == []
    ids = showcase.showcase_case_ids()
    assert ids == [f"s{i:02d}" for i in range(1, 25)]
    for case in showcase.SHOWCASE_CASES:
        assert isinstance(case["prompt"], str) and case["prompt"]
        assert isinstance(case["category"], str) and case["category"]
        assert int(case["width"]) == int(case["height"])
        assert int(case["width"]) in showcase.SHOWCASE_RESOLUTIONS


def test_showcase_prompt_uniqueness():
    prompts = [
        " ".join(str(c["prompt"]).lower().split())
        for c in showcase.SHOWCASE_CASES
    ]
    assert len(prompts) == showcase.MAX_SHOWCASE_CASES == 24
    assert len(set(prompts)) == 24


def test_showcase_prompt_specs_are_fixed_deterministic_strings():
    assert len(showcase.SHOWCASE_PROMPT_SPECS) == 24
    for spec in showcase.SHOWCASE_PROMPT_SPECS:
        assert isinstance(spec["prompt"], str) and spec["prompt"]
        assert isinstance(spec["category"], str) and spec["category"]
    for index, case in enumerate(showcase.SHOWCASE_CASES):
        spec = showcase.SHOWCASE_PROMPT_SPECS[index]
        assert case["category"] == spec["category"]
        assert case["prompt"] == spec["prompt"]


def test_showcase_mandatory_semantic_diversity():
    mandatory = showcase.SHOWCASE_CASES[:12]
    assert len({c["category"] for c in mandatory}) == 12


def test_showcase_full_category_metadata():
    assert all(c.get("category") for c in showcase.SHOWCASE_CASES)
    assert all(c.get("prompt") for c in showcase.SHOWCASE_CASES)


def test_showcase_rejects_old_single_prefix_prompt_scheme():
    source = Path("public_demo/showcase.py").read_text()
    assert "_SHOWCASE_PROMPT_PREFIX" not in source
    for case in showcase.SHOWCASE_CASES:
        assert not str(case["prompt"]).endswith(f"({case['id']})")


def test_showcase_mandatory_and_extension_layout():
    mandatory = [c for c in showcase.SHOWCASE_CASES if c["role"] == "mandatory"]
    extension = [c for c in showcase.SHOWCASE_CASES if c["role"] == "extension"]
    assert [int(c["width"]) for c in mandatory] == [512] * 4 + [768] * 4 + [1024] * 4
    assert [int(c["width"]) for c in extension] == list(showcase.SHOWCASE_RESOLUTIONS) * 4
    seeds = [int(c["seed"]) for c in showcase.SHOWCASE_CASES]
    assert seeds == list(range(1001, 1025))


def test_showcase_output_names_are_seed_qualified():
    name = showcase.showcase_output_name("s01", 512, 1001)
    assert name == "s01-512x512-seed1001.png"
    assert name != showcase.showcase_output_name("s01", 512, 1002)


def test_showcase_case_lookup_raises_for_unknown():
    with pytest.raises(KeyError):
        showcase._case_by_id("s99")


# --------------------------------------------------------------------------- #
# Dynamic pure-inference duration policy.
# --------------------------------------------------------------------------- #


def test_should_execute_next_showcase_case_decision_matrix():
    cases = [
        (0, 0.0, True),
        (11, 999.0, True),
        (12, 299.9, True),
        (12, 300.0, False),
        (13, 310.0, False),
        (23, 100.0, True),
        (24, 100.0, False),
    ]
    for executed, cumulative, expected in cases:
        assert (
            showcase.should_execute_next_showcase_case(executed, cumulative) is expected
        ), (executed, cumulative, expected)


def test_should_execute_helper_is_pure_bookkeeping():
    params = list(inspect.signature(showcase.should_execute_next_showcase_case).parameters)
    assert params == ["executed_count", "cumulative_inference_seconds"]


def test_should_execute_helper_reads_no_wall_or_load_time():
    source = inspect.getsource(showcase.should_execute_next_showcase_case)
    assert "perf_counter" not in source
    assert "start_clock" not in source
    assert "time." not in source
    assert "model_load" not in source


def test_should_execute_policy_drives_target_reached_stop():
    assert showcase.should_execute_next_showcase_case(
        12, showcase.TARGET_SHOWCASE_INFERENCE_SECONDS
    ) is False


def test_dynamic_duration_regression_vs_precomputed_schedule():
    # The previous implementation precomputed the whole extension list before
    # any image existed, consuming wall-clock probes that included model load.
    # The V4.1 policy decides after each real case using pure inference only:
    # 12 mandatory cases done, cumulative real inference still under 300s must
    # continue regardless of how long setup took.
    assert showcase.should_execute_next_showcase_case(12, 280.0) is True
    assert showcase.should_execute_next_showcase_case(12, 320.0) is False
    source = Path("public_demo/showcase.py").read_text()
    assert "showcase_case_schedule" not in source
    assert "should_execute_next_showcase_case" in source


def test_build_case_output_validation_matches_case_table():
    case = showcase.SHOWCASE_CASES[0]
    expected = showcase.build_case_output_validation([case])
    assert expected["s01"]["output_filename"] == "s01-512x512-seed1001.png"
    assert expected["s01"]["expected_resolution"] == 512
    assert expected["s01"]["steps"] == showcase.SHOWCASE_STEPS
    assert expected["s01"]["cfg"] == showcase.SHOWCASE_CFG


# --------------------------------------------------------------------------- #
# Routing facts from telemetry evidence (shared Lane A adjudication).
# --------------------------------------------------------------------------- #


def _case_telemetry(inference_id, run_id, num_blocks=2, invocations=showcase.SHOWCASE_STEPS):
    records = []
    for _ in range(invocations):
        records.append(
            {"event": "block_forward", "block": 0, "device": "cuda:0",
             "inference_id": inference_id, "run_id": run_id}
        )
        records.append(
            {"event": "cross_device_transfer", "from": "cuda:0", "to": "cuda:1",
             "inference_id": inference_id, "run_id": run_id}
        )
        records.append(
            {"event": "block_forward", "block": 1, "device": "cuda:1",
             "inference_id": inference_id, "run_id": run_id}
        )
        records.append(
            {"event": "transformer_output_return_transfer", "from": "cuda:1", "to": "cuda:0",
             "inference_id": inference_id, "run_id": run_id}
        )
    return records


def test_showcase_routing_facts_derive_from_boundary_transfer_evidence():
    run_id = "run-1"
    inference_id = "showcase-run-1-s01"
    records = _case_telemetry(inference_id, run_id)
    boundaries = [
        r for r in records
        if r.get("event") == "cross_device_transfer" and r.get("from") == "cuda:0" and r.get("to") == "cuda:1"
    ]
    returns = [
        r for r in records
        if r.get("event") == "transformer_output_return_transfer" and r.get("from") == "cuda:1" and r.get("to") == "cuda:0"
    ]
    assert len(boundaries) == showcase.SHOWCASE_STEPS
    assert len(returns) == showcase.SHOWCASE_STEPS

    facts = showcase.build_showcase_routing_facts(
        records,
        run_id=run_id,
        inference_ids=[inference_id],
        num_blocks=2,
        expected_invocations=showcase.SHOWCASE_STEPS,
    )
    entry = facts["per_inference"][inference_id]
    assert entry["observed_invocations"] == showcase.SHOWCASE_STEPS
    assert entry["block_order_valid"]
    assert entry["no_skipped_blocks"]
    assert entry["no_duplicated_blocks_within_invocation"]
    assert entry["gpu0_participation"]
    assert entry["gpu1_participation"]
    assert entry["single_t2i_instance"]
    assert entry["cpu_fallback_observed"] is False
    assert showcase.derive_case_route_ok(facts)


def test_showcase_routing_facts_fail_when_return_boundary_missing():
    run_id = "run-1"
    inference_id = "showcase-run-1-s01"
    records = _case_telemetry(inference_id, run_id)
    records.pop()  # drop the final 1->0 return: trailing unclosed invocation
    facts = showcase.build_showcase_routing_facts(
        records,
        run_id=run_id,
        inference_ids=[inference_id],
        num_blocks=2,
        expected_invocations=showcase.SHOWCASE_STEPS,
    )
    entry = facts["per_inference"][inference_id]
    assert entry["observed_invocations"] < showcase.SHOWCASE_STEPS
    assert not entry["block_order_valid"]
    assert not showcase.derive_case_route_ok(facts)


# --------------------------------------------------------------------------- #
# Evidence merger semantics (verdict can never be fabricated).
# --------------------------------------------------------------------------- #


def _passing_row(inference_id, case_id, resolution):
    spec = showcase._case_by_id(case_id)
    return {
        "inference_id": inference_id,
        "case_id": case_id,
        "category": str(spec["category"]),
        "prompt": str(spec["prompt"]),
        "resolution": resolution,
        "seed": int(case_id[1:]),
        "elapsed_seconds": 8.25,
        "cumulative_seconds": 8.25,
        "output_rel": "ignored.png",
        "output_sha256": "a" * 64,
        "peak_memory_gpu0_bytes": 100,
        "peak_memory_reserved_gpu0_bytes": 200,
        "peak_memory_gpu1_bytes": 300,
        "peak_memory_reserved_gpu1_bytes": 400,
    }


def _passing_facts(inference_ids):
    return {
        "per_inference": {
            inference_id: {
                "inference_id": inference_id,
                "observed_invocations": showcase.SHOWCASE_STEPS,
                "block_order_valid": True,
                "no_skipped_blocks": True,
                "no_duplicated_blocks_within_invocation": True,
                "gpu0_participation": True,
                "gpu1_participation": True,
                "single_t2i_instance": True,
                "cpu_fallback_observed": False,
                "block_sequence": [0, 1],
            }
            for inference_id, case_id in inference_ids
        }
    }


def _twelve_passing_cases():
    rows = []
    inference_ids = []
    for offset, case in enumerate(showcase.SHOWCASE_CASES[:12]):
        case_id = str(case["id"])
        inference_id = f"inf-{offset + 1:02d}"
        row = _passing_row(inference_id, case_id, int(case["width"]))
        row["validation_ok"] = True
        row["adapter_detached"] = True
        row["export_ok"] = True
        rows.append(row)
        inference_ids.append((inference_id, case_id))
    return rows, _passing_facts(inference_ids)


def test_finalize_verdict_not_run_without_evidence():
    summary = showcase.finalize_showcase_evidence([], {}, {"run_id": "r"})
    assert summary["cases_executed"] == 0
    assert summary["cases_passed"] == 0
    assert summary[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "NOT_RUN"


def test_finalize_verdict_pass_only_when_all_cases_green_and_minimum_met():
    rows, facts = _twelve_passing_cases()
    summary = showcase.finalize_showcase_evidence(rows, facts, {"run_id": "r"})
    assert summary["cases_executed"] == showcase.MIN_SHOWCASE_CASES
    assert summary["cases_passed"] == showcase.MIN_SHOWCASE_CASES
    assert summary[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "PASS"


def test_finalize_verdict_fail_below_minimum():
    rows, facts = _twelve_passing_cases()
    rows = rows[:3]
    restricted = {
        "per_inference": {
            k: v for k, v in facts["per_inference"].items()
            if v["inference_id"] in {r["inference_id"] for r in rows}
        }
    }
    summary = showcase.finalize_showcase_evidence(rows, restricted, {"run_id": "r"})
    assert summary["cases_executed"] == 3
    assert summary[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "FAIL"


def test_finalize_verdict_fail_when_any_case_invalid():
    rows, facts = _twelve_passing_cases()
    rows[0]["validation_ok"] = False
    summary = showcase.finalize_showcase_evidence(rows, facts, {"run_id": "r"})
    assert summary["cases_passed"] == showcase.MIN_SHOWCASE_CASES - 1
    assert summary[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "FAIL"


def test_finalize_verdict_fail_when_adapter_not_detached():
    rows, facts = _twelve_passing_cases()
    rows[3]["adapter_detached"] = False
    summary = showcase.finalize_showcase_evidence(rows, facts, {"run_id": "r"})
    assert summary["cases_passed"] == showcase.MIN_SHOWCASE_CASES - 1
    assert summary[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "FAIL"


def test_finalized_case_rows_keep_prompt_and_category():
    rows, facts = _twelve_passing_cases()
    summary = showcase.finalize_showcase_evidence(rows, facts, {"run_id": "r"})
    for row in summary["case_results"]:
        assert str(row["case_id"]).startswith("s")
        assert row["category"]
        assert row["prompt"]
        assert row["resolution"]
        assert row["seed"]
        assert row["prompt"] == str(showcase._case_by_id(row["case_id"])["prompt"])


# --------------------------------------------------------------------------- #
# Aggregate and digest evidence.
# --------------------------------------------------------------------------- #


def test_build_showcase_aggregate_uses_observed_facts_only():
    rows = [
        {"elapsed_seconds": 8.0, "resolution": 512,
         "peak_memory_gpu0_bytes": 100, "peak_memory_reserved_gpu0_bytes": 200,
         "peak_memory_gpu1_bytes": 300, "peak_memory_reserved_gpu1_bytes": 400},
        {"elapsed_seconds": 12.0, "resolution": 512,
         "peak_memory_gpu0_bytes": 150, "peak_memory_reserved_gpu0_bytes": 250,
         "peak_memory_gpu1_bytes": 350, "peak_memory_reserved_gpu1_bytes": 450},
        {"elapsed_seconds": 20.0, "resolution": 768,
         "peak_memory_gpu0_bytes": 200, "peak_memory_reserved_gpu0_bytes": 300,
         "peak_memory_gpu1_bytes": 400, "peak_memory_reserved_gpu1_bytes": 500},
    ]
    agg = showcase.build_showcase_aggregate(rows, wall_seconds=40.0, load_seconds=9.5)
    assert agg["total_images"] == 3
    assert agg["showcase_model_load_seconds"] == 9.5
    assert agg["images_per_minute"] == pytest.approx(3 * 60.0 / 40.0)
    assert agg["global_gpu1_peak_allocated"] == 400
    per_512 = agg["per_resolution"]["512x512"]
    assert per_512["count"] == 2
    assert per_512["median_seconds"] == (8.0 + 12.0) / 2
    assert per_512["min_seconds"] == 8.0
    assert agg["per_resolution"]["1024x1024"]["count"] == 0
    assert agg["per_resolution"]["1024x1024"]["median_seconds"] is None
    assert "observed_on" in agg


def test_verify_digest_matches(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    a = out / "a.png"
    a.write_bytes(b"abc")
    expected = showcase.sha256_file(str(a))
    assert showcase.verify_digest_matches(str(out), {"a.png": expected}) == []
    mismatches = showcase.verify_digest_matches(str(out), {"a.png": "0" * 64})
    assert len(mismatches) == 1
    assert "digest mismatch" in mismatches[0]
    assert "missing artifact: b.png" in showcase.verify_digest_matches(str(out), {"b.png": "0" * 64})


# --------------------------------------------------------------------------- #
# Gallery (CPU-safe Pillow contact sheet).
# --------------------------------------------------------------------------- #


def test_build_gallery_composes_contact_sheet(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    images = []
    for index in range(4):
        path = tmp_path / f"img-{index}.png"
        Image.new("RGB", (256, 256), (index * 40, 20, 200)).save(path, format="PNG")
        images.append(str(path))
    info = showcase.build_gallery(images, str(tmp_path / "gallery.png"), cols=4,
                                  annotations=["one", "two", "three", "four"])
    assert info["count"] == 4
    assert info["columns"] == 4
    assert len(info["sha256"]) == 64
    assert Path(info["path"]).is_file()


def test_build_gallery_rejects_empty_input(tmp_path):
    with pytest.raises(PublicShowcaseGateFailure) as exc:
        showcase.build_gallery([], str(tmp_path / "gallery.png"))
    assert exc.value.gate == "GALLERY"


# --------------------------------------------------------------------------- #
# Telemetry helpers.
# --------------------------------------------------------------------------- #


def test_showcase_telemetry_path_and_read_round_trip(tmp_path):
    from mage_t4x2.telemetry import TelemetryRecorder

    path = showcase.showcase_telemetry_path(tmp_path, "s01")
    assert path == tmp_path / "s01.jsonl"
    with TelemetryRecorder(str(path), run_id="run-1", phase="showcase", inference_id="inf-1") as telemetry:
        telemetry.event("probe", value=1)
        telemetry.event("probe", value=2)
    records = showcase.read_showcase_telemetry(tmp_path)
    assert len(records) == 2
    assert all(r["run_id"] == "run-1" for r in records)
    assert records[0]["inference_id"] == "inf-1"


# --------------------------------------------------------------------------- #
# Fail-closed evidence persistence.
# --------------------------------------------------------------------------- #


def test_persist_showcase_evidence_writes_manifest_covered_summary(tmp_path, monkeypatch):
    rows, facts = _twelve_passing_cases()
    summary = showcase.finalize_showcase_evidence(rows, facts, {"run_id": "r"})
    summary["status"] = "PASS"
    lines = ["[GATE] SHOWCASE_TABLE=PASS"]
    result = showcase.persist_showcase_evidence(tmp_path, summary, lines)
    assert (tmp_path / "showcase-summary.json").is_file()
    assert (tmp_path / "case-results.json").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "manifest.sha256").is_file()
    assert (tmp_path / "showcase-runtime.log").is_file()
    log = (tmp_path / "showcase-runtime.log").read_text()
    assert "[GATE] SHOWCASE_MANIFEST_BUILD=PASS" in log
    assert "PUBLIC_SHOWCASE_FINAL_VERDICT=PASS" in log
    assert result["status"] == "PASS"


def test_persist_showcase_evidence_fails_closed_on_manifest_error(tmp_path, monkeypatch):
    rows, facts = _twelve_passing_cases()
    summary = showcase.finalize_showcase_evidence(rows, facts, {"run_id": "r"})
    summary["status"] = "PASS"

    import mage_t4x2.hashing

    def broken(*args, **kwargs):
        raise RuntimeError("manifest build exploded")

    monkeypatch.setattr(mage_t4x2.hashing, "build_manifest", broken)
    result = showcase.persist_showcase_evidence(tmp_path, summary, [])
    assert result["status"] == "FAIL"
    assert result["stop_reason"] == "failure"
    assert result[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "FAIL"
    assert result["first_failed_gate"] == "SHOWCASE_MANIFEST_BUILD"
    stored = (tmp_path / "showcase-summary.json")
    assert stored.is_file()
    stored_doc = json.loads(stored.read_text())
    assert stored_doc["status"] == "FAIL"
    assert stored_doc["stop_reason"] == "failure"
    assert stored_doc[showcase.SHOWCASE_FINAL_VERDICT_KEY] == "FAIL"
    log = (tmp_path / "showcase-runtime.log").read_text()
    assert "[GATE] SHOWCASE_MANIFEST_BUILD=FAIL" in log
    assert "PUBLIC_SHOWCASE_FINAL_VERDICT=FAIL" in log


# --------------------------------------------------------------------------- #
# V4.2 — no duplicate top-level definitions (AST, not text grep).
# --------------------------------------------------------------------------- #


def test_showcase_has_no_duplicate_top_level_function_definitions():
    source = Path("public_demo/showcase.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    duplicates = sorted({
        name
        for name in names
        if names.count(name) > 1
    })

    assert duplicates == []
    assert names.count("_case_by_id") == 1


# --------------------------------------------------------------------------- #
# V4.2 — full 24-case plan catalog versus executed-only expectations.
# --------------------------------------------------------------------------- #


def test_showcase_case_catalog_covers_full_plan():
    catalog = showcase.build_showcase_case_catalog(
        showcase.SHOWCASE_CASES
    )

    assert list(catalog) == [
        f"s{i:02d}"
        for i in range(1, 25)
    ]

    assert len(catalog) == 24

    for case_id, row in catalog.items():
        assert row["case_id"] == case_id
        assert row["category"]
        assert row["prompt"]
        assert row["role"] in {"mandatory", "extension"}
        assert row["output_filename"].endswith(".png")


def test_sync_execution_scope_uses_completed_cases_only():
    summary = {
        "case_catalog": showcase.build_showcase_case_catalog(
            showcase.SHOWCASE_CASES
        )
    }

    results = [
        {
            "case_id": str(case["id"]),
        }
        for case in showcase.SHOWCASE_CASES[:12]
    ]

    showcase.sync_showcase_execution_scope(summary, results)

    assert summary["executed_case_ids"] == [
        f"s{i:02d}"
        for i in range(1, 13)
    ]

    assert summary["scheduled_case_ids"] == summary["executed_case_ids"]

    assert list(summary["expected_outputs"]) == [
        f"s{i:02d}"
        for i in range(1, 13)
    ]

    assert len(summary["case_catalog"]) == 24
    assert len(summary["expected_outputs"]) == 12


def test_sync_execution_scope_handles_no_completed_cases():
    summary = {}
    showcase.sync_showcase_execution_scope(summary, [])

    assert summary["executed_case_ids"] == []
    assert summary["scheduled_case_ids"] == []
    assert summary["expected_outputs"] == {}


def test_sync_execution_scope_handles_full_24_case_run():
    summary = {}

    results = [
        {"case_id": str(case["id"])}
        for case in showcase.SHOWCASE_CASES
    ]

    showcase.sync_showcase_execution_scope(summary, results)

    assert len(summary["executed_case_ids"]) == 24
    assert len(summary["expected_outputs"]) == 24


# --------------------------------------------------------------------------- #
# V4.2 — a FAIL summary can never keep a null or success stop reason.
# --------------------------------------------------------------------------- #


def test_persist_early_failure_cannot_keep_null_stop_reason(tmp_path):
    summary = {
        "status": "FAIL",
        "stop_reason": None,
        showcase.SHOWCASE_FINAL_VERDICT_KEY: "FAIL",
    }
    result = showcase.persist_showcase_evidence(tmp_path, summary, [])

    assert result["status"] == "FAIL"
    assert result["stop_reason"] == "failure"

    stored = json.loads(
        (tmp_path / "showcase-summary.json").read_text()
    )
    assert stored["status"] == "FAIL"
    assert stored["stop_reason"] == "failure"


def test_late_failure_overrides_prior_success_stop_reason(tmp_path):
    summary = {
        "status": "FAIL",
        "stop_reason": "target_inference_seconds_reached",
        showcase.SHOWCASE_FINAL_VERDICT_KEY: "FAIL",
    }
    result = showcase.persist_showcase_evidence(tmp_path, summary, [])

    assert result["status"] == "FAIL"
    assert result["stop_reason"] == "failure"

    stored = json.loads(
        (tmp_path / "showcase-summary.json").read_text()
    )
    assert stored["status"] == "FAIL"
    assert stored["stop_reason"] == "failure"


# --------------------------------------------------------------------------- #
# V4.2 — Pillow must stay lazy and never gate the empty-gallery path.
# --------------------------------------------------------------------------- #


def test_build_gallery_empty_gate_fires_before_pillow_lazy_import():
    source = Path("public_demo/showcase.py").read_text(encoding="utf-8")
    empty_gate = 'raise PublicShowcaseGateFailure("GALLERY"'
    lazy_pil = "from PIL import Image, ImageDraw, ImageFont"
    assert source.index(empty_gate) < source.index(lazy_pil)

    header = source.split("\nclass ", 1)[0]
    assert "PIL" not in header

    with pytest.raises(PublicShowcaseGateFailure) as exc:
        showcase.build_gallery([], "/tmp/docile-no-pillow-gate.png")
    assert exc.value.gate == "GALLERY"


# --------------------------------------------------------------------------- #
# V4.4 — model-load semantics: attempts are distinct from lifecycle events.
# --------------------------------------------------------------------------- #


def _load_events(*statuses):
    return [{"status": status} for status in statuses]


def test_v44_successful_single_load_semantics():
    doc = showcase.summarize_showcase_model_load_events(
        _load_events("STARTED", "PASS")
    )
    assert doc["model_load_count"] == 1
    assert doc["model_load_event_count"] == 2
    assert doc["started_count"] == 1
    assert doc["pass_count"] == 1
    assert doc["fail_count"] == 0
    assert doc["terminal_count"] == 1
    assert doc["statuses"] == ["STARTED", "PASS"]
    assert doc["single_successful_model_load"] is True


def test_v44_failed_single_load_not_successful():
    doc = showcase.summarize_showcase_model_load_events(
        _load_events("STARTED", "FAIL")
    )
    assert doc["model_load_count"] == 1
    assert doc["model_load_event_count"] == 2
    assert doc["single_successful_model_load"] is False


def test_v44_two_physical_load_attempts_rejected():
    doc = showcase.summarize_showcase_model_load_events(
        _load_events("STARTED", "PASS", "STARTED", "PASS")
    )
    assert doc["model_load_count"] == 2
    assert doc["model_load_event_count"] == 4
    assert doc["single_successful_model_load"] is False


def test_v44_partial_started_only_is_not_successful():
    doc = showcase.summarize_showcase_model_load_events(
        _load_events("STARTED")
    )
    assert doc["model_load_count"] == 1
    assert doc["model_load_event_count"] == 1
    assert doc["terminal_count"] == 0
    assert doc["single_successful_model_load"] is False


def test_v44_empty_events_not_successful():
    doc = showcase.summarize_showcase_model_load_events([])
    assert doc["model_load_count"] == 0
    assert doc["model_load_event_count"] == 0
    assert doc["single_successful_model_load"] is False


def test_v44_required_gate_fail_detected():
    lines = [
        "[GATE] SHOWCASE_MODEL_LOAD=PASS model_load_count=1 lifecycle_events=2",
        "[GATE] SOME_REQUIRED_GATE=FAIL something broke",
        "PUBLIC_SHOWCASE_FINAL_VERDICT=PASS",
    ]
    failed = showcase.failed_showcase_gate_lines(lines)
    assert failed == ["[GATE] SOME_REQUIRED_GATE=FAIL something broke"]


def test_v44_no_required_gate_fail_on_clean_lines():
    lines = [
        "[GATE] SHOWCASE_MODEL_LOAD=PASS",
        "PUBLIC_SHOWCASE_FINAL_VERDICT=PASS",
        "GALLERY_IMAGES=12",
    ]
    assert showcase.failed_showcase_gate_lines(lines) == []


def test_v44_failed_gate_lines_are_case_sensitive_prefix_constrained():
    assert showcase.failed_showcase_gate_lines(
        ["[GATE] X=PASS", "PUBLIC_SHOWCASE_FINAL_VERDICT=FAIL"]
    ) == []


def test_v44_source_never_uses_len_based_model_load_count():
    source = Path("public_demo/showcase.py").read_text(encoding="utf-8")
    assert '"model_load_count": len(load_events)' not in source
    assert "summary[\"model_load_count\"] = len(load_events)" not in source
    assert "summarize_showcase_model_load_events" in source


def test_v44_run_public_showcase_invokes_required_gate_invariant_before_pass():
    source = Path("public_demo/showcase.py").read_text(encoding="utf-8")
    assert "SHOWCASE_REQUIRED_GATE_INVARIANT" in source
    assert "failed_showcase_gate_lines(verdict_lines)" in source
    marker = 'summary[SHOWCASE_FINAL_VERDICT_KEY] = "PASS"'
    assert marker in source
    assert source.index("failed_showcase_gate_lines(verdict_lines)") < \
        source.rindex(marker)


def test_v44_source_requires_showcase_model_load_fail_closed():
    source = Path("public_demo/showcase.py").read_text(encoding="utf-8")
    gate_call = (
        'record_gate(\n'
        '            "SHOWCASE_MODEL_LOAD",\n'
        '            True,\n'
        '            "model_load_count=1 lifecycle_events=2",\n'
        '        )'
    )
    assert gate_call in source
    assert (
        "if not load_evidence[\"single_successful_model_load\"]:" in source
    )