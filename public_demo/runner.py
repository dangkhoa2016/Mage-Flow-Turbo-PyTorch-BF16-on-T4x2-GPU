from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mage_t4x2 import constants as C

from .bootstrap import PublicBootstrapError, prepare_public_bootstrap, public_runtime_root
from .gates import routing_cardinality_ok, routing_integrity_ok
from .live_execution import execute_live_inference

CANONICAL_PROMPT = "a red fox in a snowy forest at golden hour, high detail"
DEFAULT_NUM_BLOCKS = 12
DEFAULT_SPLIT_BLOCK = 1
REQUIRED_MODEL_FILES = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
]


class PublicDemoGateFailure(RuntimeError):
    def __init__(self, gate: str, message: str) -> None:
        self.gate = gate
        super().__init__(f"{gate}: {message}")


def _resolve_root(project_root: Optional[str] = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
    if not (root / "mage_t4x2").is_dir() or not (root / "scripts").is_dir():
        raise RuntimeError(f"project root is invalid: {root}")
    return root


def _required_checks(summary: Dict[str, Any]) -> bool:
    output = summary.get("output") or {}
    routing = summary.get("routing") or {}
    runtime = summary.get("runtime") or {}
    precision = summary.get("precision") or {}
    safety = summary.get("safety") or {}
    return (
        runtime.get("model_load_count") == 1
        and routing_integrity_ok(routing)
        and precision.get("status") == "PASS"
        and safety.get("cpu_fallback_observed") is False
        and output.get("exists") is True
        and output.get("is_512x512") is True
        and output.get("rgb_valid") is True
        and output.get("nan") is False
        and output.get("inf") is False
    )


def run_public_demo(
    project_root: Optional[str] = None,
    model_path: Optional[str] = None,
    prompt: str = CANONICAL_PROMPT,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    split_block: int = DEFAULT_SPLIT_BLOCK,
    output_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    from mage_t4x2.contracts import RunContract
    from mage_t4x2.environment import environment_summary
    from mage_t4x2.evidence import EvidenceRun, generate_run_id, write_json_doc
    from mage_t4x2.hashing import build_manifest
    from mage_t4x2.model_provenance import MODEL_SOURCE_LOCAL, ModelPathResolver
    from mage_t4x2.r2g_runtime_baseline import verify_r2g_runtime_baseline
    from mage_t4x2.sdpa_contract import assert_sdpa_frozen, freeze_sdpa_env
    from mage_t4x2.source_pin import (
        current_dependency_authority,
        pinned_source_authority,
        render_dependency_authority_text,
    )

    root = _resolve_root(project_root)
    if prompt != CANONICAL_PROMPT:
        raise RuntimeError("PUBLIC_DEMO_PROMPT_MISMATCH: only the canonical prompt is allowed")

    run_id = run_id or generate_run_id("public-demo")
    out_dir = Path(output_root or root / "artifacts" / "public-demo") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = public_runtime_root(root, run_id)
    bootstrap_target = runtime_root / "bootstrap-site"

    summary: Dict[str, Any] = {
        "status": "NOT_RUN",
        "run_id": run_id,
        "bootstrap_target": str(bootstrap_target),
        "first_failed_gate": None,
        "error": None,
    }
    verdict_lines: List[str] = []

    def record_gate(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        verdict_lines.append(f"[GATE] {name}={status}" + (f" {detail}" if detail else ""))

    try:
        baseline = verify_r2g_runtime_baseline(root)
        rows = len(baseline.get("entries", []))
        record_gate("R2G_RUNTIME_BASELINE_INTEGRITY", baseline["status"] == "PASS", f"rows={rows}")
        if baseline["status"] != "PASS":
            raise PublicDemoGateFailure("R2G_RUNTIME_BASELINE_INTEGRITY", str(baseline.get("errors")))

        freeze_sdpa_env()
        sdpa = assert_sdpa_frozen()
        record_gate("SDPA_FROZEN", sdpa.get("status") == "PASS", f"vf_hf_attn_impl={sdpa.get('vf_hf_attn_impl')}")
        if sdpa.get("status") != "PASS":
            raise PublicDemoGateFailure("SDPA_FROZEN", str(sdpa))

        if runtime_root.exists():
            raise PublicDemoGateFailure(
                "STALE_PUBLIC_RUNTIME_ROOT",
                "pre-existing public runtime root must fail closed and is never deleted/reused",
            )
        bootstrap = prepare_public_bootstrap(root, run_id)
        for gate_name in (
            "BOOTSTRAP_REQUIREMENTS_LOCK",
            "BOOTSTRAP_WHEELHOUSE_INTEGRITY",
            "BOOTSTRAP_LOCAL_SITE_FRESH",
            "BOOTSTRAP_LOCAL_PROVISION",
            "BOOTSTRAP_LOCAL_VERSION_VERIFY",
            "BOOTSTRAP_LOCAL_ORIGIN_VERIFY",
        ):
            ok = bootstrap.bootstrap_env.get(gate_name) == "PASS"
            record_gate(gate_name, ok, gate_name)
            if not ok:
                raise PublicDemoGateFailure(gate_name, str(bootstrap.bootstrap_env))
        record_gate("UPSTREAM_BOOTSTRAP", True, bootstrap.upstream.source_root)

        canonical_model = "/kaggle/input/models/" + C.MODEL_ID + "/pytorch/default/1"
        resolver = ModelPathResolver(candidates=[canonical_model], required_rel_files=REQUIRED_MODEL_FILES)
        resolution = resolver.resolve(fallback_id=None)
        selected = model_path or resolution["selected_path"]
        if resolution["model_source"] != MODEL_SOURCE_LOCAL and not model_path:
            raise PublicDemoGateFailure("MODEL_PATH", "owner-qualified local Kaggle model attachment required")
        if not selected:
            raise PublicDemoGateFailure("MODEL_PATH", "no valid owner-qualified model path found")
        record_gate("MODEL_PATH", True, f"selected={selected}")

        contract = RunContract(
            model=C.MODEL_ID,
            framework=C.FRAMEWORK,
            task=C.TASK,
            resolution=(int(height or C.RESOLUTION_HEIGHT), int(width or C.RESOLUTION_WIDTH)),
            seed=int(seed or C.SEED),
            steps=int(steps or C.STEPS),
            cfg=float(cfg_scale or C.CFG),
            accelerator=C.ACCELERATOR,
            cpu_fallback=C.CPU_FALLBACK,
            stable_diffusion_cpp=C.STABLE_DIFFUSION_CPP_ALLOWED,
            sd_cli=C.SD_CLI_ALLOWED,
        )
        errors = contract.validate()
        if errors:
            raise PublicDemoGateFailure("CONTRACT", str(errors))
        record_gate("CONTRACT", True, f"steps={contract.steps}")

        source_authority = pinned_source_authority().to_dict()
        run = EvidenceRun(str(out_dir), source_authority=source_authority)
        run.write_run_contract(contract.to_dict())
        run.write_environment(environment_summary())
        run.write_source_authority(source_authority)
        run.write_dependency_authority_text(render_dependency_authority_text(current_dependency_authority()))

        live = execute_live_inference(
            out_dir=out_dir,
            run=run,
            contract=contract,
            model_path=selected,
            source_authority=source_authority,
            run_id=run_id,
            num_blocks=num_blocks,
            split_block=split_block,
            record_gate=record_gate,
        )
        summary.update(live)
        run.write_gpu_inventory([
            {"index": 0, "name": live["hardware"]["gpu0"]},
            {"index": 1, "name": live["hardware"]["gpu1"]},
        ])

        cardinality = routing_cardinality_ok(live["routing"], int(contract.steps))
        if not cardinality:
            raise PublicDemoGateFailure("ROUTING_CARDINALITY", str(live["routing"]))
        if not _required_checks(summary):
            raise PublicDemoGateFailure("OUTPUT_OR_ROUTING_VALIDATION", "mandatory live checks did not all pass")
        summary["status"] = "PASS"

    except PublicDemoGateFailure as exc:
        summary["status"] = "FAIL"
        summary["first_failed_gate"] = exc.gate
        summary["error"] = str(exc)
        record_gate(exc.gate, False, str(exc))
    except PublicBootstrapError as exc:
        summary["status"] = "FAIL"
        summary["first_failed_gate"] = "PUBLIC_BOOTSTRAP"
        summary["error"] = str(exc)
        record_gate("PUBLIC_BOOTSTRAP", False, str(exc))
    except Exception as exc:
        summary["status"] = "FAIL"
        if not summary.get("first_failed_gate"):
            summary["first_failed_gate"] = "UNHANDLED_EXCEPTION"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        record_gate(summary["first_failed_gate"], False, summary["error"])

    write_json_doc(str(out_dir / "summary.json"), summary)
    write_json_doc(str(out_dir / "dtype-summary.json"), summary.get("precision", {}))
    write_json_doc(str(out_dir / "routing-summary.json"), summary.get("routing", {}))
    if not (out_dir / "device-map.json").exists():
        write_json_doc(str(out_dir / "device-map.json"), summary.get("placement", {}))
    try:
        EvidenceRun(str(out_dir)).write_acceptance(summary)
    except Exception:
        pass
    try:
        build_manifest(str(out_dir), sorted(p.name for p in out_dir.iterdir() if p.is_file()))
    except Exception:
        pass

    verdict_lines.append(f"PUBLIC_DEMO_FINAL_VERDICT={summary['status']}")
    (out_dir / "runtime.log").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    return {
        "status": summary["status"],
        "run_id": run_id,
        "output_dir": str(out_dir),
        "summary_path": str(out_dir / "summary.json"),
        "verdict_lines": verdict_lines,
        "first_failed_gate": summary.get("first_failed_gate"),
        "error": summary.get("error"),
    }
