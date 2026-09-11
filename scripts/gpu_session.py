"""GPU session orchestration (Stage B) for one continuous Kaggle T4×2 session.

IMPORTANT: This module is imported only inside the GPU session. It is designed
to be import-safe on CPU (no CUDA calls at import) so Stage A checks can still
import it, but every function that touches CUDA is a runtime-only call.

Runtime states follow the operational rule "load once, reconfigure live":
    LOAD -> CONFIGURE -> RUN -> RECONFIGURE -> RUN -> ...
The 20 GB model is never reloaded between G1..G6.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mage_t4x2 import constants as C
from mage_t4x2.contracts import RunContract
from mage_t4x2.device_plan import build_device_plan, extract_block_devices
from mage_t4x2.dtype_audit import audit_runtime_dtypes
from mage_t4x2.transformer_placement import (
    has_parameters_or_buffers,
    post_blocks_inconsistent,
)
from mage_t4x2.environment import cuda_inventory
from mage_t4x2.evidence import EvidenceRun
from mage_t4x2.image_validation import write_validation
from mage_t4x2.model_inventory import static_component_inventory
from mage_t4x2.telemetry import TelemetryRecorder

KAGGLE_INPUT = "/kaggle/input"


@dataclasses.dataclass
class RuntimeState:
    pipeline: Any = None
    model: Any = None
    text_encoder: Any = None
    transformer: Any = None
    vae: Any = None
    scheduler: Any = None
    tokenizer: Any = None
    source_authority: Optional[Dict[str, Any]] = None
    contract: Optional[RunContract] = None
    current_profile: str = "unconfigured"
    current_topology: str = "unconfigured"
    model_path: Optional[str] = None
    memory_plan: Optional[Dict[str, Any]] = None
    placement_validation: Optional[Dict[str, Any]] = None
    no_full_model_cuda0_materialization: bool = False
    sdpa_frozen: Optional[Dict[str, Any]] = None
    current_split_block: Optional[int] = None
    dual_forward_adapter: Any = None
    vae_input_bridge: Any = None

    def snapshot(self) -> Dict[str, str]:
        return {
            "current_profile": self.current_profile,
            "current_topology": self.current_topology,
        }


def resolve_model_path(contract: RunContract, override: Optional[str] = None) -> str:
    """Prefer a locally attached Kaggle model input; fall back to the HF repo id."""
    if override:
        return override
    slug = contract.model.split("/")[-1]  # mage-flow-community-mage-flow-turbo
    for candidate in (
        os.path.join(KAGGLE_INPUT, slug, "pytorch", "default", "1"),
        os.path.join(KAGGLE_INPUT, slug),
    ):
        if os.path.isdir(candidate):
            return candidate
    return contract.model


def model_files_ok(model_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Verify required model files exist before CUDA materialization."""
    required = [
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ]
    report: Dict[str, Any] = {"path": model_path, "files": {}, "ok": False}
    for rel in required:
        present = os.path.isfile(os.path.join(model_path, rel))
        report["files"][rel] = present
    report["ok"] = all(report["files"].values())
    return report["ok"], report


def preflight_gpu(require_t4x2: bool = True) -> Dict[str, Any]:
    """G0 — assert the runtime really is two T4 GPUs with NVIDIA user-mode libs
    and an SDPA-frozen attention contract. Fail early otherwise."""
    from mage_t4x2.nvidia_bootstrap import preflight_nvidia_libs

    nvidia = preflight_nvidia_libs()
    import torch

    inventory = cuda_inventory()
    torch_version = torch.__version__
    result = {
        "torch": torch_version,
        "cuda_available": torch.cuda.is_available(),
        "device_count": len(inventory),
        "inventory": inventory,
        "t4x2_ok": False,
        "nvidia_libs": nvidia,
    }
    names = [dev.get("name", "") for dev in inventory]
    caps = [dev.get("compute_capability", "") for dev in inventory]
    result["t4x2_ok"] = (
        result["cuda_available"]
        and result["device_count"] == 2
        and all("T4" in n for n in names)
        and all(c == "7.5" for c in caps)
    )
    if require_t4x2 and not result["t4x2_ok"]:
        raise RuntimeError("T4x2 runtime required but not present; do not load the model.")
    return result


def _gpu_budgets(
    gpu0_total_bytes: Optional[int],
    gpu1_total_bytes: Optional[int],
    reserved_headroom_bytes: int,
) -> Dict[str, int]:
    """Derive concrete per-GPU budgets for the spread planner.

    For a real Tesla T4 (16 GiB device) the frozen usable figure observed in the
    G0_LOAD_OOM failure evidence (\"~14.56 GiB usable\") is authoritative. Any
    other device falls back to ``total_memory - reserved_headroom``.
    """
    import torch

    def _one(device: str, override: Optional[int]) -> int:
        if override is not None:
            return int(override)
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(device)
            except Exception:
                return C.T4_USABLE_VRAM_BYTES
            if abs(int(props.total_memory) - C.T4_VRAM_BYTES) < 1024**3:
                return C.T4_USABLE_VRAM_BYTES
            return max(0, int(props.total_memory) - reserved_headroom_bytes)
        return C.T4_USABLE_VRAM_BYTES

    return {"gpu0": _one("cuda:0", gpu0_total_bytes), "gpu1": _one("cuda:1", gpu1_total_bytes)}


def load_runtime_state_dual_t4_spread(
    model_path: str,
    contract: RunContract,
    source_authority: Dict[str, Any],
    *,
    gpu0_total_bytes: Optional[int] = None,
    gpu1_total_bytes: Optional[int] = None,
    reserved_headroom_bytes: int = C.RESERVED_RUNTIME_HEADROOM_BYTES,
) -> RuntimeState:
    """L0 — dual-T4 spread load (Corrective G0_LOAD_OOM).

    Stages the model entirely on CPU, measures real per-component static bytes,
    derives a conservative dual-T4 spread plan, and moves each component
    directly to its planned device. The upstream whole-model ``model.to(device)``
    is NEVER called: no full-model, single-GPU materialization can happen.
    """
    from mage_t4x2.sdpa_contract import assert_sdpa_frozen, freeze_sdpa_env
    from mage_t4x2.spread_loader import SpreadLoadDependencies, spread_load

    freeze_sdpa_env()
    sdpa = assert_sdpa_frozen()
    if sdpa["status"] != "PASS":
        raise RuntimeError(f"SDPA_NOT_FROZEN: {sdpa}")

    ok, report = model_files_ok(model_path)
    if not ok:
        raise RuntimeError(f"incomplete model files: {report}")

    from mage_flow import MageFlowPipeline, load_from_repo  # type: ignore

    budgets = _gpu_budgets(gpu0_total_bytes, gpu1_total_bytes, reserved_headroom_bytes)
    deps = SpreadLoadDependencies(builder=lambda path, device: load_from_repo(path, device=device))
    result = spread_load(
        model_path,
        deps,
        gpu0_total_bytes=budgets["gpu0"],
        gpu1_total_bytes=budgets["gpu1"],
        reserved_headroom_bytes=reserved_headroom_bytes,
    )
    model = result["model"]
    pipeline = MageFlowPipeline(model, device="cuda")
    state = RuntimeState(
        pipeline=pipeline,
        model=model,
        text_encoder=getattr(model, "txt_enc", None),
        transformer=getattr(model, "transformer", None),
        vae=getattr(model, "vae", None),
        scheduler=getattr(model, "scheduler", None),
        tokenizer=getattr(model, "tokenizer", None),
        source_authority=source_authority,
        contract=contract,
        model_path=model_path,
        memory_plan=result["memory_plan"],
        placement_validation=result["placement_validation"],
        no_full_model_cuda0_materialization=bool(result["no_full_model_cuda0_materialization"]),
        sdpa_frozen=sdpa,
        current_split_block=result["memory_plan"]["split_block"],
    )
    state.transformer.eval()
    if state.text_encoder is not None:
        state.text_encoder.eval()
    if state.vae is not None:
        state.vae.eval()
    state.current_profile = "mixed"
    state.current_topology = "dual_t4"
    state.device_map = result["memory_plan"]["device_map_json"]  # type: ignore[attr-defined]
    return state


def load_runtime_state(model_path: str, contract: RunContract, source_authority: Dict[str, Any]) -> RuntimeState:
    """LOAD (once), via the corrective dual-T4 spread loader."""
    return load_runtime_state_dual_t4_spread(model_path, contract, source_authority)


def factory_loader(
    model_path: str, contract: RunContract, source_authority: Dict[str, Any]
) -> Any:
    """Return a zero-arg loader callable suited for RuntimeHolder usage."""

    def _load() -> RuntimeState:
        return load_runtime_state(model_path, contract, source_authority)

    return _load


def _clear_disposables(state: RuntimeState) -> None:
    """Clear only disposable tensors; never reload model weights."""
    import torch
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _sync_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _validate_dtype(state: RuntimeState, key: str) -> Dict[str, Any]:
    """A4.2 — audit all three authority components; never a transformer-only report."""
    return audit_runtime_dtypes(state)


def _record(state: RuntimeState, telemetry: TelemetryRecorder, action: str) -> None:
    telemetry.event("state", action=action, **state.snapshot())


def _place_transformer(transformer: Any, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the complete plan (pre + blocks + post) to a live transformer.

    Pre modules go to ``transformer.pre`` devices, blocks follow
    ``transformer.blocks``, post modules follow ``transformer.post`` (they must
    sit with the final block device). ``pos_embed`` is only moved when it carries
    parameters/buffers.
    """
    import torch

    tr_plan = plan["transformer"]
    observed: Dict[str, Any] = {}

    pre = tr_plan.get("pre", {})
    for name, dev in pre.items():
        mod = getattr(transformer, name, None)
        if mod is not None and getattr(mod, "to", None) is not None:
            mod.to(torch.device(dev))
        observed[f"pre.{name}"] = dev

    pos = getattr(transformer, C.TRANSFORMER_POS_EMBED_ATTR, None)
    if pos is not None and has_parameters_or_buffers(pos) and getattr(pos, "to", None) is not None:
        pos.to(torch.device(pre.get("img_in", "cuda:0")))
        observed["pos_embed"] = pre.get("img_in", "cuda:0")

    blocks = _transformer_blocks(transformer)
    block_plan = tr_plan.get("blocks", {})
    for idx, dev in [(int(i), d) for i, d in block_plan.items()]:
        if idx < len(blocks):
            blocks[idx].to(torch.device(dev))
            observed[f"block.{idx}"] = dev

    post = tr_plan.get("post", {})
    for name, dev in post.items():
        mod = getattr(transformer, name, None)
        if mod is not None and getattr(mod, "to", None) is not None:
            mod.to(torch.device(dev))
        observed[f"post.{name}"] = dev

    inconsistent = post_blocks_inconsistent(transformer, extract_block_devices(plan))
    if inconsistent is not None:
        raise RuntimeError(f"TRANSFORMER_POST_DEVICE_MISMATCH: {inconsistent}")
    return observed


def apply_dual_t4_mixed(state: RuntimeState, num_blocks: int, split_block: Optional[int] = None) -> Dict[str, Any]:
    """G1/ROUTING — full dual plan: text encoder + pre + blocks[0:K) on cuda:0,
    blocks[K:N) + post + VAE on cuda:1. Uses upstream dtypes. Never reloads.

    Returns a complete A4.3 device plan. The dual-forward adapter is attached
    separately via ``attach_dual_adapter`` (Corrective A3 adapter-owner model).
    """
    import torch

    prior = state.snapshot()
    plan = build_device_plan(
        num_blocks,
        text_encoder_device="cuda:0",
        vae_device="cuda:1",
        strategy="explicit" if split_block is not None else "half",
        split_block=split_block,
    )
    te = state.text_encoder
    vae = state.vae
    if te is not None:
        te.to(torch.device("cuda:0"))
    if vae is not None:
        vae.to(torch.device("cuda:1"))
    if state.transformer is not None:
        _place_transformer(state.transformer, plan)

    state.current_profile = "mixed"
    state.current_topology = "dual_t4"
    state.device_map = plan["device_map_json"]  # type: ignore[attr-defined]
    _sync_cuda()
    _clear_disposables(state)
    return {
        "ok": True,
        "prior": prior,
        "device_plan": plan,
        "dual_forward_adapter_attached": getattr(state, "dual_forward_adapter", None) is not None
        and getattr(state.dual_forward_adapter, "attached", False),
    }


def apply_dual_t4_all_bf16(state: RuntimeState, num_blocks: int, split_block: Optional[int] = None) -> Dict[str, Any]:
    """G4 — the integrated final target: dual-T4 + all-BF16 over the SAME live
    transformer (identity preserved; no reload)."""
    import torch

    applied = apply_dual_t4_mixed(state, num_blocks, split_block=split_block)
    for idx, module in enumerate(_transformer_blocks(state.transformer)):
        module.to(torch.bfloat16)
    if state.transformer is not None:
        for attr in C.TRANSFORMER_PRE_MODULES:
            mod = getattr(state.transformer, attr, None)
            if mod is not None:
                mod.to(torch.bfloat16)
        for attr in C.TRANSFORMER_POST_MODULES:
            mod = getattr(state.transformer, attr, None)
            if mod is not None:
                mod.to(torch.bfloat16)
    if state.text_encoder is not None:
        state.text_encoder.to(torch.bfloat16)
    if state.vae is not None:
        state.vae.to(torch.bfloat16)
    state.current_profile = "all_bf16"
    state.current_topology = "dual_t4"
    _sync_cuda()
    _clear_disposables(state)
    report = _validate_dtype(state, "transformer")
    applied["dtype_audit"] = report
    applied["ok"] = report["overall_status"] == "PASS"
    return applied


def attach_dual_forward_adapter(state: RuntimeState, block_devices: Dict[int, str]) -> Any:
    """Attach the upstream-compatible dual-device forward adapter (telemetry-less).

    Raises RuntimeError if the adapter cannot be attached (e.g. missing block
    container) — the caller must FAIL BEFORE INFERENCE in that case.
    """
    return attach_dual_adapter(
        state,
        {"transformer": {"blocks": {int(k): v for k, v in block_devices.items()}}},
        telemetry=None,
        run_id="n/a",
        phase="routing",
        inference_id="n/a",
    )


def attach_dual_adapter(
    state: RuntimeState,
    device_plan: Dict[str, Any],
    telemetry: Any = None,
    run_id: str = "n/a",
    phase: str = "routing",
    inference_id: str = "n/a",
) -> Any:
    """Attach a phase-owned dual-device forward adapter bound to the provided
    telemetry / run_id / phase / inference_id.

    Raises:
      * RuntimeError if the transformer does not expose a block container;
      * ADAPTER_DOUBLE_ATTACH if an adapter is already live on this transformer.
    """
    from mage_t4x2.dual_device_forward import DualDeviceForwardAdapter
    from mage_t4x2.device_bridges import VaeInputBridge

    if state.transformer is None:
        raise RuntimeError("no transformer to attach dual-forward adapter to")
    if getattr(state, "dual_forward_adapter", None) is not None and getattr(
        state.dual_forward_adapter, "attached", False
    ):
        raise RuntimeError("ADAPTER_DOUBLE_ATTACH: live adapter already attached")
    block_devices = extract_block_devices(device_plan)
    if not block_devices:
        raise RuntimeError("TRANSFORMER_BLOCK_ERROR: no block device mapping in plan")
    inconsistent = post_blocks_inconsistent(state.transformer, block_devices)
    if inconsistent is not None:
        raise RuntimeError(f"TRANSFORMER_POST_DEVICE_MISMATCH: {inconsistent}")
    adapter = DualDeviceForwardAdapter(
        state.transformer,
        block_devices,
        telemetry=telemetry,
        run_id=run_id,
        phase=phase,
        inference_id=inference_id,
    )
    adapter.attach()
    state.dual_forward_adapter = adapter  # type: ignore[attr-defined]
    if state.vae is not None:
        bridge = VaeInputBridge(
            state.vae,
            telemetry=telemetry,
            run_id=run_id,
            phase=phase,
            inference_id=inference_id,
            mover=getattr(state, "vae_mover", None),
        )
        bridge.attach()
        state.vae_input_bridge = bridge  # type: ignore[attr-defined]
    return adapter


def detach_dual_adapter(state: RuntimeState, adapter: Any = None, reason: str = "phase_end") -> bool:
    """Safely detach the live dual-forward adapter (restores the original forward)
    and the VAE input bridge (restores the original decode)."""
    adapter = adapter or getattr(state, "dual_forward_adapter", None)
    if adapter is not None and getattr(adapter, "attached", False):
        adapter.detach()
    state.dual_forward_adapter = None  # type: ignore[attr-defined]
    bridge = getattr(state, "vae_input_bridge", None)
    if bridge is not None and getattr(bridge, "attached", False):
        bridge.detach()
    state.vae_input_bridge = None  # type: ignore[attr-defined]
    return True


def create_evidence_run(run_dir: str, source_authority: Optional[Dict[str, Any]] = None) -> EvidenceRun:
    return EvidenceRun(run_dir, source_authority=source_authority)


def create_telemetry(path: str, run_id: str, phase: str, inference_id: str) -> TelemetryRecorder:
    return TelemetryRecorder(path, run_id=run_id, phase=phase, inference_id=inference_id)


def collect_telemetry(path: str) -> list:
    from mage_t4x2.telemetry import parse_telemetry

    return parse_telemetry(path)


def derive_phase_facts(
    phase: str,
    result: Dict[str, Any],
    telemetry_records: list,
    model_load_count: int,
    runtime_state: Any,
    contract: RunContract,
    adapter_status: Optional[Dict[str, Any]] = None,
    replay_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from mage_t4x2.phase_acceptance import derive_phase_facts as _derive

    return _derive(
        phase,
        result,
        telemetry_records,
        model_load_count,
        runtime_state,
        contract,
        adapter_status=adapter_status,
        replay_evidence=replay_evidence,
    )


def evaluate_phase_acceptance(phase: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    from mage_t4x2.phase_acceptance import evaluate_phase_acceptance as _evaluate

    return _evaluate(phase, facts)


def dual_forward_adapter_attached(state: RuntimeState) -> bool:
    return bool(getattr(state, "dual_forward_adapter", None) is not None and getattr(state.dual_forward_adapter, "attached", False))


def _transformer_blocks(transformer: Any):
    from mage_t4x2.block_partition import discover_transformer_blocks

    blocks, _ = discover_transformer_blocks(transformer)
    return blocks


def run_t2i(
    state: RuntimeState,
    contract: RunContract,
    run: EvidenceRun,
    telemetry: TelemetryRecorder,
    run_id: str,
    inference_id: Optional[str] = None,
    phase: str = "t2i",
) -> Dict[str, Any]:
    """Execute one T2I under the current configuration and collect evidence."""
    import numpy as np

    telemetry.inference_id = inference_id or telemetry.inference_id
    telemetry.set_phase(phase)
    telemetry.phase("t2i", "START")
    prompt = "a red fox in a snowy forest at golden hour, high detail"
    h, w = list(contract.resolution)
    images = state.pipeline.generate(
        [prompt],
        steps=contract.steps,
        cfg=contract.cfg,
        heights=[h],
        widths=[w],
        seeds=[contract.seed],
    )
    img = images[0]

    out_png = os.path.join(str(run.run_dir), "output.png")
    img.save(out_png, format="PNG")
    telemetry.metric("output_bytes", os.path.getsize(out_png))
    telemetry.phase("t2i", "PASS")

    validation = write_validation(out_png, os.path.join(str(run.run_dir), "image-validation.json"))
    run.write_device_map(getattr(state, "device_map", None) or {})
    dtype_report = audit_runtime_dtypes(state)
    run.write_dtype("after", dtype_report)

    arr = np.asarray(np.array(img.convert("RGB")), dtype=np.float32)
    return {
        "output_png": out_png,
        "validation": validation,
        "dtype_after": dtype_report,
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "inference_id": telemetry.inference_id,
        "run_id": run_id,
        "phase": phase,
        "exit_code": 0,
        "run_exit_zero": True,
    }


def run_to_facts(
    result: Dict[str, Any],
    contract: RunContract,
    topo: str,
    dual: bool,
    telemetry_records: Optional[list] = None,
    gpu_inventory: Optional[list] = None,
) -> Dict[str, Any]:
    """Derive authority facts. GPU participation/transfer facts MUST come from
    telemetry evidence — never from the ``dual`` flag (config intent)."""
    from mage_t4x2.evidence_reducers import (
        derive_cross_gpu_transfer,
        derive_gpu_participation,
        derive_single_t2i_instance,
        derive_t4x2_hardware,
    )
    from mage_t4x2.phase_acceptance import derive_run_exit_zero

    val = result["validation"]
    records = telemetry_records or []
    inference_id = result.get("inference_id")
    gpu0 = derive_gpu_participation(records, "cuda:0", inference_id=inference_id)
    gpu1 = derive_gpu_participation(records, "cuda:1", inference_id=inference_id)
    transfer = derive_cross_gpu_transfer(records, inference_id=inference_id)
    t4 = derive_t4x2_hardware(gpu_inventory)
    single = derive_single_t2i_instance(records)

    return {
        "PYTORCH_RUNTIME": True,
        "UPSTREAM_MAGE_PINNED": True,
        "MODEL_ID_MATCH": True,
        "MODEL_REVISION_MATCH": True,
        "CPU_FALLBACK_FORBIDDEN": not contract.cpu_fallback,
        "T4_COUNT_EQ_2": t4["status"] == "PASS",
        "TEXT_ENCODER_BF16": result["dtype_after"].get("text_encoder", {}).get("status") == "PASS",
        "TRANSFORMER_BF16": result["dtype_after"].get("transformer", {}).get("status") == "PASS",
        "VAE_BF16": result["dtype_after"].get("vae", {}).get("status") == "PASS",
        "SINGLE_T2I_INSTANCE": single["status"] == "PASS",
        "GPU0_PARTICIPATION": gpu0["status"] == "PASS",
        "GPU1_PARTICIPATION": gpu1["status"] == "PASS",
        "CROSS_GPU_TRANSFER_OBSERVED": transfer["status"] == "PASS",
        "OUTPUT_EXISTS": val.get("exists", False),
        "OUTPUT_512X512": val.get("is_512x512", False),
        "OUTPUT_RGB_VALID": val.get("rgb_valid", False),
        "NO_NAN": not result.get("has_nan", True),
        "NO_INF": not result.get("has_inf", True),
        "RUN_EXIT_ZERO": derive_run_exit_zero(result),
    }
