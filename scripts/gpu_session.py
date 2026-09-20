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
