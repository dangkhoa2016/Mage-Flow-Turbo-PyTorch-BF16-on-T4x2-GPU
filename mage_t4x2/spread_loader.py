"""Repository-owned dual-T4 spread loader (Corrective G0_LOAD_OOM).

The real G0_LOAD_OOM happened inside the upstream loader's

    model.to(device)            # full model -> ONE device  (the OOM)

The authority therefore never calls that call. Instead, the model is staged
entirely on CPU (upstream stages the transformer on CPU already; ``device`` is
only consumed by the final ``model.to(device)``), static bytes are measured per
component, a conservative dual-T4 spread plan is computed, and each component is
moved directly to its planned device. No full-model materialization ever
happens.

All the interesting logic is injectable so the authoritative spread behavior can
be exercised on CPU: ``builder`` stages the model, ``measure`` produces the byte
accounting, ``plan`` derives the conservative split, ``move`` applies placements
(the real one moves tensors; tests use a recorder), and ``validate`` inspects
observed placement.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

from . import constants as C
from .device_plan import iter_device_plan_leaves
from .memory_plan import (
    measure_components_from_model,
    plan_dual_t4_spread,
    validate_plan_within_budget,
)
from .transformer_placement import module_devices, validate_runtime_device_plan

MODEL_CONTAINER_ATTRS = ("blocks", "double_blocks", "single_blocks", "layers", "transformer_blocks", "dit_blocks")


def _default_measure(model: Any) -> Dict[str, Any]:
    return measure_components_from_model(model)


def _default_plan(
    component_bytes: Dict[str, Any],
    *,
    gpu0_total_bytes: int,
    gpu1_total_bytes: int,
    reserved_headroom_bytes: int,
) -> Dict[str, Any]:
    return plan_dual_t4_spread(
        component_bytes,
        gpu0_total_bytes=gpu0_total_bytes,
        gpu1_total_bytes=gpu1_total_bytes,
        reserved_headroom_bytes=reserved_headroom_bytes,
    )


def move_transformer_by_plan_items(transformer: Any, plan: Dict[str, Any], observed: Dict[str, Any]) -> None:
    """Place pre / blocks / post of ``transformer`` per the plan's nested map."""
    import torch

    tr_plan = plan["transformer"]
    pre = tr_plan.get("pre", {})
    for name, dev in pre.items():
        mod = getattr(transformer, name, None)
        if mod is not None and getattr(mod, "to", None) is not None:
            mod.to(torch.device(dev))
        observed[f"pre.{name}"] = dev

    blocks, _ = _discover_blocks(transformer)
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


def _discover_blocks(transformer: Any):
    for attr in MODEL_CONTAINER_ATTRS:
        if hasattr(transformer, attr):
            container = getattr(transformer, attr)
            items = list(container) if not isinstance(container, dict) else list(container.values())
            if items:
                return items, attr
    raise ValueError("no transformer block container found")


def default_move_by_plan(
    model: Any,
    plan: Dict[str, Any],
    *,
    text_encoder_attr: str = "txt_enc",
    transformer_attr: str = "transformer",
    vae_attr: str = "vae",
) -> Dict[str, Any]:
    """Move the live model components directly to their planned devices.

    This is the ONLY placement primitive used during spread load: it never
    calls ``model.to(device)``. Every stateful component keeps using its own
    planned device for the rest of the authority session.
    """
    import torch

    dp = plan["device_plan"]
    te = getattr(model, text_encoder_attr, None)
    tr = getattr(model, transformer_attr, None)
    vae = getattr(model, vae_attr, None)

    observed: Dict[str, Any] = {
        "text_encoder": dp["text_encoder"],
        "vae": dp["vae"],
    }
    if te is not None and getattr(te, "to", None) is not None:
        te.to(torch.device(dp["text_encoder"]))
    if vae is not None and getattr(vae, "to", None) is not None:
        vae.to(torch.device(dp["vae"]))
    if tr is not None:
        move_transformer_by_plan_items(tr, dp, observed)
    return observed


def devices_in_use(model: Any) -> set:
    """Distinct devices across all parameters and buffers (CPU-safe)."""
    import torch

    devices: set = set()
    if model is None:
        return devices
    for _n, p in model.named_parameters(recurse=True):
        devices.add(str(getattr(p, "device", "?")))
    for _n, b in model.named_buffers(recurse=True):
        devices.add(str(getattr(b, "device", "?")))
    return devices


def cuda_devices_in_use(model: Any) -> set:
    return {d for d in devices_in_use(model) if d.startswith("cuda")}


def no_full_model_single_cuda(model: Any, *, expected_devices: set) -> bool:
    """True when parameters/buffers span multiple CUDA devices (never a single
    full-model materialization) and both expected devices are in use."""
    cuda = cuda_devices_in_use(model)
    if len(cuda) < 2:
        return False
    return expected_devices.issubset(cuda)


def _resolve_leaf_module_for_validation(
    model: Any,
    path: str,
    *,
    text_encoder_attr: str,
    transformer_attr: str,
    vae_attr: str,
) -> Any:
    """Resolve a flattened ``(path, device)`` leaf to the live model module."""
    if path == "text_encoder":
        return getattr(model, text_encoder_attr, None)
    if path == "vae":
        return getattr(model, vae_attr, None)
    if path.startswith("transformer."):
        tr = getattr(model, transformer_attr, None)
        if tr is None:
            return None
        parts = path[len("transformer.") :].split(".")
        if parts[0] in ("pre", "post") and len(parts) == 2:
            return getattr(tr, parts[1], None)
        if parts[0] == "blocks" and len(parts) == 2:
            try:
                idx = int(parts[1])
            except ValueError:
                return None
            try:
                block_list, _ = _discover_blocks(tr)
            except ValueError:
                return None
            return block_list[idx] if 0 <= idx < len(block_list) else None
        try:
            idx = int(parts[0])
        except ValueError:
            return None
        try:
            block_list, _ = _discover_blocks(tr)
        except ValueError:
            return None
        return block_list[idx] if 0 <= idx < len(block_list) else None
    return None


def _observed_leaf_devices(module: Any) -> list:
    """Observed device(s) for a module: every parameter/buffer, falling back to a
    plain ``.device`` attribute so lightweight CPU test fakes remain usable."""
    devs = set()
    if module is not None:
        for kind in ("parameters", "buffers"):
            for _name, dev in module_devices(module)[kind].items():
                devs.add(str(dev))
        if not devs:
            fallback = getattr(module, "device", None)
            if fallback is not None:
                devs.add(str(fallback))
    return sorted(devs)


def _resolve_group_leaf_modules(
    model: Any,
    path: str,
    *,
    transformer_attr: str,
) -> list:
    """Resolve the flat ``transformer.pre`` / ``transformer.post`` group leaf to
    the discovered pre/post modules of the transformer (empty when absent)."""
    if path not in ("transformer.pre", "transformer.post"):
        return []
    from .transformer_placement import (
        discover_transformer_post_modules,
        discover_transformer_pre_modules,
    )

    tr = getattr(model, transformer_attr, None)
    if tr is None:
        return []
    if path == "transformer.pre":
        return list(discover_transformer_pre_modules(tr).values())
    return list(discover_transformer_post_modules(tr).values())
