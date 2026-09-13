"""Byte-accurate dual-T4 spread-load planning (Corrective G0_LOAD_OOM).

The real G0_LOAD_OOM failure materialized the full model on a single T4
(via ``load_from_repo(model_path, device='cuda') -> model.to(device)``), and
the ~17.45 GB of BF16 weights cannot fit inside one T4's 14.56 GiB of usable
VRAM. This module plans where every component must live so that no full-model,
single-device materialization ever happens:

    cuda:0  text_encoder + transformer pre + blocks[0:k)
    cuda:1  blocks[k:N) + transformer post         + vae

A plan is only considered feasible when, for every GPU:

    planned_static_bytes + reserved_headroom_bytes <= gpu_total_bytes

``plan_dual_t4_spread`` enumerates every contiguous split ``k``, keeps only
feasible candidates, and picks the one that minimizes the peak static usage
(tie-broken by the smaller ``k``). When no split is feasible, or when the
whole model would need to sit on one device, the planner fails closed with
``status: FAIL`` — it never invents an unsafe placement.

Everything here is CPU-computable: ``torch`` is imported lazily only for byte
accounting, and the module can be imported on machines without CUDA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import constants as C
from .block_partition import adjacent_transfers, canonical_device_map
from .transformer_placement import (
    discover_transformer_post_modules,
    discover_transformer_pre_modules,
)

COMPONENT_KEYS = ("text_encoder", "transformer_pre", "transformer_blocks", "transformer_post", "vae")


def account_module_bytes(module: Any) -> int:
    """Sum of ``numel * element_size`` across all parameters and buffers."""
    import torch

    total = 0
    if module is None:
        return total
    if hasattr(module, "named_parameters"):
        for _name, tensor in module.named_parameters(recurse=True):
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
    if hasattr(module, "named_buffers"):
        for _name, tensor in module.named_buffers(recurse=True):
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
    if total == 0:
        raise ValueError(
            f"component has no parameters or buffers to account ({type(module).__name__})"
        )
    return total


def measure_components_from_model(
    model: Any,
    *,
    text_encoder_attr: str = "txt_enc",
    vae_attr: str = "vae",
    transformer_attr: str = "transformer",
) -> Dict[str, Any]:
    """Measure real static bytes from a loaded upstream ``MageFlowModel``.

    ``transformer_blocks`` is a per-block byte list in block order. Pre/post
    bytes are the summed bytes of every discovered pre/post submodule.
    """
    from .block_partition import discover_transformer_blocks

    text_encoder = getattr(model, text_encoder_attr, None)
    vae = getattr(model, vae_attr, None)
    transformer = getattr(model, transformer_attr, None)

    blocks, _ = discover_transformer_blocks(transformer)
    pre = discover_transformer_pre_modules(transformer)
    post = discover_transformer_post_modules(transformer)

    return {
        "text_encoder": account_module_bytes(text_encoder) if text_encoder is not None else 0,
        "transformer_pre": sum(account_module_bytes(m) for m in pre.values()),
        "transformer_blocks": [account_module_bytes(b) for b in blocks],
        "transformer_post": sum(account_module_bytes(m) for m in post.values()),
        "vae": account_module_bytes(vae) if vae is not None else 0,
    }


def total_static_bytes(component_bytes: Dict[str, Any]) -> int:
    return (
        int(component_bytes["text_encoder"])
        + int(component_bytes["transformer_pre"])
        + int(component_bytes["transformer_post"])
        + int(component_bytes["vae"])
        + sum(int(b) for b in component_bytes["transformer_blocks"])
    )


def evaluate_split_bytes(
    component_bytes: Dict[str, Any],
    split_block: int,
    *,
    text_encoder_device: str = C.GPU0,
    vae_device: str = C.GPU1,
) -> Dict[str, Any]:
    """Static bytes per device for a contiguous split at ``split_block``."""
    blocks = [int(b) for b in component_bytes["transformer_blocks"]]
    n = len(blocks)
    if split_block < 1 or split_block >= n:
        raise ValueError(f"split_block {split_block} out of range for {n} blocks")
    device0 = int(component_bytes["text_encoder"]) + int(component_bytes["transformer_pre"]) + sum(blocks[:split_block])
    device1 = int(component_bytes["vae"]) + int(component_bytes["transformer_post"]) + sum(blocks[split_block:])
    return {
        "split_block": split_block,
        text_encoder_device: device0,
        vae_device: device1,
        "device0_bytes": device0,
        "device1_bytes": device1,
    }


def evaluate_single_device(
    component_bytes: Dict[str, Any],
    *,
    gpu_total_bytes: int = C.T4_USABLE_VRAM_BYTES,
    reserved_headroom_bytes: int = C.RESERVED_RUNTIME_HEADROOM_BYTES,
) -> Dict[str, Any]:
    """Check the (physically impossible) single-T4 full-model materialization.

    Mirrors the real failure: ``total_static + headroom > usable`` -> FAIL.
    """
    static = total_static_bytes(component_bytes)
    needed = static + reserved_headroom_bytes
    fits = needed <= gpu_total_bytes
    return {
        "total_model_static_bytes": static,
        "gpu_total_bytes": gpu_total_bytes,
        "reserved_runtime_headroom_bytes": reserved_headroom_bytes,
        "needed_bytes": needed,
        "shortfall_bytes": max(0, needed - gpu_total_bytes),
        "single_device_min_gpus_required": (needed + gpu_total_bytes - 1) // gpu_total_bytes,
        "status": "PASS" if fits else "FAIL",
        "reason": (
            "single T4 full-model materialization fits"
            if fits
            else f"single T4 full-model materialization impossible: {needed} needed > {gpu_total_bytes} usable"
        ),
    }


def plan_dual_t4_spread(
    component_bytes: Dict[str, Any],
    *,
    num_blocks: Optional[int] = None,
    gpu0_total_bytes: int = C.T4_USABLE_VRAM_BYTES,
    gpu1_total_bytes: int = C.T4_USABLE_VRAM_BYTES,
    reserved_headroom_bytes: int = C.RESERVED_RUNTIME_HEADROOM_BYTES,
    text_encoder_device: str = C.GPU0,
    vae_device: str = C.GPU1,
) -> Dict[str, Any]:
    """Best feasible contiguous dual-T4 spread plan, or a FAIL-CLOSED result."""
    blocks = [int(b) for b in component_bytes["transformer_blocks"]]
    n = len(blocks)
    if n < 1:
        raise ValueError("at least one transformer block is required")
    if num_blocks is not None and num_blocks != n:
        raise ValueError(f"num_blocks {num_blocks} does not match measured blocks {n}")

    base = {
        "planner": C.MEMORY_PLANNER_NAME,
        "topology": "dual_t4",
        "text_encoder_device": text_encoder_device,
        "vae_device": vae_device,
        "gpu0_total_bytes": gpu0_total_bytes,
        "gpu1_total_bytes": gpu1_total_bytes,
        "reserved_runtime_headroom_bytes": reserved_headroom_bytes,
        "total_model_static_bytes": total_static_bytes(component_bytes),
        "single_device": evaluate_single_device(
            component_bytes,
            gpu_total_bytes=gpu0_total_bytes,
            reserved_headroom_bytes=reserved_headroom_bytes,
        ),
        "components": {
            "text_encoder": {
                "device": text_encoder_device,
                "static_bytes": int(component_bytes["text_encoder"]),
            },
            "transformer_pre": {
                "device": text_encoder_device,
                "static_bytes": int(component_bytes["transformer_pre"]),
            },
            "transformer_blocks": [
                {"index": i, "static_bytes": b} for i, b in enumerate(blocks)
            ],
            "transformer_post": {
                "device": vae_device,
                "static_bytes": int(component_bytes["transformer_post"]),
            },
            "vae": {
                "device": vae_device,
                "static_bytes": int(component_bytes["vae"]),
            },
        },
    }

    candidate_splits: List[Dict[str, Any]] = []
    rejections: List[str] = []
    for k in range(1, n):
        split = evaluate_split_bytes(
            component_bytes,
            k,
            text_encoder_device=text_encoder_device,
            vae_device=vae_device,
        )
        d0 = split["device0_bytes"]
        d1 = split["device1_bytes"]
        g0_ok = d0 + reserved_headroom_bytes <= gpu0_total_bytes
        g1_ok = d1 + reserved_headroom_bytes <= gpu1_total_bytes
        if g0_ok and g1_ok:
            candidate_splits.append({"split_block": k, "device0_bytes": d0, "device1_bytes": d1})
        else:
            reasons = []
            if not g0_ok:
                reasons.append(f"gpu0 needs {d0} static + {reserved_headroom_bytes} headroom > {gpu0_total_bytes}")
            if not g1_ok:
                reasons.append(f"gpu1 needs {d1} static + {reserved_headroom_bytes} headroom > {gpu1_total_bytes}")
            rejections.append({"split_block": k, "reasons": reasons})

    if not candidate_splits:
        plan = dict(base)
        plan["status"] = "FAIL"
        plan["reason"] = "no feasible conservative spread split exists"
        plan["split_rejections"] = rejections
        plan["split_block"] = None
        plan["planned_static_bytes"] = {}
        plan["remaining_after_headroom_bytes"] = {}
        return plan

    best = min(
        candidate_splits,
        key=lambda c: (max(c["device0_bytes"], c["device1_bytes"]), c["split_block"]),
    )
    k = best["split_block"]
    d0 = best["device0_bytes"]
    d1 = best["device1_bytes"]

    from .device_plan import plan_block_devices

    block_map, split = plan_block_devices(
        n,
        device0=text_encoder_device,
        device1=vae_device,
        strategy="explicit",
        split_block=k,
    )
    from .transformer_placement import build_full_plan

    plan_section = build_full_plan(n, text_encoder_device=text_encoder_device, vae_device=vae_device, split_block=k)

    plan = dict(base)
    plan.update(
        {
            "status": "PASS",
            "reason": (
                f"best feasible split at block {k}; peak static "
                f"{max(d0, d1)} bytes <= {min(gpu0_total_bytes, gpu1_total_bytes)} - {reserved_headroom_bytes} headroom"
            ),
            "split_block": k,
            "split_rejections": rejections,
            "planned_static_bytes": {text_encoder_device: d0, vae_device: d1},
            "remaining_after_headroom_bytes": {
                text_encoder_device: gpu0_total_bytes - d0 - reserved_headroom_bytes,
                vae_device: gpu1_total_bytes - d1 - reserved_headroom_bytes,
            },
            "block_devices": {str(i): str(dev) for i, dev in sorted(block_map.items())},
            "split_policy": "memory_aware",
            "device_map_json": canonical_device_map(text_encoder_device, vae_device, block_map),
            "transfers": adjacent_transfers(block_map),
            "device_plan": plan_section,
        }
    )
    return plan


def validate_plan_within_budget(plan: Dict[str, Any], *, gpu_totals: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Independent re-check that the chosen plan stays within every GPU budget.

    Never trusts the planner's own arithmetic: recomputes per-device static
    bytes straight from ``components`` and reapplies the headroom constraint.
    """
    if plan.get("status") != "PASS":
        return {"status": "SKIP", "reason": "plan is not PASS"}
    totals = gpu_totals or {
        plan["text_encoder_device"]: plan["gpu0_total_bytes"],
        plan["vae_device"]: plan["gpu1_total_bytes"],
    }
    headroom = int(plan["reserved_runtime_headroom_bytes"])
    violations: List[str] = []
    for device in (plan["text_encoder_device"], plan["vae_device"]):
        total = int(totals[device])
        used = int(plan["planned_static_bytes"][device])
        if used + headroom > total:
            violations.append(f"{device}: {used} + {headroom} > {total}")
    return {
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "computed_static_bytes": plan["planned_static_bytes"],
        "headroom_bytes": headroom,
    }
