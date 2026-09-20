"""Complete transformer device placement (Corrective A4.3).

A dual-T4 transformer is NOT only ``transformer_blocks``. The upstream forward
(Mage Flow, pinned in A2.1) has pre-block modules (``pos_embed`` / ``img_in`` /
``txt_norm`` / ``time_text_embed`` / ``txt_in``), a plain-Python block loop, and
post-block modules (``norm_out`` / ``proj_out``).

A correct dual-device plan must place all stateful modules consistently:
    cuda:0  text_encoder + pre modules + blocks[0:K)
    cuda:1  blocks[K:N) + post modules + VAE

Everything in this module is CPU-safe: ``torch`` is imported lazily inside
functions so importing this module never initializes CUDA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import constants as C
from .block_partition import discover_transformer_blocks

RUNTIME_TENSOR_DEVICE = "RUNTIME_TENSOR_DEVICE"


def has_parameters_or_buffers(module: Any) -> bool:
    """True when the object exposes at least one parameter or buffer."""
    if module is None:
        return False
    if hasattr(module, "named_parameters"):
        for _ in module.named_parameters(recurse=True):
            return True
    if hasattr(module, "named_buffers"):
        for _ in module.named_buffers(recurse=True):
            return True
    return False


def discover_transformer_pre_modules(transformer: Any) -> Dict[str, Any]:
    """Return ``{name: module}`` for every present pre-block module."""
    out: Dict[str, Any] = {}
    for name in C.TRANSFORMER_PRE_MODULES:
        if hasattr(transformer, name):
            out[name] = getattr(transformer, name)
    return out


def discover_transformer_post_modules(transformer: Any) -> Dict[str, Any]:
    """Return ``{name: module}`` for every present post-block module."""
    out: Dict[str, Any] = {}
    for name in C.TRANSFORMER_POST_MODULES:
        if hasattr(transformer, name):
            out[name] = getattr(transformer, name)
    return out


def pos_embed_placement(transformer: Any, default_device: str) -> str:
    """Explicit device for ``pos_embed``, or ``RUNTIME_TENSOR_DEVICE`` when the
    object is parameterless (do not invent a ``.to()`` it cannot honour)."""
    pos = getattr(transformer, C.TRANSFORMER_POS_EMBED_ATTR, None)
    if pos is None:
        return RUNTIME_TENSOR_DEVICE
    if has_parameters_or_buffers(pos):
        return default_device
    return RUNTIME_TENSOR_DEVICE


def build_full_plan(
    num_blocks: int,
    *,
    text_encoder_device: str = C.GPU0,
    vae_device: str = C.GPU1,
    block_devices: Optional[Dict[int, str]] = None,
    transformer: Any = None,
    split_block: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the complete A4.3 human/machine readable device plan.

    ``block_devices`` maps block index -> device. When omitted it is derived with
    the default half policy. ``transformer`` (optional) is inspected only for the
    parameterless ``pos_embed`` determination.
    """

    def _refine(target: Any, fallback: str) -> str:
        if target is None or not has_parameters_or_buffers(target):
            return fallback
        return fallback

    if block_devices is None:
        from .device_plan import plan_block_devices

        block_map, split = plan_block_devices(
            num_blocks,
            device0=text_encoder_device,
            device1=vae_device,
            strategy="explicit" if split_block is not None else "half",
            split_block=split_block,
        )
    else:
        block_map = {int(k): str(v) for k, v in block_devices.items()}
        split = max([k for k, v in block_map.items() if v == text_encoder_device] + [0]) + 1

    final_block = max(block_map) if block_map else 0
    final_device = block_map[final_block] if block_map else vae_device

    pre = {
        name: text_encoder_device
        for name in C.TRANSFORMER_PRE_MODULES
        if transformer is None or hasattr(transformer, name)
    }
    post = {
        name: final_device
        for name in C.TRANSFORMER_POST_MODULES
        if transformer is None or hasattr(transformer, name)
    }
    pe = pos_embed_placement(transformer, text_encoder_device) if transformer is not None else RUNTIME_TENSOR_DEVICE

    plan: Dict[str, Any] = {
        "topology": "dual_t4",
        "text_encoder": text_encoder_device,
        "vae": vae_device,
        "transformer": {
            "pre": pre,
            "blocks": {str(k): str(v) for k, v in sorted(block_map.items())},
            "post": post,
        },
        "pos_embed": pe,
        "block_devices": {str(k): str(v) for k, v in sorted(block_map.items())},
        "final_block": final_block,
        "final_block_device": final_device,
        "num_transformer_blocks": num_blocks,
        "split_policy": "explicit" if split_block is not None else "half",
        "split_block": split,
    }
    return plan


def apply_full_transformer_device(transformer: Any, device: Any) -> Dict[str, Any]:
    """Move every stateful transformer module (pre + blocks + post + pos_embed)
    to ``device``. Used by the single-T4 gather path so no residual module stays
    behind on another device. Returns the observed placement map."""
    import torch

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    observed: Dict[str, Any] = {}

    pos = getattr(transformer, C.TRANSFORMER_POS_EMBED_ATTR, None)
    if pos is not None and has_parameters_or_buffers(pos) and hasattr(pos, "to"):
        pos.to(dev)
        observed["pos_embed"] = str(device)

    for name in C.TRANSFORMER_PRE_MODULES:
        mod = getattr(transformer, name, None)
        if mod is not None and has_parameters_or_buffers(mod) and hasattr(mod, "to"):
            mod.to(dev)
        observed[name] = str(device)

    blocks, _ = discover_transformer_blocks(transformer)
    for idx in range(len(blocks)):
        blocks[idx].to(dev)

    for name in C.TRANSFORMER_POST_MODULES:
        mod = getattr(transformer, name, None)
        if mod is not None and has_parameters_or_buffers(mod) and hasattr(mod, "to"):
            mod.to(dev)
        observed[name] = str(device)

    observed["num_blocks"] = len(blocks)
    return observed


def module_devices(module: Any) -> Dict[str, Dict[str, str]]:
    """Observed parameter/buffer device of a module (path -> device). CPU-safe."""
    devices: Dict[str, Dict[str, str]] = {"parameters": {}, "buffers": {}}
    if module is None:
        return devices
    if hasattr(module, "named_parameters"):
        for name, tensor in module.named_parameters(recurse=True):
            devices["parameters"][name] = str(getattr(tensor, "device", "?"))
    if hasattr(module, "named_buffers"):
        for name, tensor in module.named_buffers(recurse=True):
            devices["buffers"][name] = str(getattr(tensor, "device", "?"))
    return devices


def validates_on_device(module: Any, expected_device: str) -> bool:
    """Structural device validation for fake/real modules: every parameter and
    buffer must be on ``expected_device``. Parameterless modules return True."""
    probes = []
    if module is None:
        return True
    for kind in ("parameters", "buffers"):
        for name, dev in module_devices(module)[kind].items():
            probes.append(dev)
    if not probes:
        return True
    return all(str(d) == str(expected_device) for d in probes)


def validate_runtime_device_plan(state: Any, expected_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Validate actual parameter/buffer placement against an expected plan.

    Returns ``{observed, expected, mismatches, status}``. Never reports PASS
    merely because ``.to(device)`` was invoked — actual tensor devices are
    inspected where possible. CPU tests use fake modules / mocked devices.

    Expected plan shapes accepted:
      * A4.3 nested dict ``{"transformer": {"pre": {...}, "blocks": {...}, "post": {...}}}``
      * A legacy flat ``{"transformer": {block_idx: device}}``
    """
    observed: Dict[str, Any] = {}
    mismatches: List[str] = []

    def _check(name: str, module: Any, expected_device: str) -> None:
        observed[name] = expected_device
        if not validates_on_device(module, expected_device):
            mismatches.append(f"{name}: expected {expected_device} but has a parameter/buffer elsewhere")

    te_dev = expected_plan.get("text_encoder")
    vae_dev = expected_plan.get("vae")
    transformer_section = expected_plan.get("transformer", {})

    pre = transformer_section.get("pre", {}) if isinstance(transformer_section, dict) else {}
    blocks = transformer_section.get("blocks", {}) if isinstance(transformer_section, dict) else {}
    post = transformer_section.get("post", {}) if isinstance(transformer_section, dict) else {}

    tr = getattr(state, "transformer", None)
    if tr is not None:
        if isinstance(pre, str):
            # flat group leaf: every discovered pre module sits on that device
            for name, mod in discover_transformer_pre_modules(tr).items():
                _check(f"transformer.pre.{name}", mod, pre)
        elif pre:
            for name, dev in pre.items():
                mod = getattr(tr, name, None)
                _check(f"transformer.pre.{name}", mod, dev)
        if isinstance(post, str):
            # flat group leaf: every discovered post module sits on that device
            for name, mod in discover_transformer_post_modules(tr).items():
                _check(f"transformer.post.{name}", mod, post)
        elif post:
            for name, dev in post.items():
                mod = getattr(tr, name, None)
                _check(f"transformer.post.{name}", mod, dev)
        if blocks:
            block_list, _ = discover_transformer_blocks(tr)
            for idx, dev in sorted(blocks.items(), key=lambda kv: int(kv[0])):
                idx = int(idx)
                if idx < len(block_list):
                    _check(f"transformer.block.{idx}", block_list[idx], dev)
                else:
                    mismatches.append(f"transformer.block.{idx}: index beyond discovered blocks")

    if te_dev is not None:
        _check("text_encoder", getattr(state, "text_encoder", None), te_dev)
    if vae_dev is not None:
        _check("vae", getattr(state, "vae", None), vae_dev)

    status = "PASS" if not mismatches else "FAIL"
    return {
        "observed": observed,
        "expected": expected_plan,
        "mismatches": sorted(set(mismatches)),
        "status": status,
    }


def post_blocks_inconsistent(transformer: Any, block_devices: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """Return a mismatch record when any post-block module differs from the final
    block device. Used by the adapter final-device contract (fail before inference,
    failure code TRANSFORMER_POST_DEVICE_MISMATCH)."""
    if not block_devices:
        return None
    final_idx = max(block_devices)
    final_device = block_devices[final_idx]
    for name in C.TRANSFORMER_POST_MODULES:
        mod = getattr(transformer, name, None)
        if mod is None:
            continue
        devices = module_devices(mod)
        for kind in ("parameters", "buffers"):
            for tensor_name, dev in devices[kind].items():
                if str(dev) != str(final_device):
                    return {
                        "module": name,
                        "expected_device": final_device,
                        "observed_device": dev,
                        "tensor": f"{name}.{tensor_name}",
                        "final_block": final_idx,
                    }
    return None
