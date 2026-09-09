"""Deterministic dual-T4 device-plan generation (CPU-computable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import constants as C
from .block_partition import adjacent_transfers, canonical_device_map, plan_block_devices

# Placement namespaces inside a device_plan. The real schema (build_full_plan /
# build_device_plan) mixes these placement leaves with read-only metadata keys
# (topology / pos_embed / block_devices / final_block / ...); only the placement
# namespace is traversed by the flatten helpers.
_TRANSFORMER_NAMESPACES = ("pre", "blocks", "post")


class DevicePlanSchemaError(ValueError):
    """Unknown or unsupported device-plan structure. Fail closed — never guess."""


def build_device_plan(
    num_blocks: int,
    text_encoder_device: str = C.GPU0,
    vae_device: str = C.GPU1,
    strategy: str = "half",
    split_block: Optional[int] = None,
    memory_weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Build the full dual-T4 device plan for one T2I graph (Corrective A4.3).

    Topology:
        text encoder + transformer pre modules + blocks[0:K)  -> device0
        blocks[K:N) + transformer post modules + VAE          -> device1

    The plan encodes pre / blocks / post explicitly so the real backend places
    every stateful transformer module — not just the block loop. ``pos_embed``
    is recorded as ``RUNTIME_TENSOR_DEVICE`` when parameterless (this pure
    builder has no module to inspect), and placement helpers refine it at
    runtime when the actual module carries parameters/buffers.
    """
    block_devices, split = plan_block_devices(
        num_blocks,
        device0=text_encoder_device,
        device1=vae_device,
        strategy=strategy,
        split_block=split_block,
        memory_weights=memory_weights,
    )
    final_block = max(block_devices)
    final_device = block_devices[final_block]
    transformer = {
        "pre": {
            name: text_encoder_device
            for name in C.TRANSFORMER_PRE_MODULES
        },
        "blocks": {str(k): v for k, v in sorted(block_devices.items())},
        "post": {
            name: final_device
            for name in C.TRANSFORMER_POST_MODULES
        },
    }
    return {
        "topology": "dual_t4",
        "text_encoder": text_encoder_device,
        "vae": vae_device,
        "transformer": transformer,
        "pos_embed": "RUNTIME_TENSOR_DEVICE",
        "block_devices": {str(k): v for k, v in sorted(block_devices.items())},
        "final_block": final_block,
        "final_block_device": final_device,
        "num_transformer_blocks": num_blocks,
        "split_policy": strategy,
        "split_block": split,
        "device_map_json": canonical_device_map(
            text_encoder_device, vae_device, block_devices
        ),
        "transfers": adjacent_transfers(block_devices),
    }


def extract_block_devices(plan: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """Extract a flat ``{block_idx: device}`` map from any accepted plan shape.

    Supports:
      * A4.3 nested ``{"transformer": {"blocks": {...}}}`` (preferred)
      * A4.3 ``{"block_devices": {...}}``
      * legacy flat ``{"transformer": {block_idx: device}}``
    """
    if not plan:
        return {}
    transformer = plan.get("transformer")
    if isinstance(transformer, dict):
        blocks = transformer.get("blocks")
        if isinstance(blocks, dict):
            return {int(k): str(v) for k, v in blocks.items()}
        # legacy flat transformer map
        if blocks is None and all(not isinstance(v, dict) for v in transformer.values()):
            return {int(k): str(v) for k, v in transformer.items()}
    flat = plan.get("block_devices")
    if isinstance(flat, dict):
        return {int(k): str(v) for k, v in flat.items()}
    return {}


def infer_num_blocks_from_config(config_json: Dict[str, Any]) -> Optional[int]:
    """Discover the transformer block count from a diffusers-style config.json.

    Mage's NR-MMDiT stores the block depth under ``depth`` (optionally with a
    ``depth_single_blocks`` companion). Returns the total double-block depth.
    """
    depth = config_json.get("depth")
    if isinstance(depth, int) and depth > 0:
        return depth
    return None


def save_device_plan(plan: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_device_plan(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_device_plan_leaves(device_plan: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yield deterministic ``(path, device)`` pairs for every placement leaf.

    Walks the nested placement schema:

        text_encoder            -> leaf
        vae                     -> leaf
        transformer.pre.<name>  -> leaf (per pre-block module)
        transformer.blocks.<i>  -> leaf (block index, numeric order)
        transformer.post.<name> -> leaf (per post-block module)

    also accepting the legacy flat ``{"transformer": {idx: device}}`` shape.
    Block indices are always emitted in numeric order. Any unknown nested
    namespace and any non-string leaf value raises ``DevicePlanSchemaError``
    (fail closed); top-level metadata keys (topology, pos_embed, block_devices,
    ...) are skipped, never treated as placement leaves.
    """
    if not isinstance(device_plan, dict):
        raise DevicePlanSchemaError(
            f"device_plan must be a dict, got {type(device_plan).__name__}"
        )

    for key in ("text_encoder", "vae"):
        if key in device_plan:
            value = device_plan[key]
            if not isinstance(value, str):
                raise DevicePlanSchemaError(
                    f"{key}: device leaf must be a str, got {type(value).__name__}"
                )
            yield (key, value)

    transformer = device_plan.get("transformer")
    if transformer is None:
        return
    if not isinstance(transformer, dict):
        raise DevicePlanSchemaError(
            f"transformer: expected a dict namespace, got {type(transformer).__name__}"
        )
    if transformer and all(isinstance(v, str) for v in transformer.values()):
        # legacy flat transformer block map: {"block_idx": device}
        for idx in _sorted_numeric_keys(transformer, "transformer"):
            yield (f"transformer.{idx}", transformer[idx])
        return

    unknown = sorted(set(transformer) - set(_TRANSFORMER_NAMESPACES))
    if unknown:
        raise DevicePlanSchemaError(
            "transformer: unknown nested namespace(s): " + ", ".join(unknown)
        )

    pre = transformer.get("pre")
    if pre is not None:
        yield from _iter_prepost_leaves(pre, "transformer.pre")

    blocks = transformer.get("blocks")
    if blocks is not None:
        if not isinstance(blocks, dict):
            raise DevicePlanSchemaError(
                f"transformer.blocks: expected dict of index->device, got {type(blocks).__name__}"
            )
        for idx in _sorted_numeric_keys(blocks, "transformer.blocks"):
            yield (f"transformer.blocks.{idx}", _coerce_device_leaf(blocks, idx, "transformer.blocks"))

    post = transformer.get("post")
    if post is not None:
        yield from _iter_prepost_leaves(post, "transformer.post")


def _iter_prepost_leaves(value: Any, prefix: str) -> Iterator[Tuple[str, str]]:
    """Emit pre/post leaves for either supported form:

      * dict of module->device (the real planner schema) -> per-module leaves
      * flat device string (conceptual schema)           -> one group leaf
    """
    if isinstance(value, str):
        yield (prefix, value)
        return
    if isinstance(value, dict):
        for name in sorted(value):
            yield (f"{prefix}.{name}", _coerce_device_leaf(value, name, prefix))
        return
    raise DevicePlanSchemaError(
        f"{prefix}: expected dict of module->device or a flat device str, "
        f"got {type(value).__name__}"
    )


def flatten_device_plan(device_plan: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return the deterministic list of ``(path, device)`` placement leaves."""
    return list(iter_device_plan_leaves(device_plan))


def _coerce_device_leaf(mapping: Dict[str, Any], key: Any, prefix: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise DevicePlanSchemaError(
            f"{prefix}.{key}: device leaf must be a str, got {type(value).__name__}"
        )
    return value


def _sorted_numeric_keys(mapping: Dict[Any, Any], prefix: str) -> List[Any]:
    try:
        return sorted(mapping, key=lambda k: int(k))
    except (TypeError, ValueError) as exc:
        raise DevicePlanSchemaError(f"{prefix}: block keys must be numeric: {exc}") from exc
