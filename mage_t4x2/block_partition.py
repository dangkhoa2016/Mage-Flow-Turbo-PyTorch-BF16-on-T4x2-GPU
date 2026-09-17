"""Contiguous block-level partitioning of the Mage DiT transformer across two
devices (block-wise model parallelism; deliberately NOT Tensor Parallel).

Everything here is CPU-computable and deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import constants as C


def discover_transformer_blocks(module: Any) -> Tuple[List[Any], str]:
    """Locate the transformer block container on a module.

    Probes a fixed ordered list of idiomatically-named attributes. Returns a
    (blocks, attribute_name) tuple. Raises ValueError when none is found.
    """
    for attr in C.BLOCK_ATTRIBUTE_NAMES:
        if hasattr(module, attr):
            container = getattr(module, attr)
            items = list(container) if not isinstance(container, dict) else list(container.values())
            if items:
                return items, attr
    raise ValueError(
        f"no transformer block container found; probed: {C.BLOCK_ATTRIBUTE_NAMES}"
    )


def plan_block_devices(
    num_blocks: int,
    device0: str = C.GPU0,
    device1: str = C.GPU1,
    strategy: str = "half",
    split_block: Optional[int] = None,
    memory_weights: Optional[Sequence[float]] = None,
) -> Tuple[Dict[int, str], int]:
    """Return (block->device map, split_point).

    Strategies:
      * ``half``      — half the blocks on device0, the rest on device1.
      * ``explicit``  — a specific ``split_block`` boundary.
      * ``weighted``  — choose the split that best balances memory_weights.
    """
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")

    if strategy == "explicit":
        if split_block is None:
            raise ValueError("explicit strategy requires split_block")
        split = int(split_block)
    elif strategy == "weighted":
        if memory_weights is None:
            raise ValueError("weighted strategy requires memory_weights")
        weights = [float(w) for w in memory_weights]
        if len(weights) != num_blocks:
            raise ValueError("memory_weights length must equal num_blocks")
        total = sum(weights)
        best = None
        for k in range(1, num_blocks):
            diff = abs(sum(weights[:k]) - sum(weights[k:]))
            if best is None or diff < best[0]:
                best = (diff, k)
        split = best[1] if best else num_blocks // 2
    else:  # half
        split = num_blocks // 2

    if split < 1 or split >= num_blocks:
        raise ValueError(f"split_block {split} out of range for {num_blocks} blocks")

    mapping: Dict[int, str] = {}
    for i in range(num_blocks):
        mapping[i] = device0 if i < split else device1
    return mapping, split


def adjacent_transfers(block_devices: Dict[int, str]) -> List[Dict[str, Any]]:
    """List boundary transfers between consecutive blocks on different devices."""
    transfers: List[Dict[str, Any]] = []
    for i in sorted(block_devices):
        if i + 1 in block_devices and block_devices[i] != block_devices[i + 1]:
            transfers.append(
                {
                    "from_block": i,
                    "to_block": i + 1,
                    "from_device": block_devices[i],
                    "to_device": block_devices[i + 1],
                }
            )
    return transfers


def canonical_device_map(
    text_encoder_device: str,
    vae_device: str,
    block_devices: Dict[int, str],
) -> Dict[str, Any]:
    """Assemble the canonical JSON device map document."""
    return {
        "text_encoder": text_encoder_device,
        "transformer": {str(k): v for k, v in sorted(block_devices.items())},
        "vae": vae_device,
    }