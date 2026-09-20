"""Cross-device tensor transfer abstractions.

The real GPU session uses ``CudaTensorMover``; CPU-only tests use
``MockTensorMover`` so the routing logic is provable without CUDA.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class MockPayload:
    """An opaque payload whose logical device is tracked without touching CUDA."""

    tensor: Any
    device: str = "logical_gpu_0"

    def to_dict(self) -> Dict[str, Any]:
        return {"device": self.device, "tensor_type": type(self.tensor).__name__}


class MockTensorMover:
    """Records moves and re-tags a MockPayload's logical device."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def move(self, payload: MockPayload, from_device: str, to_device: str) -> MockPayload:
        payload.device = to_device
        self.events.append(
            {
                "event": "transfer",
                "from": from_device,
                "to": to_device,
                "payload_device_after": to_device,
            }
        )
        return payload


class CudaTensorMover:
    """Real CUDA mover used only on the GPU session (never imported on CPU stage except
    inside this class)."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def move(self, tensor, from_device: str, to_device: str):
        import torch

        moved = tensor.to(torch.device(to_device))
        self.events.append(
            {
                "event": "transfer",
                "from": from_device,
                "to": to_device,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
        return moved


class TransferPlanner:
    """Plan boundary transfers from a device map."""

    @staticmethod
    def from_device_map(device_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        transfers: List[Dict[str, Any]] = []
        order: List[str] = []
        if "text_encoder" in device_map:
            order.append("text_encoder")
        transformer = device_map.get("transformer", {})
        for idx in sorted(transformer, key=int):
            order.append(f"transformer.{idx}")
        if "vae" in device_map:
            order.append("vae")

        for i in range(len(order) - 1):
            src, dst = order[i], order[i + 1]
            from_dev = _device_of(device_map, src)
            to_dev = _device_of(device_map, dst)
            if from_dev != to_dev:
                transfers.append(
                    {
                        "from_component": src,
                        "to_component": dst,
                        "from_device": from_dev,
                        "to_device": to_dev,
                    }
                )
        return transfers


def _device_of(device_map: Dict[str, Any], key: str) -> str:
    if key.startswith("transformer."):
        idx = key.split(".", 1)[1]
        return str(device_map["transformer"][str(idx)])
    return str(device_map[key])
