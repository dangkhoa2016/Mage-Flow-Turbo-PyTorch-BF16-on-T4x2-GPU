"""Dual-device forward adapter for the Mage-Flow-Turbo transformer.

Shaped around the exact upstream forward contract frozen in A2.1:

    MageFlow.forward(img, txt, timesteps, img_shapes, img_cu_seqlens,
                     txt_cu_seqlens, attention_kwargs)
        -> pre-block: pos_embed / img_in / txt_norm / time_text_embed / txt_in
        -> for block in self.transformer_blocks:  (python loop)
             txt, img = block(hidden_states=img, encoder_hidden_states=txt, ...)
        -> post-block: norm_out / proj_out

The adapter replicates that forward but partitions the block loop across two
devices with an explicit boundary payload transfer that includes the SHARED
conditioning tensors (``temb``, ``ms_pe``, ``txt_cu_seqlens``, ``img_cu_seqlens``)
because each subsequent block needs them on the same device.

Import-safety: ``import mage_t4x2.dual_device_forward`` must not initialize CUDA.
``torch`` is imported lazily inside functions.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

from .block_partition import discover_transformer_blocks
from . import constants as C

# ---------------------------------------------------------------------------
# Recursive boundary payload mover
# ---------------------------------------------------------------------------


def move_payload(value: Any, device: Any) -> Any:
    """Recursively move every torch tensor inside ``value`` to ``device``.

    Supports Tensor, tuple, list, dict, None, scalars, and dataclasses (dataclass
    fields are moved in place only for tensor fields, keeping metadata scalars
    untouched).
    """
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_payload(v, device) for v in value)
    if isinstance(value, list):
        return [move_payload(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_payload(v, device) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Move tensor fields only; preserve scalar/metadata fields.
        fields = dataclasses.fields(value)
        updates: Dict[str, Any] = {}
        for f in fields:
            current = getattr(value, f.name)
            if isinstance(current, (torch.Tensor, tuple, list, dict)):
                updates[f.name] = move_payload(current, device)
            elif dataclasses.is_dataclass(current) and not isinstance(current, type):
                updates[f.name] = move_payload(current, device)
        if updates:
            import dataclasses as dc

            return dc.replace(value, **updates)
        return value
    # Scalars / python metadata: leave untouched.
    return value


def payload_tensor_count(value: Any) -> int:
    """Count every torch.Tensor inside ``value`` (recursive)."""
    import torch

    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, (tuple, list)):
        return sum(payload_tensor_count(v) for v in value)
    if isinstance(value, dict):
        return sum(payload_tensor_count(v) for v in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            payload_tensor_count(getattr(value, f.name))
            for f in dataclasses.fields(value)
        )
    return 0


def payload_bytes(value: Any) -> int:
    """Total bytes of all torch tensors inside ``value`` (recursive)."""
    import torch

    if isinstance(value, torch.Tensor):
        try:
            return value.numel() * value.element_size()
        except Exception:
            return 0
    if isinstance(value, (tuple, list)):
        return sum(payload_bytes(v) for v in value)
    if isinstance(value, dict):
        return sum(payload_bytes(v) for v in value.values())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(
            payload_bytes(getattr(value, f.name))
            for f in dataclasses.fields(value)
        )
    return 0


# ---------------------------------------------------------------------------
# Cross-device boundary transfer
# ---------------------------------------------------------------------------


class CrossDeviceTransferHelper:
    """Explicit boundary transfer with evidence events.

    The tensor_count / bytes fields are computed from the actual payload on
    EVERY transfer; never hard-coded.
    """

    def __init__(
        self,
        telemetry: Any = None,
        run_id: str = "n/a",
        phase: str = "n/a",
        inference_id: str = "n/a",
        mover: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self.telemetry = telemetry
        self.run_id = run_id
        self.phase = phase
        self.inference_id = inference_id
        self._mover = mover or move_payload
        self.events: List[Dict[str, Any]] = []

    def move(
        self,
        payload: Any,
        from_device: str,
        to_device: str,
        boundary: str = "?->?",
    ) -> Any:
        moved = self._mover(payload, to_device)
        tensor_count = payload_tensor_count(payload)
        bytes_moved = payload_bytes(payload)
        event: Dict[str, Any] = {
            "event": "cross_device_transfer",
            "run_id": self.run_id,
            "phase": self.phase,
            "inference_id": self.inference_id,
            "from": from_device,
            "to": to_device,
            "block_boundary": boundary,
            "tensor_count": tensor_count,
            "bytes": bytes_moved,
        }
        self.events.append(event)
        if self.telemetry is not None:
            self.telemetry.event(
                "cross_device_transfer",
                run_id=self.run_id,
                phase=self.phase,
                inference_id=self.inference_id,
                from_=from_device,
                to=to_device,
                block_boundary=boundary,
                tensor_count=tensor_count,
                bytes=bytes_moved,
            )
        return moved


# ---------------------------------------------------------------------------
# Device-aware activation (used by synthetic CPU qualification)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DeviceTaggedTensor:
    """Wraps a torch tensor with a logical device label.

    Used ONLY by synthetic CPU tests to prove routing without CUDA.
    ``to(device)`` retags the label; the underlying tensor stays put.
    """

    tensor: Any
    device: str = "logical_gpu_0"

    def to(self, device: Any) -> "DeviceTaggedTensor":
        return DeviceTaggedTensor(self.tensor, str(device))

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype


def tag_payload_mover(value: Any, device: Any) -> Any:
    """Mover used by synthetic CPU tests: retag DeviceTaggedTensor labels."""
    if isinstance(value, DeviceTaggedTensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(tag_payload_mover(v, device) for v in value)
    if isinstance(value, list):
        return [tag_payload_mover(v, device) for v in value]
    if isinstance(value, dict):
        return {k: tag_payload_mover(v, device) for k, v in value.items()}
    import torch

    if isinstance(value, torch.Tensor):
        return value
    return value
