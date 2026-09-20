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


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class DualDeviceForwardAdapter:
    """Intercepts one upstream transformer forward and executes a contiguous
    block partition across two devices with an explicit boundary transfer.

    Pattern: monkey-patched bound forward (least invasive). Original forward is
    restored on ``detach``.
    """

    def __init__(
        self,
        transformer: Any,
        block_devices: Dict[int, str],
        transfer_helper: Optional[CrossDeviceTransferHelper] = None,
        telemetry: Any = None,
        run_id: str = "n/a",
        phase: str = "n/a",
        inference_id: str = "n/a",
        block_attr: Optional[str] = None,
    ) -> None:
        self.transformer = transformer
        self.block_devices = {int(k): str(v) for k, v in block_devices.items()}
        blocks, attr = discover_transformer_blocks(transformer)
        self.blocks = blocks
        self.block_attr = attr if block_attr is None else block_attr
        self.telemetry = telemetry
        self.transfer_helper = transfer_helper or CrossDeviceTransferHelper(
            telemetry=telemetry,
            run_id=run_id,
            phase=phase,
            inference_id=inference_id,
        )
        self._original_forward = None
        self.attached = False
        self.block_events: List[Dict[str, Any]] = []

    # -- device helpers ----------------------------------------------------
    def _device_of(self, idx: int) -> str:
        return self.block_devices[idx]

    def _log_device(self) -> str:
        """Return the logical device for telemetry (proxy: cuda vs logical)."""
        # In a real run the caller passes actual devices; this is a structural stub.
        return "?"

    # -- attaching ----------------------------------------------------------
    def attach(self) -> bool:
        if self.attached:
            return True
        if not self.block_devices:
            raise RuntimeError("dual-device adapter needs a non-empty block_devices map")
        import types

        # The wrapped forward must keep ``self`` = the ADAPTER (its body reads
        # adapter attributes: transformer, blocks, transfer_helper, ...). The
        # bound method is already bound to the adapter, so any further
        # ``MethodType`` binding would shift every argument by one.
        self._original_forward = self.transformer.forward
        self.transformer.forward = self._wrapped_forward
        self.attached = True
        return True

    def detach(self) -> bool:
        if self._original_forward is not None and self.attached:
            self.transformer.forward = self._original_forward
        self.attached = False
        return True

    # -- telemetry -----------------------------------------------------------
    def _emit_block(self, idx: int, device: str) -> None:
        event: Dict[str, Any] = {
            "event": "block_forward",
            "run_id": self.transfer_helper.run_id,
            "phase": self.transfer_helper.phase,
            "inference_id": self.transfer_helper.inference_id,
            "block": idx,
            "device": device,
            "dtype": "not-sampled",
        }
        self.block_events.append(event)
        if self.telemetry is not None:
            self.transfer_helper.telemetry.event(
                "block_forward",
                run_id=self.transfer_helper.run_id,
                phase=self.transfer_helper.phase,
                inference_id=self.transfer_helper.inference_id,
                block=idx,
                device=device,
            )

    def _emit_return(self, source: str, target: str) -> None:
        """Emit the transformer-output return evidence (cuda:1 -> caller)."""
        if self.telemetry is not None:
            self.transfer_helper.telemetry.event(
                "transformer_output_return_transfer",
                run_id=self.transfer_helper.run_id,
                phase=self.transfer_helper.phase,
                inference_id=self.transfer_helper.inference_id,
                from_=source,
                to=target,
            )

    # -- forward -------------------------------------------------------------
    def _wrapped_forward(
        self,
        img: Any,
        txt: Any,
        timesteps: Any,
        img_shapes=None,
        img_cu_seqlens=None,
        txt_cu_seqlens=None,
        attention_kwargs=None,
    ) -> Any:
        """Mirrors upstream MageFlow.forward (mage_flow.py:93-153) but routes
        the block loop across ``block_devices``. The prediction is returned on
        the caller device with an auditable ``transformer_output_return_transfer``
        event whenever the final transformer device differs from the caller."""
        import torch

        tr = self.transformer
        # Fail closed before touching the transformer: the caller device must
        # be determinable from the actual input object state (never config).
        caller_device = resolve_device_label(img)
        three_d = _has_ndim_3(img) and _has_ndim_3(txt)
        if three_d is False:
            return self._original_forward(
                img, txt, timesteps,
                img_shapes, img_cu_seqlens, txt_cu_seqlens, attention_kwargs,
            )

        # ---- pre-block (matches upstream exactly) ---------------------------
        ms_pe = tr.pos_embed(img_shapes, device=_payload_device(img))
        img = tr.img_in(img)
        txt = tr.txt_norm(txt)
        timesteps = timesteps.to(_payload_dtype(img))
        temb = tr.time_text_embed(timesteps, img)
        txt = tr.txt_in(txt)
        txt_vec = torch.zeros(
            _payload_shape0(txt), tr.inner_dim,
            dtype=_payload_dtype(txt), device=_payload_device_actual(txt),
        )
        temb = temb + txt_vec
        attention_kwargs = attention_kwargs or {}

        # ---- block loop with boundary transfer ------------------------------
        running_payload = (txt, img)
        prev_device: Optional[str] = None
        for idx, block in enumerate(self.blocks):
            target = self._device_of(idx)
            if prev_device is not None and target != prev_device:
                boundary = f"{idx - 1}->{idx}"
                running_payload = self.transfer_helper.move(
                    running_payload,
                    from_device=prev_device,
                    to_device=target,
                    boundary=boundary,
                )
                # Shared conditioning must follow the payload to the target device.
                temb = self.transfer_helper._mover(temb, target)
                ms_pe = self.transfer_helper._mover(ms_pe, target)
                img_cu_seqlens = self.transfer_helper._mover(img_cu_seqlens, target)
                txt_cu_seqlens = self.transfer_helper._mover(txt_cu_seqlens, target)
            txt, img = running_payload
            if self._training_with_checkpoint():
                txt, img = torch.utils.checkpoint.checkpoint(
                    block, img, txt, temb, ms_pe, txt_cu_seqlens, img_cu_seqlens,
                    use_reentrant=False,
                )
            else:
                txt, img = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    txt_cu_lens=txt_cu_seqlens,
                    img_cu_lens=img_cu_seqlens,
                    temb=temb,
                    image_rotary_emb=ms_pe,
                    joint_attention_kwargs=attention_kwargs,
                )
            running_payload = (txt, img)
            self._emit_block(idx, target)
            prev_device = target

        # ---- post-block (matches upstream) -----------------------------------
        txt, img = running_payload
        img = tr.norm_out(img, temb, cu_seqlens=img_cu_seqlens)
        img = tr.proj_out(img)

        # ---- return bridge: prediction must land on the caller device --------
        final_device = resolve_device_label(img)
        if str(final_device) != str(caller_device):
            moved = self.transfer_helper._mover(img, caller_device)
            moved_device = resolve_device_label(moved)
            if str(moved_device) != str(caller_device):
                raise RuntimeError(
                    f"TRANSFORMER_RETURN_MOVE_FAILED: expected {caller_device}, "
                    f"observed {moved_device}"
                )
            self._emit_return(str(final_device), caller_device)
            return moved
        return img

    def _training_with_checkpoint(self) -> bool:
        tr = self.transformer
        return bool(getattr(tr, "training", False)) and bool(getattr(tr, "checkpoint", False))


# ---------------------------------------------------------------------------
# Structural verification helpers
# ---------------------------------------------------------------------------


def _has_ndim_3(value: Any) -> Optional[bool]:
    if hasattr(value, "ndim"):
        return value.ndim == 3
    if hasattr(value, "tensor"):
        return getattr(value.tensor, "ndim", None) == 3
    return None


def _payload_device(value: Any) -> str:
    dev = getattr(value, "device", None)
    if dev is not None:
        return str(dev)
    tag = getattr(value, "device", None)
    return str(tag)


def _payload_device_actual(value: Any) -> Any:
    import torch

    dev = getattr(value, "device", None)
    if isinstance(dev, torch.device) or isinstance(dev, str) and dev.startswith(("cuda", "cpu", "meta")):
        return dev
    return torch.device("cpu")


def _payload_dtype(value: Any) -> Any:
    import torch

    dt = getattr(value, "dtype", None)
    if isinstance(dt, torch.dtype):
        return dt
    tensor = getattr(value, "tensor", None)
    if tensor is not None and hasattr(tensor, "dtype"):
        return tensor.dtype
    return torch.float32


def _payload_shape0(value: Any) -> int:
    shp = getattr(value, "shape", None)
    if shp is not None:
        return int(shp[0])
    return 0


def block_execution_summary(events: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    return [(int(e["block"]), str(e["device"])) for e in events]


def transfer_summary(events: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    return [(str(e["from"]), str(e["to"]), str(e["block_boundary"])) for e in events]
