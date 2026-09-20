"""G1 corrective — explicit VAE input device bridge (device_bridges).

Boundary B (runbook section 10/11/19/20):

    final latent   cuda:0
    VAE            cuda:1

    final latent cuda:0
    -> vae_input_transfer cuda:0 -> actual VAE device cuda:1
    -> VAE decode cuda:1
    -> CPU/PIL output

The bridge is bound by reference to the LIVE VAE module (monkey-patched
``decode``), so it can be attached to an already-loaded model without any
weight reload or rematerialization.

Fail-closed rules:

    * the actual VAE device is ALWAYS derived from live module state
      (parameters/buffers, then the module ``.device`` attribute) — never
      hard-coded and never taken from pipeline/planner intent;
    * an ambiguous/unknown/meta device raises before any decode;
    * after a bridge move the payload device is verified empirically; a
      failed move raises.

Import-safety: importing this module never initializes CUDA (torch is
imported lazily inside functions).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .dual_device_forward import move_payload, payload_bytes, payload_tensor_count


def resolve_device_label(value: Any) -> str:
    """Return the canonical device label of a torch tensor or logical
    (DeviceTaggedTensor) wrapper. Raises when it cannot be determined.

    The source device is ALWAYS taken from the actual object state.
    """
    dev = getattr(value, "device", None)
    if dev is None:
        raise RuntimeError(
            "DEVICE_CONTRACT_DEVICE_UNKNOWN: object carries no device; "
            "cannot satisfy the device contract"
        )
    return str(dev)


def derive_vae_device(vae: Any) -> str:
    """Actual device of a LIVE VAE module derived from its own state.

    Inspection order (fail-closed):
      1. parameter devices (every one, recurse=True);
      2. buffer devices;
      3. fallback: the module ``.device`` attribute (synthetic/fake modules).

    Raises VAE_DEVICE_UNKNOWN / VAE_DEVICE_AMBIGUOUS / VAE_DEVICE_META when the
    device cannot be determined unambiguously. Never returns a hard-coded value.
    """
    seen: List[str] = []
    if vae is None:
        raise RuntimeError("VAE_DEVICE_UNKNOWN: VAE module is None")
    if hasattr(vae, "named_parameters"):
        for _name, p in vae.named_parameters(recurse=True):
            seen.append(str(p.device))
    if hasattr(vae, "named_buffers"):
        for _name, b in vae.named_buffers(recurse=True):
            seen.append(str(b.device))
    if not seen:
        own = getattr(vae, "device", None)
        if own is not None:
            return str(own)
        raise RuntimeError("VAE_DEVICE_UNKNOWN: no parameters/buffers and no module device observed")
    distinct: List[str] = sorted(set(seen))
    if len(distinct) != 1:
        raise RuntimeError(f"VAE_DEVICE_AMBIGUOUS: observed devices {distinct}")
    if distinct[0].startswith("meta"):
        raise RuntimeError("VAE_DEVICE_META: VAE parameters live on meta")
    return distinct[0]


class VaeInputBridge:
    """Binds an explicit latent->VAE device transfer around ``vae.decode``.

    ``decode(z, ...)`` is wrapped so every decode entry receives a payload on
    the ACTUAL VAE device. Transfers are auditable via ``vae_input_transfer``
    events (moved through the existing project mover abstraction).
    """

    def __init__(
        self,
        vae: Any,
        telemetry: Any = None,
        run_id: str = "n/a",
        phase: str = "n/a",
        inference_id: str = "n/a",
        mover: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self.vae = vae
        # Fail closed BEFORE any decode: the real VAE device must be derivable.
        self.vae_device: str = derive_vae_device(vae)
        self.telemetry = telemetry
        self.run_id = run_id
        self.phase = phase
        self.inference_id = inference_id
        self._mover = mover or move_payload
        self._original_decode: Optional[Callable[..., Any]] = None
        self.attached = False
        self.events: List[Dict[str, Any]] = []

    def attach(self) -> bool:
        if self.attached:
            raise RuntimeError("VAE_BRIDGE_DOUBLE_ATTACH: VAE bridge already attached")
        if not callable(getattr(self.vae, "decode", None)):
            raise RuntimeError("VAE_DECODE_MISSING: VAE has no decode callable")
        self._original_decode = self.vae.decode
        self.vae.decode = self._decode  # type: ignore[method-assign]
        self.attached = True
        return True

    def detach(self) -> bool:
        if self._original_decode is not None and self.attached:
            self.vae.decode = self._original_decode  # type: ignore[method-assign]
        self.attached = False
        return True

    # -- telemetry --------------------------------------------------------
    def _emit(self, source: str, target: str, payload: Any) -> None:
        event: Dict[str, Any] = {
            "event": "vae_input_transfer",
            "run_id": self.run_id,
            "phase": self.phase,
            "inference_id": self.inference_id,
            "from": source,
            "to": target,
            "tensor_count": payload_tensor_count(payload),
            "bytes": payload_bytes(payload),
        }
        self.events.append(event)
        if self.telemetry is not None:
            self.telemetry.event(
                "vae_input_transfer",
                run_id=self.run_id,
                phase=self.phase,
                inference_id=self.inference_id,
                from_=source,
                to=target,
                tensor_count=payload_tensor_count(payload),
                bytes=payload_bytes(payload),
            )

    # -- decode wrapper -----------------------------------------------------
    def _decode(self, z: Any, *args: Any, **kwargs: Any) -> Any:
        source = resolve_device_label(z)
        if source != self.vae_device:
            moved = self._mover(z, self.vae_device)
            moved_device = resolve_device_label(moved)
            if moved_device != self.vae_device:
                raise RuntimeError(
                    f"VAE_BRIDGE_MOVE_FAILED: expected {self.vae_device}, observed {moved_device}"
                )
            self._emit(source, self.vae_device, z)
            z = moved
        return self._original_decode(z, *args, **kwargs)
