"""Model-parallel routing for one T2I inference graph spanning two devices.

Pure routing logic lives in ``plan_route`` and is fully CPU-testable via
``MockGraphRunner``. ``ModelParallelWrapper`` is the real torch vehicle used
during the GPU session.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional


@dataclasses.dataclass(frozen=True)
class RouteStep:
    kind: str  # "forward" | "transfer"
    component: str  # "text_encoder" | "transformer.3" | "vae"
    block_index: Optional[int]
    device: Optional[str]  # set on forward steps
    from_component: Optional[str] = None  # set on transfer steps
    from_device: Optional[str] = None
    to_device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "component": self.component,
            "block_index": self.block_index,
            "device": self.device,
            "from_component": self.from_component,
            "from_device": self.from_device,
            "to_device": self.to_device,
        }


def _flatten_device_map(device_map: Dict[str, Any]) -> Dict[str, str]:
    """Accept nested (text_encoder / transformer{i} / vae) or flat maps."""
    flat: Dict[str, str] = {}
    for key, value in device_map.items():
        if key == "transformer" and isinstance(value, dict):
            for idx, dev in value.items():
                flat[f"transformer.{idx}"] = str(dev)
        elif isinstance(value, str):
            flat[key] = value
    return flat


def _canonical_order(flat: Dict[str, str]) -> List[str]:
    order: List[str] = []
    for prefix in ("tokenizer", "text_encoder"):
        if prefix in flat:
            order.append(prefix)
    order.extend(f"transformer.{i}" for i in sorted(_transformer_indices(flat)))
    for suffix in ("scheduler", "vae"):
        if suffix in flat:
            order.append(suffix)
    return order


def _transformer_indices(flat: Dict[str, str]) -> List[int]:
    indices: List[int] = []
    for key in flat:
        if key.startswith("transformer."):
            try:
                indices.append(int(key.split(".", 1)[1]))
            except ValueError:
                continue
    return sorted(set(indices))


def plan_route(device_map: Dict[str, Any]) -> List[RouteStep]:
    """Compile the route (forward + transfer steps) for one inference graph."""
    flat = _flatten_device_map(device_map)
    order = _canonical_order(flat)
    if not order:
        return []

    steps: List[RouteStep] = []
    prev_key = None
    prev_device = None
    for key in order:
        device = flat[key]
        if prev_device is not None and device != prev_device:
            steps.append(
                RouteStep(
                    kind="transfer",
                    component=key,
                    block_index=_block_index(key),
                    device=device,
                    from_component=prev_key,
                    from_device=prev_device,
                    to_device=device,
                )
            )
        steps.append(
            RouteStep(
                kind="forward",
                component=key,
                block_index=_block_index(key),
                device=device,
            )
        )
        prev_key = key
        prev_device = device
    return steps


def _block_index(component: str) -> Optional[int]:
    if component.startswith("transformer."):
        try:
            return int(component.split(".", 1)[1])
        except ValueError:
            return None
    return None


class MockGraphRunner:
    """Executes a route over synthetic callables on logical devices.

    Events are recorded exactly like GPU telemetry and asserted by CPU tests.
    """

    def __init__(self, mover: Any, telemetry: Optional["TelemetryRecorder"] = None) -> None:  # type: ignore[name-defined]
        self.mover = mover
        self.telemetry = telemetry
        self.events: List[Dict[str, Any]] = []
        self.payload_device: Optional[str] = None

    def execute(self, plan: List[RouteStep], components: Dict[str, Callable[[Any, str], Any]]) -> Any:
        payload: Any = None
        for step in plan:
            if step.kind == "forward":
                dev = step.device or "?"
                payload = components[step.component](payload, dev)
                event = {
                    "event": "forward",
                    "component": step.component,
                    "block": step.block_index,
                    "device": dev,
                }
                self.events.append(event)
                self.payload_device = dev
                if self.telemetry is not None:
                    self.telemetry.event(
                        "block_forward",
                        block=step.block_index,
                        component=step.component,
                        device=dev,
                    )
            elif step.kind == "transfer":
                moved = self.mover.move(payload, step.from_device or "?", step.to_device or "?")
                event = {
                    "event": "transfer",
                    "from": step.from_device,
                    "to": step.to_device,
                    "from_component": step.from_component,
                    "to_component": step.component,
                }
                self.events.append(event)
                self.payload_device = step.to_device
                if self.telemetry is not None:
                    self.telemetry.event(
                        "transfer",
                        from_=step.from_device,
                        to=step.to_device,
                    )
                payload = moved
        return payload

    def to_json(self) -> str:
        return json.dumps(self.events, indent=2, sort_keys=True)


class ModelParallelWrapper:
    """Real torch vehicle used during the GPU session.

    Splits a transformer's block container along the compiled route. This class
    is only constructed with an actual model on the GPU session; nothing here
    runs during CPU Stage A.
    """

    def __init__(
        self,
        blocks,
        device_map: Dict[str, Any],
        mover: Any,
        telemetry: Optional[Any] = None,
    ) -> None:
        self.blocks = list(blocks)
        self.device_map = device_map
        self.mover = mover
        self.telemetry = telemetry
        self.route = plan_route(device_map)

    def forward(self, x, **kwargs):
        current = x
        for step in self.route:
            if step.kind == "transfer":
                current = self.mover.move(current, step.from_device or "", step.to_device or "")
                if self.telemetry is not None:
                    self.telemetry.event("transfer", from_=step.from_device, to=step.to_device)
                continue
            if step.block_index is None:
                continue
            block = self.blocks[step.block_index]
            import torch

            current = block.to(torch.device(step.device or ""))(
                current.to(torch.device(step.device or "")), **kwargs
            )
            if self.telemetry is not None:
                dtype = str(current.dtype) if hasattr(current, "dtype") else "n/a"
                self.telemetry.event(
                    "block_forward",
                    block=step.block_index,
                    device=step.device,
                    dtype=dtype,
                )
        return current
