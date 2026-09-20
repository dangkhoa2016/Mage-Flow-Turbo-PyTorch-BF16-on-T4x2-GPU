"""JSONL telemetry recording and parsing.

Format matches the directive: each line is one JSON object such as
``{"event":"block_forward","block":12,"device":"cuda:0","dtype":"torch.bfloat16"}``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class TelemetryRecorder:
    def __init__(
        self,
        path: str,
        run_id: str = "n/a",
        phase: str = "n/a",
        inference_id: Optional[str] = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self._phase = phase
        self.inference_id = inference_id
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def event(self, event: str, **fields: Any) -> None:
        record: Dict[str, Any] = {"event": event}
        for key, value in fields.items():
            record["from" if key == "from_" else key] = value
        # Every event carries the common schema: timestamp/run_id/phase/inference_id.
        if "timestamp" not in record:
            record["timestamp"] = time.time()
        record.setdefault("run_id", self.run_id)
        record.setdefault("phase", self._phase)
        if self.inference_id is not None:
            record.setdefault("inference_id", self.inference_id)
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def block_forward(self, block: int, device: str, dtype: str) -> None:
        self.event("block_forward", block=block, device=device, dtype=dtype)

    def transfer(
        self,
        from_: str,
        to: str,
        shape: Optional[List[int]] = None,
        dtype: Optional[str] = None,
        boundary: Optional[str] = None,
    ) -> None:
        self.event(
            "transfer",
            from_=from_,
            to=to,
            shape=shape,
            dtype=dtype,
            block_boundary=boundary,
        )

    def enqueue_model_load(self, load_index: int) -> None:
        self.event("model_load", load_index=load_index)

    def phase(self, name: str, status: str) -> None:
        self.event("phase", phase=name, status=status)

    def metric(self, name: str, value: Any) -> None:
        self.event("metric", metric=name, value=value)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TelemetryRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def parse_telemetry(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid telemetry line {line_no}: {line!r}") from exc
    return records


def event_timeline(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Render a human/assertion-friendly event sequence (stable order)."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("event") == "block_forward":
            out.append({"kind": "forward", "component": rec.get("component"), "block": rec.get("block"), "device": rec.get("device")})
        elif rec.get("event") == "transfer":
            out.append({"kind": "transfer", "from": rec.get("from"), "to": rec.get("to")})
    return out
