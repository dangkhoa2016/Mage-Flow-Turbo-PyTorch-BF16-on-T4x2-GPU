"""Injectable phase backends for CPU-only authority qualification.

``SyntheticPhaseBackend`` is a deterministic fake backend that implements the
same API surface as the real GPU backend (``scripts.gpu_session``) so the
``StageBSession`` phase runners can be qualified on CPU without CUDA and without
loading the real 20 GB model. It simulates both successful and failing inference
scenarios (Corrective A3 section 16).

Import-safety: importing this module must never initialize CUDA.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import image_validation
from .constants import TRANSFORMER_POST_MODULES
from .contracts import RunContract
from .device_plan import extract_block_devices
from .dual_device_forward import (
    CrossDeviceTransferHelper,
    DualDeviceForwardAdapter,
    SyntheticTransformer,
    _cpu_mover,
)
from .evidence import EvidenceRun
from .telemetry import TelemetryRecorder

GPU0 = "cuda:0"
GPU1 = "cuda:1"

SCENARIOS = (
    "success",
    "run_t2i_exception",
    "no_inference",
    "invalid_output",
    "missing_gpu1",
    "missing_transfer",
    "duplicate_block",
    "skipped_block",
    "wrong_inference_id",
    "dtype_fail",
    "second_model_load",
    "adapter_missing",
    "post_device_mismatch",
)


class _CaptureTelemetry:
    """In-memory telemetry used during the capture pass so scenarios can
    transform evidence before it is written to the real JSONL file."""

    def __init__(self, run_id: str, phase: str, inference_id: str) -> None:
        self.records: List[Dict[str, Any]] = []
        self.run_id = run_id
        self.phase = phase
        self._phase = phase
        self.inference_id = inference_id

    def event(self, event: str, **fields: Any) -> None:
        record: Dict[str, Any] = {"event": event}
        for key, value in fields.items():
            record["from" if key == "from_" else key] = value
        record.setdefault("run_id", self.run_id)
        record.setdefault("phase", self.phase)
        if self.inference_id is not None:
            record.setdefault("inference_id", self.inference_id)
        self.records.append(record)

    def phase(self, name: str, status: str) -> None:
        self.event("phase", phase=name, status=status)

    def metric(self, name: str, value: Any) -> None:
        self.event("metric", metric=name, value=value)


class SyntheticRuntimeState:
    """Minimal live runtime state used by the synthetic backend.

    Mirrors the real ``scripts.gpu_session.RuntimeState`` surface that the phase
    runners inspect (current_profile / current_topology / transformer / etc.).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        contract: Any = None,
        source_authority: Optional[Dict[str, Any]] = None,
        model_load_count: int = 1,
        depth: int = 24,
    ) -> None:
        self.transformer = SyntheticTransformer(depth=depth, dim=8)
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.tokenizer = None
        self.model_path = model_path
        self.contract = contract
        self.source_authority = source_authority
        self.current_profile = "unconfigured"
        self.current_topology = "unconfigured"
        self.device_map: Dict[str, Any] = {}
        self.dual_forward_adapter = None
        self.dual_forward_adapter_attached = False
        self.model_load_count = model_load_count
        self.memory_plan: Dict[str, Any] = {}
        self.placement_validation: Optional[Dict[str, Any]] = None
        self.no_full_model_cuda0_materialization = True

    def snapshot(self) -> Dict[str, str]:
        return {
            "current_profile": self.current_profile,
            "current_topology": self.current_topology,
        }


class SyntheticPhaseBackend:
    """Deterministic fake backend implementing the PhaseBackendProtocol.

    Every method is reachable through ``backend["name"](...)`` (mapping style)
    to match the session's existing backend convention.
    """

    def __init__(self, scenario: str = "success") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown synthetic scenario: {scenario!r}")
        self.scenario = scenario
        self.run_t2i_call_count = 0
        self.phase_inference_calls: Dict[str, int] = {}
        self.load_calls = 0
        self.attach_count = 0
        self.detach_count = 0
        self.attach_attempts = 0
        self.adapters: List[Any] = []
        self.state_ids: List[int] = []
        self.inference_ids: Dict[str, str] = {}
        self.output_hashes: Dict[str, Optional[str]] = {}
        self.block_devices_per_phase: Dict[str, Dict[int, str]] = {}
        self.detach_reasons: List[str] = []
        self.loaded_states: List[Any] = []

    # -- mapping-style backend access --------------------------------------
    def __getitem__(self, key: str):
        table: Dict[str, Any] = {
            "load_runtime_state": self.load_runtime_state,
            "preflight_gpu": self.preflight_gpu,
            "apply_dual_t4_mixed": self.apply_dual_t4_mixed,
            "apply_dual_t4_all_bf16": self.apply_dual_t4_all_bf16,
            "create_evidence_run": self.create_evidence_run,
            "create_telemetry": self.create_telemetry,
            "attach_dual_adapter": self.attach_dual_adapter,
            "detach_dual_adapter": self.detach_dual_adapter,
            "run_t2i": self.run_t2i,
            "collect_telemetry": self.collect_telemetry,
            "derive_phase_facts": self.derive_phase_facts,
            "evaluate_phase_acceptance": self.evaluate_phase_acceptance,
            "resolve_model_path": self.resolve_model_path,
            "run_to_facts": self.run_to_facts,
        }
        if key not in table:
            raise KeyError(key)
        return table[key]

    # -- load / preflight --------------------------------------------------
    def load_runtime_state(self, model_path: str, contract: Any, source_authority: Dict[str, Any]) -> SyntheticRuntimeState:
        self.load_calls += 1
        state = SyntheticRuntimeState(model_path, contract, source_authority, model_load_count=self.load_calls)
        state.memory_plan = {
            "status": "PASS",
            "planner": "plan_dual_t4_spread",
            "components": {"transformer_blocks": [str(i) for i in range(24)]},
            "split_block": 12,
            "single_device": {"status": "FAIL"},
            "device_map_json": {},
        }
        state.placement_validation = {"status": "PASS"}
        self.loaded_states.append(state)
        return state

    def preflight_gpu(self, require_t4x2: bool = True) -> Dict[str, Any]:
        return {
            "torch": "synthetic-cpu",
            "cuda_available": False,
            "device_count": 2,
            "inventory": [
                {"name": "Tesla T4", "compute_capability": "7.5"},
                {"name": "Tesla T4", "compute_capability": "7.5"},
            ],
            "t4x2_ok": True,
        }

    def resolve_model_path(self, contract: Any) -> str:
        return "/synthetic/model/path"

    # -- profile / topology application ------------------------------------
    def _plan(self, num_blocks: int, split_block: Optional[int]) -> Dict[int, str]:
        if self.scenario == "missing_gpu1":
            return {i: GPU0 for i in range(num_blocks)}
        split = split_block if split_block is not None else num_blocks // 2
        return {i: (GPU0 if i < split else GPU1) for i in range(num_blocks)}

    def apply_dual_t4_mixed(self, state: SyntheticRuntimeState, num_blocks: int, split_block: Optional[int] = None) -> Dict[str, Any]:
        block_devices = self._plan(num_blocks, split_block)
        final_block = max(block_devices)
        final_device = block_devices[final_block]
        top = {
            "topology": "dual_t4",
            "text_encoder": GPU0,
            "vae": GPU1,
            "transformer": {
                "pre": {
                    "img_in": GPU0,
                    "txt_norm": GPU0,
                    "time_text_embed": GPU0,
                    "txt_in": GPU0,
                },
                "blocks": {str(k): v for k, v in block_devices.items()},
                "post": {
                    "norm_out": final_device,
                    "proj_out": final_device,
                },
            },
            "pos_embed": "RUNTIME_TENSOR_DEVICE",
            "block_devices": {str(k): v for k, v in block_devices.items()},
            "num_transformer_blocks": num_blocks,
            "split_policy": "explicit" if split_block is not None else "half",
            "split_block": split_block if split_block is not None else num_blocks // 2,
        }
        state.device_map = {
            "transformer": {str(k): v for k, v in block_devices.items()},
            "vae": GPU1,
        }
        state.current_profile = "mixed"
        state.current_topology = "dual_t4"
        return {"ok": True, "device_plan": top}

    def apply_dual_t4_all_bf16(self, state: SyntheticRuntimeState, num_blocks: int, split_block: Optional[int] = None) -> Dict[str, Any]:
        out = self.apply_dual_t4_mixed(state, num_blocks, split_block=split_block)
        state.current_profile = "all_bf16"
        return {"ok": True, "dtype_audit": self._dtype_report(state), **out}

    def _dtype_report(self, state: SyntheticRuntimeState) -> Dict[str, Any]:
        profile = state.current_profile
        all_ok = profile == "all_bf16"
        status = "PASS" if all_ok else "FAIL"
        te_status = "FAIL"
        tr_status = "FAIL"
        vae_status = "FAIL"
        if self.scenario == "dtype_fail" and all_ok:
            tr_status = "FAIL"
        elif all_ok:
            te_status = "PASS"
            tr_status = "PASS"
            vae_status = "PASS"
        return {
            "overall_status": status,
            "text_encoder": {"status": te_status, "parameter_dtype_histogram": {}, "buffer_dtype_histogram": {}},
            "transformer": {"status": tr_status, "parameter_dtype_histogram": {}, "buffer_dtype_histogram": {}, "unexpected_dtype_list": ["torch.float32"] if self.scenario == "dtype_fail" and all_ok else []},
            "vae": {"status": vae_status, "parameter_dtype_histogram": {}, "buffer_dtype_histogram": {}},
        }

    # -- evidence / telemetry factories ------------------------------------
    def create_evidence_run(self, run_dir: str, source_authority: Optional[Dict[str, Any]] = None) -> EvidenceRun:
        return EvidenceRun(run_dir, source_authority=source_authority)

    def create_telemetry(self, path: str, run_id: str, phase: str, inference_id: str) -> TelemetryRecorder:
        return TelemetryRecorder(path, run_id=run_id, phase=phase, inference_id=inference_id)

    def collect_telemetry(self, path: str) -> List[Dict[str, Any]]:
        from .telemetry import parse_telemetry

        return parse_telemetry(path)

    # -- adapter ownership -------------------------------------------------
    def attach_dual_adapter(
        self,
        state: SyntheticRuntimeState,
        device_plan: Dict[str, Any],
        telemetry: Any,
        run_id: str,
        phase: str,
        inference_id: str,
    ) -> DualDeviceForwardAdapter:
        self.attach_attempts += 1
        if self.scenario == "adapter_missing":
            raise RuntimeError("DUAL_FORWARD_ADAPTER_MISSING: synthetic adapter not attached")
        if state.dual_forward_adapter is not None and getattr(state.dual_forward_adapter, "attached", False):
            raise RuntimeError("ADAPTER_DOUBLE_ATTACH: adapter already attached on this transformer")
        if self.scenario == "post_device_mismatch":
            raise RuntimeError("TRANSFORMER_POST_DEVICE_MISMATCH: synthetic post module on wrong device")
        block_devices = extract_block_devices(device_plan)
        adapter = DualDeviceForwardAdapter(
            state.transformer,
            block_devices,
            transfer_helper=CrossDeviceTransferHelper(
                mover=_cpu_mover,
                run_id=run_id,
                phase=phase,
                inference_id=inference_id,
            ),
            telemetry=_CaptureTelemetry(run_id, phase, inference_id),
            run_id=run_id,
            phase=phase,
            inference_id=inference_id,
        )
        adapter.attach()
        state.dual_forward_adapter = adapter
        state.dual_forward_adapter_attached = True
        self.adapters.append(adapter)
        self.block_devices_per_phase[phase] = block_devices.copy()
        self.attach_count += 1
        return adapter

    def detach_dual_adapter(self, state: Any, adapter: Any = None, reason: str = "phase_end") -> bool:
        adapter = adapter or getattr(state, "dual_forward_adapter", None)
        if adapter is not None and getattr(adapter, "attached", False):
            adapter.detach()
        if state is not None:
            state.dual_forward_adapter = None
            state.dual_forward_adapter_attached = False
        self.detach_count += 1
        self.detach_reasons.append(reason)
        return True

    # -- run one T2I -------------------------------------------------------
    def run_t2i(
        self,
        state: SyntheticRuntimeState,
        contract: RunContract,
        run: EvidenceRun,
        telemetry: TelemetryRecorder,
        run_id: str,
        inference_id: Optional[str] = None,
        phase: str = "t2i",
    ) -> Dict[str, Any]:
        self.run_t2i_call_count += 1
        self.phase_inference_calls[phase] = self.phase_inference_calls.get(phase, 0) + 1
        self.state_ids.append(id(state))
        self.inference_ids[phase] = inference_id or telemetry.inference_id or "n/a"

        if self.scenario == "run_t2i_exception":
            telemetry.phase("t2i", "START")
            raise RuntimeError(f"run_t2i synthetic failure in {phase}")

        if self.scenario == "no_inference":
            self._no_inference_telemetry(telemetry, run_id, phase, inference_id)
            return {
                "output_png": None,
                "validation": {"exists": False, "error": "no_inference"},
                "dtype_after": self._dtype_report(state),
                "has_nan": True,
                "has_inf": False,
                "inference_id": inference_id or telemetry.inference_id,
                "run_id": run_id,
                "phase": phase,
                "run_exit_zero": True,
            }

        # Run the partitioned/reference forward once per contract denoising
        # step (multistep parity with the real backend: each step emits a full
        # transformer block sequence plus one return-transfer boundary).
        steps = int(getattr(contract, "steps", 0) or 0)
        if steps <= 0:
            steps = 1
        capture = self._capture_forward(state, phase, steps=steps)

        # Apply scenario transforms on captured evidence.
        records = self._transform_records(capture, run_id, phase, inference_id)

        # Write the real telemetry file.
        for rec in records:
            telemetry.event(rec["event"], **{k: v for k, v in rec.items() if k != "event"})
        telemetry.close()

        # Output image.
        invalid = self.scenario == "invalid_output"
        out_png = str(run.run_dir / "output.png")
        if invalid:
            image_validation.make_fixture_png(out_png, width=32, height=32)
        else:
            image_validation.make_fixture_png(out_png, width=512, height=512)

        validation = image_validation.validate_image(out_png)
        run.write_image_validation(validation)
        run.write_device_map(getattr(state, "device_map", None) or {})
        dtype_after = self._dtype_report(state)
        run.write_dtype("after", dtype_after)

        return {
            "output_png": out_png,
            "validation": validation,
            "dtype_after": dtype_after,
            "has_nan": bool(validation.get("has_nan")),
            "has_inf": bool(validation.get("has_inf")),
            "inference_id": inference_id or telemetry.inference_id,
            "run_id": run_id,
            "phase": phase,
            "run_exit_zero": True,
            "exit_code": 0,
        }

    def _no_inference_telemetry(self, telemetry: TelemetryRecorder, run_id: str, phase: str, inference_id: Optional[str]) -> None:
        telemetry.event(
            "phase",
            phase="t2i",
            status="START",
            run_id=run_id,
            inference_id=inference_id,
        )
        telemetry.close()

    def _capture_forward(self, state: SyntheticRuntimeState, phase: str, steps: int = 1) -> _CaptureTelemetry:
        """Run one real forward per contract step through the (possibly)
        attached adapter so routing evidence is genuinely produced, not
        hard-coded. Each step emits the full transformer block sequence plus one
        ``transformer_output_return_transfer`` boundary (and its cross-device
        transfer at the split boundary), mirroring the real backend's per-step
        ``pipeline.generate`` invocation."""
        capture: _CaptureTelemetry = _CaptureTelemetry("capture", phase, "capture")

        # Caller device is cuda:0 (image carrier); final transformer device is
        # the last block's device per the live plan. On the CPU synthetic the
        # adapter's return bridge never fires (payload devices resolve to
        # ``cpu``), so the per-step return boundary is emitted here to preserve
        # parity with the real GPU backend.
        caller_device = "cuda:0"
        transformer_map = getattr(state, "device_map", {}).get("transformer") or {}
        block_devices = {int(k): str(v) for k, v in transformer_map.items()}
        if block_devices:
            last_block_device = block_devices[max(block_devices)]

        adapter = getattr(state, "dual_forward_adapter", None)
        if adapter is not None and getattr(adapter, "attached", False):
            prev = adapter.telemetry
            prev_transfer = adapter.transfer_helper.telemetry
            adapter.telemetry = capture
            adapter.transfer_helper.telemetry = capture
            try:
                for _ in range(steps):
                    self._forward(state, capture)
                    if last_block_device is not None and last_block_device != caller_device:
                        capture.event(
                            "transformer_output_return_transfer",
                            from_=last_block_device,
                            to=caller_device,
                        )
            finally:
                adapter.telemetry = prev
                adapter.transfer_helper.telemetry = prev
        else:
            for _ in range(steps):
                self._forward(state, capture)
                if last_block_device is not None and last_block_device != caller_device:
                    capture.event(
                        "transformer_output_return_transfer",
                        from_=last_block_device,
                        to=caller_device,
                    )
        return capture

    def _forward(self, state: SyntheticRuntimeState, capture: _CaptureTelemetry) -> Any:
        import torch

        synth = state.transformer
        img = torch.randn(1, 16, 8)
        txt = torch.randn(1, 8, 8)
        timesteps = torch.full((1,), 0.42)
        cu_i = torch.tensor([0, 16], dtype=torch.int32)
        cu_t = torch.tensor([0, 8], dtype=torch.int32)
        shapes = [[(1, 4, 4)]]
        return synth.forward(img, txt, timesteps, shapes, cu_i, cu_t)

    def _transform_records(
        self,
        capture: _CaptureTelemetry,
        run_id: str,
        phase: str,
        inference_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        records.append({"event": "phase", "phase": "t2i", "status": "START", "run_id": run_id, "inference_id": inference_id})
        for rec in capture.records:
            if rec.get("event") == "metric":
                continue
            normalized = dict(rec)
            normalized["run_id"] = run_id
            normalized["phase"] = phase
            normalized["inference_id"] = inference_id
            records.append(normalized)
        records.append({"event": "phase", "phase": "t2i", "status": "PASS", "run_id": run_id, "inference_id": inference_id})
        if self.scenario == "second_model_load":
            records.append({"event": "model_load", "load_index": 2, "run_id": run_id, "phase": phase, "inference_id": inference_id})

        if self.scenario == "wrong_inference_id":
            wrong = "t2i-WRONG-INFERENCE"
            for rec in records:
                rec["inference_id"] = wrong
        if self.scenario == "missing_transfer":
            records = [r for r in records if r.get("event") != "cross_device_transfer"]
        elif self.scenario == "duplicate_block":
            first_block = next((i for i, r in enumerate(records) if r.get("event") == "block_forward"), None)
            if first_block is not None:
                dup = dict(records[first_block])
                records.insert(first_block + 1, dup)
        elif self.scenario == "skipped_block":
            dropped = False
            kept = []
            for r in records:
                if not dropped and r.get("event") == "block_forward" and int(r.get("block", -1)) == 5:
                    dropped = True
                    continue
                kept.append(r)
            records = kept
        return records

    # -- facts + acceptance ------------------------------------------------
    def derive_phase_facts(self, phase: str, result: Dict[str, Any], telemetry_records: list, model_load_count: int, runtime_state: Any, contract: Any, adapter_status: Optional[Dict[str, Any]] = None, replay_evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from .phase_acceptance import derive_phase_facts as _derive

        return _derive(
            phase,
            result,
            telemetry_records,
            model_load_count,
            runtime_state,
            contract,
            adapter_status=adapter_status,
            replay_evidence=replay_evidence,
        )

    def evaluate_phase_acceptance(self, phase: str, facts: Dict[str, Any]) -> Dict[str, Any]:
        from .phase_acceptance import evaluate_phase_acceptance as _evaluate

        return _evaluate(phase, facts)

    def run_to_facts(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"PYTORCH_RUNTIME": True}