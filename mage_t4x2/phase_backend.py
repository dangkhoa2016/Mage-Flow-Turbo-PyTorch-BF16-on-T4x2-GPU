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
