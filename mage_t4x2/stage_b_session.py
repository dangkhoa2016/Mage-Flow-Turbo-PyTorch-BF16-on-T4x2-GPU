"""Same-process Stage B session orchestration.

Guarantees that G0..G6 run in ONE Python process with ONE live ``RuntimeState``
and ONE model load, the load-once/reconfigure-live rule. Subprocess orchestration
for post-load phases is forbidden.

Corrective A3 semantics: every authority phase (G1..G5) executes a real backend
inference call through ``run_t2i``, derives observed facts from evidence, runs
phase-specific acceptance, and only then transitions to ``G*_PASS``. A phase can
never PASS from configuration intent alone. The dual-forward adapter has exactly
one owner (this session, via the backend) and is detached + re-attached per phase
with telemetry bound to the current inference id.

Import-safety: importing this module must NOT initialize CUDA and must NOT import
``scripts.gpu_session`` at module scope. GPU work is delegated to an injectable
``phase_backend`` (defaults to a lazy import of ``scripts.gpu_session``).
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import constants as C

# ---------------------------------------------------------------------------
# Session states
# ---------------------------------------------------------------------------

CREATED = "CREATED"
G0_PASS = "G0_PASS"
L0_SPREAD_LOAD_PASS = C.L0_ACCEPTED_STATE_PREFIX
G1_PASS = "G1_PASS"
G2_PASS = "G2_PASS"
G3_PASS = "G3_PASS"
G4_PASS = "G4_PASS"
G5_PASS = "G5_PASS"
FINALIZED = "FINALIZED"

FAILED_G0 = "FAILED_G0"
FAILED_L0 = "FAILED_L0"
FAILED_G1 = "FAILED_G1"
FAILED_G2 = "FAILED_G2"
FAILED_G3 = "FAILED_G3"
FAILED_G4 = "FAILED_G4"
FAILED_G5 = "FAILED_G5"
FAILED_G6 = "FAILED_G6"

_STABLE_STATES = (
    CREATED, G0_PASS, L0_SPREAD_LOAD_PASS, G1_PASS, G2_PASS,
    G3_PASS, G4_PASS, G5_PASS, FINALIZED,
    FAILED_G0, FAILED_L0, FAILED_G1, FAILED_G2, FAILED_G3, FAILED_G4, FAILED_G5, FAILED_G6,
)

# Allowed transitions: (from, to)
_TRANSITIONS: set = {
    (CREATED, G0_PASS),
    (CREATED, FAILED_G0),
    (G0_PASS, L0_SPREAD_LOAD_PASS),
    (G0_PASS, FAILED_L0),
    (L0_SPREAD_LOAD_PASS, G1_PASS),
    (L0_SPREAD_LOAD_PASS, FAILED_G1),
    (G1_PASS, G2_PASS),
    (G1_PASS, FAILED_G2),
    (G2_PASS, G3_PASS),
    (G2_PASS, FAILED_G3),
    (G3_PASS, G4_PASS),
    (G3_PASS, FAILED_G4),
    (G4_PASS, G5_PASS),
    (G4_PASS, FAILED_G5),
    (G5_PASS, FINALIZED),
    (G5_PASS, FAILED_G6),
}


class AuthorityStateError(RuntimeError):
    """Raised on illegal state transitions or guard violations."""


@dataclasses.dataclass
class SessionState:
    """Explicit machine-readable session state."""

    name: str = CREATED


class _StateMachine:
    def __init__(self) -> None:
        self._name = CREATED

    @property
    def name(self) -> str:
        return self._name

    def transition(self, to: str) -> None:
        if (self._name, to) not in _TRANSITIONS:
            raise AuthorityStateError(
                f"illegal session transition {self._name!r} -> {to!r}"
            )
        self._name = to

    def fail(self, to: str) -> None:
        if to not in _STABLE_STATES or not to.startswith("FAILED_"):
            raise AuthorityStateError(f"invalid failure state {to!r}")
        if (self._name, to) not in _TRANSITIONS:
            raise AuthorityStateError(
                f"illegal failure transition {self._name!r} -> {to!r}"
            )
        self._name = to


# ---------------------------------------------------------------------------
# Phase result record
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PhaseResult:
    """Result of one GPU authority phase (G0..G6) with real acceptance detail."""

    phase: str
    run_id: str
    inference_id: str
    exit_code: int
    status: str  # "PASS" | "FAIL" | "NOT_RUN"
    model_load_count: Optional[int] = None
    runtime_state_id: Optional[str] = None
    inference_call_count: int = 0
    output_path: Optional[str] = None
    acceptance: Optional[Dict[str, Any]] = None
    observed_facts: Optional[Dict[str, Any]] = None
    evidence_paths: Optional[Dict[str, Any]] = None
    failure_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


class AuthorityViolation(RuntimeError):
    """Raised when the authority invariant is violated (e.g. model reload)."""


# ---------------------------------------------------------------------------
# Phase context
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PhaseContext:
    """One phase-local authority context (Corrective A3 section 5)."""

    phase: str
    run_id: str
    inference_id: str
    evidence_run: Any
    telemetry: Any
    run_dir: str

    @property
    def telemetry_path(self) -> str:
        return str(Path(self.run_dir) / "telemetry.jsonl")


# ---------------------------------------------------------------------------
# Config intent / evidence separation
# ---------------------------------------------------------------------------


def requested_intent() -> Dict[str, Any]:
    return {
        "topology": None,
        "precision": None,
        "dual_gpu": False,
        "all_bf16": False,
    }


# ---------------------------------------------------------------------------
# Stage B session
# ---------------------------------------------------------------------------

_DEFAULT_EVIDENCE_ROOT = Path(__file__).resolve().parent.parent / "evidence" / "gpu"


class StageBSession:
    """One continuous session for G0..G6 in a single notebook kernel.

    All phases after model load reuse ``self.state`` (same object, never
    reloaded). The backend is injectable for CPU-only synthetic qualification.
    The dual-forward adapter is owned by this session (attached per phase with
    telemetry-bound evidence; detached at phase end).
    """

    def __init__(
        self,
        contract: Any,
        source_authority: Dict[str, Any],
        phase_backend: Any = None,
        loader: Optional[Callable[[], Any]] = None,
        evidence_root: Optional[str] = None,
    ) -> None:
        self.contract = contract
        self.source_authority = source_authority
        self.state: Any = None
        self.model_load_count: int = 0
        self.model_loaded: bool = False
        self.finalized: bool = False
        self.session_id: str = uuid.uuid4().hex[:12]
        self.evidence_root: str = evidence_root or str(
            _DEFAULT_EVIDENCE_ROOT / f"run-{self.session_id}"
        )
        self._sm = _StateMachine()
        self._beam = phase_backend
        self._loader = loader
        self.inference_id: Optional[str] = None
        self.results: Dict[str, PhaseResult] = {}
        self.device_map_log: Dict[str, Dict[str, Any]] = {}
        self.precision_progression: list = []
        self._inference_calls: Dict[str, int] = {}
        self.requested = requested_intent()

        # Adapter owner model (Corrective A3 section 6).
        self.active_dual_adapter: Any = None
        self._current_block_devices: Dict[int, str] = {}
        self._last_dual_plan: Optional[Dict[str, Any]] = None
        self._g4_replay_record: Optional[Dict[str, Any]] = None

    # -- state helpers -----------------------------------------------------
    @property
    def state_name(self) -> str:
        return self._sm.name

    @property
    def dual_forward_adapter_attached(self) -> bool:
        adapter = self.active_dual_adapter
        return bool(adapter is not None and getattr(adapter, "attached", False))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state_name,
            "model_load_count": self.model_load_count,
            "model_loaded": self.model_loaded,
            "finalized": self.finalized,
            "inference_id": self.inference_id,
            "dual_forward_adapter_attached": self.dual_forward_adapter_attached,
            "precision_progression": list(self.precision_progression),
            "inference_calls": dict(self._inference_calls),
        }

    # -- authority guards --------------------------------------------------
    @property
    def _l0_split_block(self) -> Optional[int]:
        mp = getattr(self.state, "memory_plan", None) or {}
        return mp.get("split_block")

    def _memory_plan_params(self) -> tuple:
        mp = getattr(self.state, "memory_plan", None) or {}
        blocks = (mp.get("components") or {}).get("transformer_blocks") or []
        return (len(blocks) or None, mp.get("split_block"))

    def load_once(self, model_path: str) -> Any:
        """Load the model exactly once, via the L0 dual-T4 spread load.

        L0 is a recorded authority phase: facts are derived from the live
        ``memory_plan`` / ``placement_validation`` evidence and must PASS
        (SPREAD_LOAD_PLAN_PASS, SPREAD_LOAD_OBSERVED_PLACEMENT_PASS,
        NO_FULL_MODEL_CUDA0_MATERIALIZATION, MODEL_LOAD_COUNT_EQ_1).
        """
        if self.finalized:
            raise AuthorityStateError("load_once after finalization is forbidden")
        if self.state is not None:
            return self.state
        if self._sm.name not in (CREATED, G0_PASS):
            raise AuthorityStateError(
                f"load_once called from state {self._sm.name!r}; must be CREATED or G0_PASS"
            )

        if self._loader is not None:
            state = self._loader()
        else:
            backend = self._backend()
            state = self._load_with_backend(backend, model_path)

        self.state = state
        self.model_load_count += 1
        self.model_loaded = True
        if self.model_load_count != 1:
            raise AuthorityViolation(
                f"model_load_count={self.model_load_count}; authority allows exactly 1"
            )
        if self._sm.name == CREATED:
            self._sm.transition(G0_PASS)

        l0 = self._record_l0_result(state)
        if l0.status != "PASS":
            self._sm.fail(FAILED_L0)
            raise AuthorityStateError(f"L0 spread-load acceptance failed: {l0.failure_code}")
        self._sm.transition(L0_SPREAD_LOAD_PASS)
        return self.state

    def _record_l0_result(self, state: Any) -> PhaseResult:
        """Run L0 acceptance on the loaded state and write L0 evidence files."""
        from .evidence import write_json_doc

        backend = self._backend()
        observed = self._call(
            backend, "derive_phase_facts",
            phase="L0",
            result={},
            telemetry_records=[],
            model_load_count=self.model_load_count,
            runtime_state=state,
            contract=self.contract,
        )
        acceptance = self._call(backend, "evaluate_phase_acceptance", "L0", observed)
        l0_dir = Path(self.evidence_root) / "L0"
        write_json_doc(str(l0_dir / "observed-facts.json"), observed)
        write_json_doc(str(l0_dir / "acceptance.json"), acceptance)
        write_json_doc(
            str(l0_dir / "memory-plan.json"),
            getattr(state, "memory_plan", None) or {},
        )
        write_json_doc(
            str(l0_dir / "placement-validation.json"),
            getattr(state, "placement_validation", None) or {},
        )
        result = self._finalize_result(
            None, "L0", 0 if acceptance.get("status") == "PASS" else 1,
            "PASS" if acceptance.get("status") == "PASS" else "FAIL",
            observed=observed,
            acceptance=acceptance,
            failure_code=None if acceptance.get("status") == "PASS" else "PHASE_ACCEPTANCE_FAILED",
            details={
                "split_block": getattr(state, "memory_plan", {}).get("split_block"),
                "single_device_rejected": (getattr(state, "memory_plan", {}).get("single_device") or {}).get("status") == "FAIL",
            },
        )
        result.evidence_paths = {
            "phase_result": str(l0_dir / "phase-result.json"),
            "observed_facts": str(l0_dir / "observed-facts.json"),
            "acceptance": str(l0_dir / "acceptance.json"),
            "memory_plan": str(l0_dir / "memory-plan.json"),
            "placement_validation": str(l0_dir / "placement-validation.json"),
        }
        result.save(str(l0_dir / "phase-result.json"))
        return result

    def _load_with_backend(self, backend: Any, model_path: str) -> Any:
        return self._call(backend, "load_runtime_state", model_path, self.contract, self.source_authority)

    # -- backend dispatch --------------------------------------------------
    def _backend(self) -> Any:
        if self._beam is None:
            from scripts import gpu_session  # type: ignore

            self._beam = {
                "load_runtime_state": gpu_session.load_runtime_state,
                "preflight_gpu": gpu_session.preflight_gpu,
                "apply_dual_t4_mixed": gpu_session.apply_dual_t4_mixed,
                "apply_dual_t4_all_bf16": gpu_session.apply_dual_t4_all_bf16,
                "create_evidence_run": gpu_session.create_evidence_run,
                "create_telemetry": gpu_session.create_telemetry,
                "attach_dual_adapter": gpu_session.attach_dual_adapter,
                "detach_dual_adapter": gpu_session.detach_dual_adapter,
                "run_t2i": gpu_session.run_t2i,
                "collect_telemetry": gpu_session.collect_telemetry,
                "derive_phase_facts": gpu_session.derive_phase_facts,
                "evaluate_phase_acceptance": gpu_session.evaluate_phase_acceptance,
                "resolve_model_path": gpu_session.resolve_model_path,
                "run_to_facts": gpu_session.run_to_facts,
            }
        return self._beam

    @staticmethod
    def _call(backend: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        if isinstance(backend, dict) or hasattr(backend, "__getitem__"):
            fn = backend[name]
        else:
            fn = getattr(backend, name)
        return fn(*args, **kwargs)

    # -- phase context -----------------------------------------------------
    def _new_phase_context(self, phase: str) -> PhaseContext:
        backend = self._backend()
        inference_id = self._new_inference_id()
        run_dir = Path(self.evidence_root) / phase
        evidence_run = self._call(backend, "create_evidence_run", str(run_dir), self.source_authority)
        telemetry = self._call(
            backend, "create_telemetry",
            str(run_dir / "telemetry.jsonl"),
            run_id=self.session_id,
            phase=phase,
            inference_id=inference_id,
        )
        return PhaseContext(
            phase=phase,
            run_id=self.session_id,
            inference_id=inference_id,
            evidence_run=evidence_run,
            telemetry=telemetry,
            run_dir=str(run_dir),
        )

    # -- adapter ownership / structural truth ------------------------------
    def _ensure_no_active_adapter(self, phase: str) -> None:
        """Never stack monkeypatches: safely detach any stale live adapter."""
        if self.active_dual_adapter is not None:
            adapter = self.active_dual_adapter
            if getattr(adapter, "attached", False):
                self._detach_adapter(reason=f"{phase}_pre_attach_cleanup")
            else:
                self.active_dual_adapter = None

    def _attach_adapter(self, ctx: PhaseContext, plan: Optional[Dict[str, Any]]) -> Any:
        from mage_t4x2.device_plan import extract_block_devices

        backend = self._backend()
        adapter = self._call(
            backend, "attach_dual_adapter",
            self.state,
            plan,
            ctx.telemetry,
            ctx.run_id,
            ctx.phase,
            ctx.inference_id,
        )
        self.active_dual_adapter = adapter
        self._current_block_devices = extract_block_devices(plan)
        return adapter

    def _detach_adapter(self, reason: str = "phase_end") -> None:
        backend = self._backend()
        adapter = self.active_dual_adapter
        try:
            self._call(backend, "detach_dual_adapter", self.state, adapter, reason=reason)
        except Exception:
            # Detach is best-effort; the reference is cleared regardless.
            if adapter is not None and getattr(adapter, "attached", False):
                adapter.detach()
        self.active_dual_adapter = None
        self._current_block_devices = {}

    def _adapter_status(self, ctx: PhaseContext) -> Dict[str, Any]:
        """Adapter truth is observed, not declared (Corrective A3 section 7)."""
        adapter = self.active_dual_adapter
        attrs = {
            "active_adapter_exists": adapter is not None,
            "attached": bool(adapter is not None and getattr(adapter, "attached", False)),
            "transformer_matches": bool(
                adapter is not None and adapter.transformer is self.state.transformer
            ),
            "patched_forward": bool(
                adapter is not None
                and getattr(adapter, "attached", False)
                and getattr(adapter.transformer, "forward", None)
                is not getattr(adapter, "_original_forward", object())
            ),
            "block_map_matches": bool(
                adapter is not None
                and adapter.block_devices == self._current_block_devices
            ),
            "telemetry_run_id_matches": bool(
                adapter is not None
                and getattr(adapter.telemetry, "run_id", None) == ctx.run_id
            ),
            "telemetry_phase_matches": bool(
                adapter is not None
                and getattr(adapter.telemetry, "_phase", None) == ctx.phase
            ),
            "telemetry_inference_id_matches": bool(
                adapter is not None
                and getattr(adapter.telemetry, "inference_id", None) == ctx.inference_id
            ),
        }
        attrs["check_summary"] = all(attrs.values())
        return attrs

    # -- phase runner (canonical lifecycle) --------------------------------
    def _execute_phase(
        self,
        phase: str,
        pass_state: str,
        failed_state: str,
        config_fn: Callable[[PhaseContext, Any], Dict[str, Any]],
        require_dual_adapter: bool = False,
        replay: bool = False,
    ) -> PhaseResult:
        backend = self._backend()
        ctx: Optional[PhaseContext] = None
        detach_done = False
        try:
            self._assert_index()
            ctx = self._new_phase_context(phase)
            applied = config_fn(ctx, backend) or {}

            adapter_status: Optional[Dict[str, Any]] = None
            if require_dual_adapter:
                plan = applied.get("device_plan")
                if plan is None and self._last_dual_plan is not None:
                    plan = self._last_dual_plan
                if plan is None:
                    plan = {"transformer": {"blocks": dict(self._current_block_devices)}}
                self._ensure_no_active_adapter(phase)
                self._attach_adapter(ctx, plan)
                adapter_status = self._adapter_status(ctx)
                if not adapter_status.get("attached"):
                    self._sm.fail(failed_state)
                    return self._fail_result(
                        ctx, phase, "DUAL_FORWARD_ADAPTER_MISSING",
                        details={"adapter_status": adapter_status},
                    )

            self._inference_calls[phase] = self._inference_calls.get(phase, 0) + 1
            result = self._call(
                backend, "run_t2i",
                self.state, self.contract, ctx.evidence_run, ctx.telemetry,
                self.session_id,
                inference_id=ctx.inference_id,
                phase=phase,
            )
            if not self._validate_inference_result(result):
                self._sm.fail(failed_state)
                return self._fail_result(
                    ctx, phase, "INFERENCE_NOT_EXECUTED",
                    details={"explanation": "backend returned no executable inference evidence", "result": result},
                )

            telemetry_records = self._call(backend, "collect_telemetry", ctx.telemetry_path)
            observed = self._call(
                backend, "derive_phase_facts",
                phase=phase,
                result=result,
                telemetry_records=telemetry_records,
                model_load_count=self.model_load_count,
                runtime_state=self.state,
                contract=self.contract,
                adapter_status=adapter_status,
                replay_evidence=self._replay_evidence(phase, ctx, result) if replay else None,
            )
            acceptance = self._call(backend, "evaluate_phase_acceptance", phase, observed)
            self._write_phase_evidence(ctx, result, observed, acceptance)
            self._record_replay_state(phase, ctx, result)

            if require_dual_adapter:
                self._detach_adapter(reason=f"{phase}_exit")
                detach_done = True

            if acceptance.get("status") != "PASS":
                self._sm.fail(failed_state)
                return self._fail_result(
                    ctx, phase, "PHASE_ACCEPTANCE_FAILED",
                    observed=observed,
                    acceptance=acceptance,
                    details={"acceptance": acceptance},
                )

            self._sm.transition(pass_state)
            return self._pass_result(
                ctx, phase, observed=observed, acceptance=acceptance,
                output_path=result.get("output_png"),
            )
        except Exception as exc:
            if ctx is not None and require_dual_adapter and not detach_done:
                try:
                    self._detach_adapter(reason=f"{phase}_exception_cleanup")
                except Exception:
                    pass
            code = self._classify_failure(exc)
            try:
                self._sm.fail(failed_state)
            except AuthorityStateError:
                pass
            return self._fail_result(ctx, phase, code, details={"error": str(exc)})

    def _validate_inference_result(self, result: Optional[Dict[str, Any]]) -> bool:
        if result is None:
            return False
        val = result.get("validation") or {}
        if not val.get("exists"):
            return False
        if not result.get("output_png"):
            return False
        return True

