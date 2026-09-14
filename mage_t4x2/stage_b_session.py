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
ROUTING_PASS = "ROUTING_PASS"
G3_PASS = "G3_PASS"
G4_PASS = "G4_PASS"
G5_PASS = "G5_PASS"
FINALIZED = "FINALIZED"

FAILED_G0 = "FAILED_G0"
FAILED_L0 = "FAILED_L0"
FAILED_G1 = "FAILED_G1"
FAILED_ROUTING = "FAILED_ROUTING"
FAILED_G3 = "FAILED_G3"
FAILED_G4 = "FAILED_G4"
FAILED_G5 = "FAILED_G5"
FAILED_G6 = "FAILED_G6"

_STABLE_STATES = (
    CREATED, G0_PASS, L0_SPREAD_LOAD_PASS, G1_PASS, ROUTING_PASS,
    G3_PASS, G4_PASS, G5_PASS, FINALIZED,
    FAILED_G0, FAILED_L0, FAILED_G1, FAILED_ROUTING, FAILED_G3, FAILED_G4, FAILED_G5, FAILED_G6,
)

# Allowed transitions: (from, to)
_TRANSITIONS: set = {
    (CREATED, G0_PASS),
    (CREATED, FAILED_G0),
    (G0_PASS, L0_SPREAD_LOAD_PASS),
    (G0_PASS, FAILED_L0),
    (L0_SPREAD_LOAD_PASS, G1_PASS),
    (L0_SPREAD_LOAD_PASS, FAILED_G1),
    (G1_PASS, ROUTING_PASS),
    (G1_PASS, FAILED_ROUTING),
    (ROUTING_PASS, G3_PASS),
    (ROUTING_PASS, FAILED_G3),
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
