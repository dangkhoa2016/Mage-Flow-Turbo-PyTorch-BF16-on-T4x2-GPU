#!/usr/bin/env python3
"""G2 routing authority R2G multistep block-integrity corrective fresh-kernel authoritative driver.

R2G targets (static runbook 2026-09-18, successor to R2D/R2E/R2F):

* R2G-1  NEW STATIC R2G RUNTIME REBASELINE - ``authority/r2g-runtime-baseline.json``
         (QUALIFIED_STATIC) covering the successor runtime authority files
         and engaged as the FIRST authority gate
         (R2G_RUNTIME_BASELINE_INTEGRITY) before any other cold-start action.
         The R2D/R2E baselines remain historical and immutable; this successor
         baseline describes the corrected successor source.
* R2G-2  FAIL-CLOSED BOOTSTRAP (DEFECT A CORRECTED, PRESERVED) - the bootstrap input
         verifier now reads the ``artifacts`` schema, verifies the lock SHA-256
         against the manifest requirements SHA-256, and byte-verifies every
         wheelhouse artifact (filename dist-version-python-tag, size, SHA-256).
         A corrupt wheelhouse can never pass the authority gate again.
* R2G-3  GENUINELY FRESH BOOTSTRAP SITE (DEFECT B CORRECTED, PRESERVED) - a pre-existing
         bootstrap target can never be reused; ``create_fresh_bootstrap_site``
         raises STALE_BOOTSTRAP_SITE whenever the target already exists, so
         the real freshness marker is always freshly created.
* R2G-4  R2G AUTHORITY DRIVER (MULTISTEP BLOCK-INTEGRITY CORRECTIVE) -
         corrections (dual-T4 split-boundary transfers, return-to-caller,
         latent/caller anchor, placement validation, no-CPU-fallback, exact
         block coverage; per-invocation multi-step segmentation now refutes the
         genuine G1 false negative) under the R2G machine gate names.

Single-run flow (semantic order, never repeats; runbook section 57):

    fresh kernel import -> R2G runtime baseline integrity ->
    freeze SDPA env -> verify bootstrap lock+wheelhouse -> create fresh
    local bootstrap site -> offline provision exact lock -> activate local
    site -> verify local version + origin -> bootstrap_upstream_mage ->
    P0 -> G0 (hardware+session) -> L0 (load exactly once) ->
    derive split-boundary endpoints -> SDPA -> one G1 baseline ->
    one G2 routing authority -> live placement/acceptance adjudication ->
    machine summary -> human summary -> STOP.

G3..G6 are NEVER executed and the model is NEVER loaded twice.  Every authority
gate is fail-closed: the first failed gate ends the run with non-zero exit.

R2G CORRECTIVE ROOT CAUSE: phase acceptance previously derived global block-order
validity across all multistep invocation cycles, so a genuine G1 multistep
block-integrity false negative was refused (BLOCK_ORDER_VALID=false,
NO_DUPLICATED_BLOCKS=false). The reducer now segments per invocation before
adjudication; the G1/G2 acceptance gates below verify per-invocation block
sequences exactly (expected_invocations x [0..num_blocks-1]).

Import-safe on CPU for static validation (no CUDA touched at import; the SDPA
freeze is applied inside run(), never at module scope).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# P0-a: fresh-kernel environment preflight (self-contained repo bootstrap)
# ---------------------------------------------------------------------------

REPO = Path("/kaggle/working/Mage-Flow-Turbo-PyTorch-BF16-T4x2")

if not REPO.is_dir():
    raise RuntimeError(f"REPO_MISSING: {REPO}")

os.chdir(REPO)

for _entry in (str(REPO), str(REPO / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# SDPA freeze is the authoritative attention backend override for T4/sm_75
# (the upstream default flash2 backend cannot run).  The R2G driver applies the
# freeze inside run() AFTER the runtime baseline gate and BEFORE any upstream
# mage_flow import or any local bootstrap activation — never at module scope,
# so importing the driver is side-effect free on CPU.
from mage_t4x2.sdpa_contract import (  # noqa: E402
    assert_sdpa_frozen,
    freeze_sdpa_env,
)

# R2G runtime baseline authority (first gate) + clean local bootstrap env.
# The R2G driver verifies the R2G successor baseline only; the R2D/R2E/R2F
# baselines remain historical and are not the successor authority.
from mage_t4x2.r2g_runtime_baseline import verify_r2g_runtime_baseline  # noqa: E402
from mage_t4x2.r2d_bootstrap_environment import (  # noqa: E402
    R2DError,
    prepare_r2d_bootstrap_environment,
)

# R2G clean bootstrap: provisioning is deliberately NOT performed at module
# scope.  It happens once inside run() via prepare_r2d_bootstrap_environment()
# after the runtime baseline gate, using a fresh local target and the exact
# bootstrap lock from the local wheelhouse (offline only, never the network).

# R3 corrective: the ONLY mechanism that may provision upstream ``mage_flow``.
from mage_t4x2.upstream_bootstrap import (  # noqa: E402
    UpstreamMageBootstrapError,
    bootstrap_upstream_mage,
)

# T4 + CUDA driver path used on Kaggle GPU kernels (harmless when absent).
os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)
if "/usr/local/nvidia/lib64" not in sys.path:
    sys.path.insert(0, "/usr/local/nvidia/lib64")

# Corrected project modules: import fresh, never importlib.reload.
from mage_t4x2.device_bridges import VaeInputBridge, derive_vae_device  # noqa: E402
from mage_t4x2.dual_device_forward import DualDeviceForwardAdapter  # noqa: E402
from scripts import gpu_session  # noqa: E402
from scripts.gpu_session import attach_dual_adapter  # noqa: E402

# Authority objects.
from mage_t4x2 import constants as C  # noqa: E402
from mage_t4x2.block_partition import discover_transformer_blocks  # noqa: E402
from mage_t4x2.contracts import default_run_contract  # noqa: E402
from mage_t4x2.image_validation import validate_image  # noqa: E402
from mage_t4x2.model_provenance import ModelPathResolver  # noqa: E402
from mage_t4x2.source_pin import pinned_source_authority  # noqa: E402
from mage_t4x2.stage_b_session import StageBSession  # noqa: E402
from mage_t4x2.telemetry import parse_telemetry  # noqa: E402

# Pinned authority values (runbook authority).
PINNED_MAGE_SHA = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
PINNED_MODEL_REVISION = "65bb3500f0da9df6a41ec6383716fc02cf014773"
EXPECTED_LOCAL_MODEL = (
    "/kaggle/input/models/dangkhoa2016/"
    "mage-flow-community-mage-flow-turbo/pytorch/default/1"
)
REQUIRED_MODEL_FILES = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
]


class AuthorityGateFailure(RuntimeError):
    """First failed authority gate.  Carries the gate name for the summary."""

    def __init__(self, gate: str, message: str) -> None:
        self.gate = gate
        super().__init__(f"{gate}: {message}")


def _require(condition: bool, gate: str, message: str) -> None:
    if not condition:
        raise AuthorityGateFailure(gate, message)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _first_param_ids(module: Any, limit: int = 3) -> Dict[str, int]:
    if module is None:
        return {}
    out: Dict[str, int] = {}
    try:
        for name, tensor in module.named_parameters(recurse=True):
            out[name] = id(tensor)
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def _records_for(
    records: List[Dict[str, Any]],
    run_id: str,
    inference_id: str,
    event: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return [
        r
        for r in records
        if r.get("run_id") == run_id
        and r.get("inference_id") == inference_id
        and (event is None or r.get("event") == event)
    ]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def requirements_sha256_local() -> str:
    """SHA-256 of the pinned bootstrap requirements lock (authority fact)."""
    from mage_t4x2 import bootstrap_dependencies as _bd

    return _bd.requirements_sha256()


def _manifest_sha256_or_none() -> Optional[str]:
    """SHA-256 of the wheelhouse manifest file, or None when missing."""
    manifest_path = REPO / "vendor" / "bootstrap-wheelhouse-manifest.json"
    if not manifest_path.is_file():
        return None
    import hashlib

    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# R2A corrective authority helpers (CPU-safe, fail-closed)
# ---------------------------------------------------------------------------
#
# Live device signature (authority source of truth; plan never substitutes).


def module_device_signature(module: Any) -> Dict[str, Any]:
    """Read the LIVE device signature of a module.

    Scans every parameter and buffer tensor, reporting the full unique device
    set.  ``discrete_device`` is the sole device when exactly one is present
    (ambiguous/None otherwise).  ``parameterless`` modules carry NO fabricated
    device; their placement must be independently corroborated (observed
    placement-validation leaf).  A missing module is ``present: False`` and is
    never assumed CUDA.
    """
    if module is None:
        return {
            "present": False,
            "parameter_count": 0,
            "buffer_count": 0,
            "devices": [],
            "discrete_device": None,
            "cuda_only": None,
            "parameterless": False,
            "non_cuda_devices": [],
            "meta_device_present": False,
        }
    param_devices: List[str] = []
    buffer_devices: List[str] = []
    try:
        for _name, tensor in module.named_parameters(recurse=True):
            param_devices.append(str(tensor.device))
    except Exception:
        pass
    try:
        for _name, tensor in module.named_buffers(recurse=True):
            buffer_devices.append(str(tensor.device))
    except Exception:
        pass
    all_devices = param_devices + buffer_devices
    unique = sorted(set(all_devices))
    parameterless = not bool(unique)
    non_cuda = [d for d in unique if not str(d).startswith("cuda:")]
    meta = "meta" in unique
    if parameterless:
        cuda_only: Optional[bool] = None
    else:
        cuda_only = not non_cuda
    discrete = unique[0] if len(unique) == 1 else None
    return {
        "present": True,
        "parameter_count": len(param_devices),
        "buffer_count": len(buffer_devices),
        "devices": unique,
        "discrete_device": discrete,
        "cuda_only": cuda_only,
        "parameterless": parameterless,
        "non_cuda_devices": non_cuda,
        "meta_device_present": meta,
    }


def _live_block_modules(transformer: Any) -> List[Any]:
    try:
        blocks, _attr = discover_transformer_blocks(transformer)
        return list(blocks)
    except Exception:
        container = getattr(transformer, "blocks", None)
        if container is None:
            return []
        if isinstance(container, dict):
            return list(container.values())
        return list(container)


def authority_placement_paths(live_placements: Dict[str, Any]) -> List[str]:
    """Every authority path whose placement must be independently observed."""
    lp = live_placements or {}
    paths: List[str] = []
    if lp.get("text_encoder") is not None:
        paths.append("text_encoder")
    if lp.get("vae") is not None:
        paths.append("vae")
