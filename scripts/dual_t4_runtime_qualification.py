#!/usr/bin/env python3
"""ROUTING routing authority qualified multistep block-integrity corrective fresh-kernel authoritative driver.

qualified targets (static runbook 2026-09-18, successor to initial/intermediate/hardened):

* qualified-1  NEW STATIC qualified RUNTIME REBASELINE - ``authority/runtime-baseline.json``
         (QUALIFIED_STATIC) covering the successor runtime authority files
         and engaged as the FIRST authority gate
         (RUNTIME_BASELINE_INTEGRITY) before any other cold-start action.
         The initial/intermediate baselines remain historical and immutable; this successor
         baseline describes the corrected successor source.
* qualified-2  FAIL-CLOSED BOOTSTRAP (DEFECT A CORRECTED, PRESERVED) - the bootstrap input
         verifier now reads the ``artifacts`` schema, verifies the lock SHA-256
         against the manifest requirements SHA-256, and byte-verifies every
         wheelhouse artifact (filename dist-version-python-tag, size, SHA-256).
         A corrupt wheelhouse can never pass the authority gate again.
* qualified-3  GENUINELY FRESH BOOTSTRAP SITE (DEFECT B CORRECTED, PRESERVED) - a pre-existing
         bootstrap target can never be reused; ``create_fresh_bootstrap_site``
         raises STALE_BOOTSTRAP_SITE whenever the target already exists, so
         the real freshness marker is always freshly created.
* qualified-4  qualified AUTHORITY DRIVER (MULTISTEP BLOCK-INTEGRITY CORRECTIVE) -
         corrections (dual-T4 split-boundary transfers, return-to-caller,
         latent/caller anchor, placement validation, no-CPU-fallback, exact
         block coverage; per-invocation multi-step segmentation now refutes the
         genuine G1 false negative) under the qualified machine gate names.

Single-run flow (semantic order, never repeats; runbook section 57):

    fresh kernel import -> qualified runtime baseline integrity ->
    freeze SDPA env -> verify bootstrap lock+wheelhouse -> create fresh
    local bootstrap site -> offline provision exact lock -> activate local
    site -> verify local version + origin -> bootstrap_upstream_mage ->
    P0 -> G0 (hardware+session) -> L0 (load exactly once) ->
    derive split-boundary endpoints -> SDPA -> one G1 baseline ->
    one ROUTING routing authority -> live placement/acceptance adjudication ->
    machine summary -> human summary -> STOP.

G3..G6 are NEVER executed and the model is NEVER loaded twice.  Every authority
gate is fail-closed: the first failed gate ends the run with non-zero exit.

qualified CORRECTIVE ROOT CAUSE: phase acceptance previously derived global block-order
validity across all multistep invocation cycles, so a genuine G1 multistep
block-integrity false negative was refused (BLOCK_ORDER_VALID=false,
NO_DUPLICATED_BLOCKS=false). The reducer now segments per invocation before
adjudication; the G1/ROUTING acceptance gates below verify per-invocation block
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
# (the upstream default flash2 backend cannot run).  The qualified driver applies the
# freeze inside run() AFTER the runtime baseline gate and BEFORE any upstream
# mage_flow import or any local bootstrap activation — never at module scope,
# so importing the driver is side-effect free on CPU.
from mage_t4x2.sdpa_contract import (  # noqa: E402
    assert_sdpa_frozen,
    freeze_sdpa_env,
)

# qualified runtime baseline authority (first gate) + clean local bootstrap env.
# The qualified driver verifies the qualified successor baseline only; the initial/intermediate/hardened
# baselines remain historical and are not the successor authority.
from mage_t4x2.runtime_baseline import verify_runtime_baseline  # noqa: E402
from mage_t4x2.bootstrap_environment import (  # noqa: E402
    BootstrapEnvironmentError,
    prepare_bootstrap_environment,
)

# qualified clean bootstrap: provisioning is deliberately NOT performed at module
# scope.  It happens once inside run() via prepare_bootstrap_environment()
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
# qualification corrective authority helpers (CPU-safe, fail-closed)
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
    for name in (lp.get("transformer_pre") or {}):
        paths.append("transformer.pre." + str(name))
    for idx in (lp.get("transformer_blocks") or {}):
        paths.append("transformer.blocks." + str(idx))
    for name in (lp.get("transformer_post") or {}):
        paths.append("transformer.post." + str(name))
    return sorted(paths)


def collect_live_authority_placements(state: Any) -> Dict[str, Any]:
    """Capture the LIVE device signatures of the authority compute path.

    Uses only the live module graph (parameters + buffers).  A module that is
    absent on the state object is captured as ``present: False``.
    """
    live: Dict[str, Any] = {}
    live["text_encoder"] = module_device_signature(getattr(state, "text_encoder", None))
    live["vae"] = module_device_signature(getattr(state, "vae", None))
    transformer = getattr(state, "transformer", None)
    pre: Dict[str, Any] = {}
    for name in C.TRANSFORMER_PRE_MODULES:
        pre[name] = module_device_signature(
            getattr(transformer, name, None) if transformer is not None else None
        )
    post: Dict[str, Any] = {}
    for name in C.TRANSFORMER_POST_MODULES:
        post[name] = module_device_signature(
            getattr(transformer, name, None) if transformer is not None else None
        )
    blocks: Dict[str, Any] = {}
    for i, block in enumerate(_live_block_modules(transformer)):
        blocks[str(i)] = module_device_signature(block)
    live["transformer_pre"] = pre
    live["transformer_blocks"] = blocks
    live["transformer_post"] = post
    return live


def adjudicate_live_authority_placements(live_placements: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-closed adjudication of the LIVE authority placement evidence.

    PASS requires every required component present and CUDA-only.  Parameterless
    modules are classified (never fabricated a device) and require independent
    observed corroboration downstream.
    """
    lp = live_placements or {}
    missing_components: List[str] = []
    for group in ("text_encoder", "vae", "transformer_pre", "transformer_blocks", "transformer_post"):
        value = lp.get(group)
        if group in ("text_encoder", "vae"):
            sig = value or {}
            if value is None or sig.get("present") is not True:
                missing_components.append(group)
        elif value is None or value == {}:
            missing_components.append(group)
    sig_rows: List[Tuple[str, Dict[str, Any]]] = []
    te = lp.get("text_encoder") or {}
    sig_rows.append(("text_encoder", te))
    vae = lp.get("vae") or {}
    sig_rows.append(("vae", vae))
    for name, sig in (lp.get("transformer_pre") or {}).items():
        sig_rows.append(("transformer.pre." + str(name), sig))
    for idx, sig in (lp.get("transformer_blocks") or {}).items():
        sig_rows.append(("transformer.blocks." + str(idx), sig))
    for name, sig in (lp.get("transformer_post") or {}).items():
        sig_rows.append(("transformer.post." + str(name), sig))
    non_cuda: List[str] = []
    parameterless: List[str] = []
    cuda_ok = True
    for path, sig in sig_rows:
        sig = sig or {}
        if sig.get("present") is not True:
            non_cuda.append(path)
            cuda_ok = False
        elif sig.get("cuda_only") is False:
            non_cuda.append(path)
            cuda_ok = False
        elif sig.get("cuda_only") is None and not sig.get("parameterless"):
            non_cuda.append(path)
            cuda_ok = False
        elif sig.get("parameterless"):
            parameterless.append(path)
    present = bool(lp)
    cuda_only = cuda_ok and not missing_components and not non_cuda
    return {
        "present": present,
        "pass": bool(present and cuda_only),
        "cuda_only": bool(cuda_only),
        "missing_components": missing_components,
        "non_cuda_live_devices": non_cuda,
        "parameterless_components": parameterless,
    }


def collect_observed_placement_validation(
    placement_validation: Any,
) -> Dict[str, Any]:
    """Reduce the placement-validation document to OBSERVED device evidence.

    Every leaf carries an ``observed_device``; evidence is present only when the
    observed device is a real non-empty ``cuda:N`` value (never '' / 'N/A' /
    'meta' / ambiguous comma-joined).  Missing or CPU observed evidence FAILs
    closed and is surfaced for the no-CPU-fallback adjudication.
    """
    if not isinstance(placement_validation, dict):
        return {
            "present": False,
            "status": "NOT_RUN",
            "cuda_only": False,
            "pass": False,
            "path_to_observed": {},
            "missing_validation_evidence": ["placement.leaves"],
            "non_cuda_validation_devices": [],
        }
    leaves = ((placement_validation.get("placement") or {}).get("leaves")) or []
    if leaves == []:
        leaves = []
    path_to_observed: Dict[str, str] = {}
    missing: List[str] = []
    non_cuda: List[Dict[str, Any]] = []
    for leaf in leaves:
        if not isinstance(leaf, dict):
            continue
        path = leaf.get("path")
        if path is None:
            continue
        observed = leaf.get("observed_device")
        if observed is None:
            missing.append(str(path))
            continue
        value = str(observed).strip()
        if value == "" or str(value).lower() == "n/a":
            missing.append(str(path))
            continue
        path_to_observed[str(path)] = value
        if not str(value).startswith("cuda:") or "," in str(value):
            non_cuda.append({"path": str(path), "observed_device": value})
    pv_status = str(placement_validation.get("status") or "NOT_RUN")
    placement_status = str(
        ((placement_validation.get("placement") or {}).get("status")) or "NOT_RUN"
    )
    present = bool(leaves)
    if not present:
        missing.append("placement.leaves")
    cuda_only = bool(present and not non_cuda and not missing)
    pass_ok = bool(present and pv_status == "PASS" and placement_status == "PASS" and cuda_only)
    return {
        "present": present,
        "status": pv_status,
        "cuda_only": bool(cuda_only),
        "pass": bool(pass_ok),
        "path_to_observed": path_to_observed,
        "missing_validation_evidence": missing,
        "non_cuda_validation_devices": non_cuda,
    }


def _telemetry_device_endpoints(scoped_telemetry: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract device-bearing endpoints from scoped ROUTING telemetry events.

    Only routing-authority events are honored: block_forward (device), and the
    cross / return / vae-input transfer families (from,to).  Unknown event kinds
    are ignored rather than fabulating endpoints.
    """
    known = {
        "block_forward",
        "cross_device_transfer",
        "transformer_output_return_transfer",
        "vae_input_transfer",
    }
    endpoints: List[str] = []
    non_cuda: List[Dict[str, Any]] = []
    for ev in scoped_telemetry or []:
        event = ev.get("event")
        if event not in known:
            continue
        values: List[Tuple[str, str]] = []
        if event == "block_forward":
            device = ev.get("device")
            if device is not None and str(device).strip() != "":
                values.append(("device", str(device)))
        else:
            if ev.get("from") is not None:
                values.append(("from", str(ev.get("from"))))
            if ev.get("to") is not None:
                values.append(("to", str(ev.get("to"))))
        for label, value in values:
            endpoints.append(value)
            if not str(value).startswith("cuda:"):
                non_cuda.append({"event": event, "endpoint": label, "device": value})
    present = bool(endpoints)
    return {
        "device_endpoints": endpoints,
        "present": bool(present),
        "cuda_only": bool(present and not non_cuda),
        "non_cuda_devices": non_cuda,
    }


def adjudicate_no_cpu_fallback(
    live_placements: Dict[str, Any],
    pv_observed: Dict[str, Any],
    scoped_telemetry: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """TRUE observed NO-CPU-FALLBACK proof over three independent sources.

    Never accepts memory-plan-only evidence.  PASS requires:

    * live authority placements present and CUDA-only (parameterless modules
      permitted but requiring observed corroboration below);
    * observed placement-validation evidence present, status PASS, CUDA-only,
      and covering EVERY required authority path (missing/uncovered leaf fails);
    * every parameterless authority component has an observed CUDA leaf;
    * scoped ROUTING device-bearing routing evidence present and CUDA-only.
    """
    live_adj = adjudicate_live_authority_placements(live_placements or {})
    pv: Dict[str, Any] = pv_observed or {}
    covered = (pv.get("path_to_observed") or {})
    uncovered: List[str] = []
    for path in authority_placement_paths(live_placements or {}):
        device = covered.get(path)
        if not device or not str(device).startswith("cuda:"):
            uncovered.append(path)
    missing_param: List[str] = []
    for path in live_adj.get("parameterless_components") or []:
        device = covered.get(path)
        if not device or not str(device).startswith("cuda:"):
            missing_param.append(path)
    routing = _telemetry_device_endpoints(scoped_telemetry)
    ok = bool(
        live_adj.get("present")
        and live_adj.get("cuda_only")
        and pv.get("present")
        and pv.get("pass")
        and pv.get("cuda_only")
        and routing.get("present")
        and routing.get("cuda_only")
        and not uncovered
        and not missing_param
    )
    adjudication = {
        "live_placement_present": bool(live_adj.get("present")),
        "live_placement_cuda_only": bool(live_adj.get("cuda_only")),
        "placement_validation_present": bool(pv.get("present")),
        "placement_validation_status": str(pv.get("status")),
        "placement_validation_cuda_only": bool(pv.get("cuda_only")),
        "routing_evidence_present": bool(routing.get("present")),
        "routing_cuda_only": bool(routing.get("cuda_only")),
        "uncovered_authority_paths": uncovered,
        "missing_parameterless_corroboration": missing_param,
        "non_cuda_routing_devices": routing.get("non_cuda_devices") or [],
        "routing_device_endpoints": routing.get("device_endpoints") or [],
    }
    return {
        "no_cpu_fallback_observed": ok,
        "status": "PASS" if ok else "FAIL",
        "no_cpu_fallback_adjudication": adjudication,
    }


def derive_cpu_fallback_config(contract: Any) -> Dict[str, Any]:
    """Derive CPU_FALLBACK_FORBIDDEN_CONFIG from the LIVE RunContract.

    Authority truth is the live ``cpu_fallback`` field of the run contract.
    A missing field or a truthy value fails closed; only an explicit
    ``cpu_fallback`` value of False forbids CPU fallback.
    """
    observed = getattr(contract, "cpu_fallback", None)
    return {
        "cpu_fallback_config_observed_value": observed,
        "cpu_fallback_forbidden_config": observed is False,
        "authority_source": "live RunContract.cpu_fallback",
    }


def derive_split_boundary_endpoints(
    block_devices: Dict[str, Any],
    split_block: Optional[int],
) -> Dict[str, Any]:
    """EXACT split-boundary transfer endpoints ``block_devices[K-1] -> block_devices[K]``.

    Rejects min/max block-index endpoint derivation.  The boundary is validated
    only when both adjacent blocks exist, sit on distinct devices, and both are
    CUDA.
    """
    bd = block_devices or {}
    num_blocks = len(bd)
    missing = {
        "found": False,
        "validated": False,
        "expected_from": None,
        "expected_to": None,
        "from_block": None,
        "to_block": None,
        "reason": None,
    }
    if split_block is None:
        out = dict(missing)
        out["reason"] = "split_block_missing"
        return out
    try:
        k = int(split_block)
    except (TypeError, ValueError):
        out = dict(missing)
        out["reason"] = "split_block_invalid"
        return out
    if k < 1:
        out = dict(missing)
        out["reason"] = "split_block_lt_1"
        return out
    if k >= num_blocks:
        out = dict(missing)
        out["reason"] = "split_block_out_of_range"
        return out
    key_from = str(k - 1)
    key_to = str(k)
    if key_from not in bd or key_to not in bd:
        out = dict(missing)
        out["from_block"] = k - 1
        out["to_block"] = k
        out["reason"] = "boundary_block_missing"
        return out
    expected_from = str(bd[key_from])
    expected_to = str(bd[key_to])
    base = {
        "found": True,
        "expected_from": expected_from,
        "expected_to": expected_to,
        "from_block": k - 1,
        "to_block": k,
    }
    if expected_from == expected_to:
        out = dict(base)
        out["validated"] = False
        out["reason"] = "boundary_devices_equal"
        return out
    if not (expected_from.startswith("cuda:") and expected_to.startswith("cuda:")):
        out = dict(base)
        out["validated"] = False
        out["reason"] = "boundary_endpoint_non_cuda"
        return out
    out = dict(base)
    out["validated"] = True
    out["reason"] = None
    return out


def derive_latent_anchor_device(
    live_placements: Dict[str, Any],
    memory_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Latent/caller anchor from the LIVE transformer-pre authority.

    The anchor is the discrete CUDA device of the transformer-pre input/caller
    modules (img_in first).  When live evidence is unavailable, the canonical
    single-consistent transformer-pre plan placement corroborates; a
    contradictory or absent anchor is UNPROVEN and fails closed.
    """
    lp = live_placements or {}
    pre = lp.get("transformer_pre") or {}
    candidates: List[Tuple[str, str]] = []
    for name in C.TRANSFORMER_PRE_MODULES:
        sig = pre.get(name) or {}
        if sig.get("present") and not sig.get("parameterless"):
            device = sig.get("discrete_device")
            if device is not None and str(device).startswith("cuda:"):
                candidates.append((str(name), str(device)))
    if candidates:
        distinct = sorted({d for _n, d in candidates})
        if len(distinct) == 1:
            name = next(n for n, d in candidates if d == distinct[0])
            return {
                "anchor_device": distinct[0],
                "source": "live_transformer_pre." + name,
                "unproven": False,
                "reason": None,
            }
        return {
            "anchor_device": None,
            "source": "live_transformer_pre_ambiguous",
            "unproven": True,
            "reason": "live transformer-pre devices disagree",
        }
    m = memory_plan or {}
    components = m.get("components") or {}
    device_plan = m.get("device_plan") or {}
    tr_plan = device_plan.get("transformer") or {}
    plan_values: List[str] = []
    comp_dev = (components.get("transformer_pre") or {}).get("device")
    if comp_dev is not None:
        plan_values.append(str(comp_dev))
    for _name, dev in (tr_plan.get("pre") or {}).items():
        if dev is not None:
            plan_values.append(str(dev))
    distinct_plan = sorted(set(plan_values))
    if distinct_plan and len(distinct_plan) == 1 and distinct_plan[0].startswith("cuda:"):
        return {
            "anchor_device": distinct_plan[0],
            "source": "plan_transformer_pre",
            "unproven": False,
            "reason": None,
        }
    return {
        "anchor_device": None,
        "source": "unproven",
        "unproven": True,
        "reason": "latent anchor unproven",
    }


def derive_post_head_device_expected(
    memory_plan: Dict[str, Any],
    live_post: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Post/head expected device cross-check (plan + final block + live post).

    Every corroborating source must agree on a single CUDA device: plan
    ``device_plan.transformer.post`` entries, ``components.transformer_post``,
    the planned ``final_block_device``, and (when available) the live post
    modules.  Disagreements are reported per source and FAIL the check.
    """
    m = memory_plan or {}
    components = m.get("components") or {}
    device_plan = m.get("device_plan") or {}
    tr_plan = device_plan.get("transformer") or {}
    post_plan = tr_plan.get("post") or {}
    comp_post = components.get("transformer_post") or {}
    final_block_device = device_plan.get("final_block_device")
    sources: List[Tuple[str, str]] = []
    if isinstance(post_plan, dict):
        for name, dev in post_plan.items():
            if dev is not None:
                sources.append(("plan_post." + str(name), str(dev)))
    comp_dev = comp_post.get("device")
    if comp_dev is not None:
        sources.append(("components_transformer_post", str(comp_dev)))
    primary_present = bool(sources)
    if final_block_device is not None:
        sources.append(("final_block_device", str(final_block_device)))
    uniform = primary_present and len({v for _s, v in sources}) == 1
    expected = sources[0][1] if primary_present else None
    expected_cuda = bool(expected) and str(expected).startswith("cuda:")
    disagreement: List[str] = []
    if expected is not None:
        for label, dev in sources:
            if dev != expected:
                disagreement.append(label + "=" + dev)
    live_present = live_post is not None
    if live_present:
        for name, sig in live_post.items():
            if not isinstance(sig, dict):
                continue
            device = sig.get("discrete_device")
            if device is not None and str(device) != expected:
                disagreement.append("live_post." + str(name) + "=" + str(device))
    valid = bool(primary_present and uniform and expected_cuda and not disagreement)
    if not live_present:
        valid = False
        disagreement.append("live_post=missing")
    return {
        "expected": expected,
        "valid": valid,
        "uniform": uniform,
        "cuda": expected_cuda,
        "sources": {label: dev for label, dev in sources},
        "disagreement": disagreement,
        "live_post_present": live_present,
        "reason": None if valid else "post_head_expected_disagreement",
    }


def derive_live_vs_plan_placement_match(
    live_placements: Dict[str, Any],
    memory_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-path live-vs-plan placement consistency over the authority path.

    Every live discrete device is compared to its plan expectation; a live
    device that is missing, ambiguous, or different fails the match.
    """
    m = memory_plan or {}
    components = m.get("components") or {}
    device_plan = m.get("device_plan") or {}
    tr_plan = device_plan.get("transformer") or {}
    pre_plan = tr_plan.get("pre") or {}
    post_plan = tr_plan.get("post") or {}
    block_devices = m.get("block_devices") or {}
    checked: Dict[str, Any] = {}
    mismatches: List[str] = []

    def _cmp(path: str, live: Any, planned: Any) -> None:
        lv = str(live) if live is not None else None
        pv = str(planned) if planned is not None else None
        checked[path] = {"live": lv, "plan": pv}
        if lv != pv:
            mismatches.append(f"{path}: live={lv} plan={pv}")

    lp = live_placements or {}
    _cmp(
        "text_encoder",
        (lp.get("text_encoder") or {}).get("discrete_device"),
        (components.get("text_encoder") or {}).get("device"),
    )
    pre_default = (components.get("transformer_pre") or {}).get("device")
    for name, sig in (lp.get("transformer_pre") or {}).items():
        plan_value = (pre_plan or {}).get(name)
        if plan_value is None:
            plan_value = pre_default
        _cmp("transformer.pre." + str(name), sig.get("discrete_device"), plan_value)
    for idx, sig in (lp.get("transformer_blocks") or {}).items():
        _cmp(
            "transformer.blocks." + str(idx),
            sig.get("discrete_device"),
            block_devices.get(str(idx)),
        )
    post_default = (components.get("transformer_post") or {}).get("device")
    for name, sig in (lp.get("transformer_post") or {}).items():
        plan_value = (post_plan or {}).get(name)
        if plan_value is None:
            plan_value = post_default
        _cmp("transformer.post." + str(name), sig.get("discrete_device"), plan_value)
    _cmp(
        "vae",
        (lp.get("vae") or {}).get("discrete_device"),
        (components.get("vae") or {}).get("device"),
    )
    return {"match": not mismatches, "mismatches": mismatches, "checked": checked}


def adjudicate_transfer_cardinality(
    events: List[Dict[str, Any]],
    from_device: str,
    to_device: str,
    expected_count: int,
) -> Dict[str, Any]:
    """STRICT transfer cardinality over scoped events.

    PASS requires the observed event count to equal the EXPLICIT expected count
    AND every event to satisfy ``from == from_device, to == to_device,
    from != to``.  Zero events FAIL.  The expected count is never derived from
    the observed length.
    """
    events = events or []
    expected = int(expected_count)
    violations: List[Dict[str, Any]] = []
    for ev in events:
        frm = str(ev.get("from"))
        to = str(ev.get("to"))
        if not (
            frm == str(from_device)
            and to == str(to_device)
            and frm != to
        ):
            violations.append({"from": frm, "to": to})
    exact_count = len(events) == expected
    ok = bool(events) and exact_count and not violations
    return {
        "pass": ok,
        "event_count": len(events),
        "expected_count": expected,
        "exact_count": exact_count,
        "violation_count": len(violations),
        "violations": violations,
    }


def adjudicate_post_head_device(
    return_events: List[Dict[str, Any]],
    expected_device: str,
    expected_count: int,
    anchor_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Post/head device proof over the transformer-output-return witness.

    PASS requires every return event to originate on the live-expected post/head
    device (and, when known, to land on the live latent-anchor device), with the
    observed count equal to N.  Missing/mixed/CPU evidence FAILs.
    """
    events = return_events or []
    expected = int(expected_count)
    violations: List[Dict[str, Any]] = []
    from_devices = sorted({str(ev.get("from")) for ev in events})
    for ev in events:
        frm = str(ev.get("from"))
        to = str(ev.get("to"))
        ok_from = frm == str(expected_device)
        ok_to = True
        if anchor_device:
            ok_to = to == str(anchor_device)
        if not (ok_from and ok_to and frm != to):
            violations.append({"from": frm, "to": to})
    exact_count = len(events) == expected
    ok = bool(events) and exact_count and not violations
    return {
        "pass": ok,
        "event_count": len(events),
        "expected_count": expected,
        "exact_count": exact_count,
        "expected_device": str(expected_device),
        "observed_from_devices": from_devices,
        "violation_count": len(violations),
        "violations": violations,
    }


def _gate_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() == "PASS"


def derive_p0_g0_summary(
    p0_status: Any,
    g0_hardware_status: Any,
    g0_session_status: Any,
) -> Dict[str, str]:
    """Combine the separate P0 / G0 sub-gates without aliasing P0 to G0."""
    p0 = _gate_bool(p0_status)
    hw = _gate_bool(g0_hardware_status)
    sess = _gate_bool(g0_session_status)
    return {
        "p0_status": "PASS" if p0 else "FAIL",
        "g0_hardware_status": "PASS" if hw else "FAIL",
        "g0_session_status": "PASS" if sess else "FAIL",
        "g0_status": "PASS" if (hw and sess) else "FAIL",
    }


def build_output_acceptance(
    image_valid: bool,
    no_nan: bool,
    no_inf: bool,
    path_exists: bool,
) -> Dict[str, Any]:
    """Combined ROUTING output gate: file present + valid image + no NaN + no Inf."""
    return {
        "path_exists": bool(path_exists),
        "image_valid": bool(image_valid),
        "no_nan": bool(no_nan),
        "no_inf": bool(no_inf),
        "valid": bool(path_exists and image_valid and no_nan and no_inf),
    }


def persist_live_block_count(summary: Dict[str, Any], blocks_live: Any) -> Dict[str, Any]:
    """Persist ``num_blocks`` from the LIVE discovered block count."""
    if isinstance(blocks_live, int):
        n = max(int(blocks_live), 0)
    else:
        n = len(blocks_live) if blocks_live is not None else 0
    summary["num_blocks"] = n
    return summary


def human_fact(s: Dict[str, Any], key: str) -> str:
    """Same-named fact rendering: PASS only when the key is exactly True."""
    return "PASS" if s.get(key) is True else "FAIL"


def _human_passthrough(s: Dict[str, Any], key: str) -> str:
    """Same-named fact rendering for string/bool gate statuses."""
    value = s.get(key)
    if value is True or str(value).strip().upper() == "PASS":
        return "PASS"
    return "FAIL"


def human_g3_g6(s: Dict[str, Any]) -> str:
    """Machine-derived G3/G6 execution fact (never a hard-coded literal).

    ``NO`` only when the observed machine fact is explicitly False; absent
    evidence renders ``UNPROVEN`` so the human line is never PASS-liegraded.
    """
    if s.get("g3_g6_executed") is False:
        return "NO"
    if s.get("g3_g6_executed") is True:
        return "YES"
    return "UNPROVEN"


def summary_provenance_map() -> Dict[str, Dict[str, str]]:
    """provenance summary provenance: every authority line maps to its OWN machine fact.

    Authority entries use ``default_if_missing`` FAIL: missing evidence is
    NEVER reported PASS.  Informational entries are classified explicitly with
    ``kind=informational``.  Coverage is exact: the map keys equal every line
    emitted by ``_print_human_summary``.
    """
    authority = {
        "BOOTSTRAP_REQUIREMENTS_LOCK": {
            "source": "bootstrap_requirements_sha256",
            "evidence": "pinned bootstrap requirements-lock authority (sha256 present)",
        },
        "BOOTSTRAP_WHEELHOUSE_INTEGRITY": {
            "source": "bootstrap_wheelhouse_artifacts",
            "evidence": "local offline wheelhouse verified against hashed manifest",
        },
        "BOOTSTRAP_DEP_PREFLIGHT": {
            "source": "bootstrap_dependency_preflight_gate",
            "evidence": "installed bootstrap dependency states classified fail-closed",
        },
        "BOOTSTRAP_DEP_PROVISIONING": {
            "source": "bootstrap_dependency_provisioning_gate",
            "evidence": "offline local provisioning result (PASS / NOT_NEEDED)",
        },
        "BOOTSTRAP_DEP_POSTVERIFY": {
            "source": "bootstrap_dependency_states_after.status",
            "evidence": "post-provision exact-version + import verification",
        },
        "UPSTREAM_BOOTSTRAP": {
            "source": "upstream_bootstrap.source_root",
            "evidence": "pinned upstream source-root authority",
        },
        "P0": {
            "source": "p0_status",
            "evidence": "P0 fresh-kernel preflight gate",
        },
        "G0": {
            "source": "g0_status",
            "evidence": "combined G0 (hardware + session) status",
        },
        "L0": {
            "source": "l0_status",
            "evidence": "dual-T4 spread load acceptance",
        },
        "SDPA": {
            "source": "sdpa.status",
            "evidence": "frozen SDPA env gate (optimal path)",
        },
        "G1_PREREQUISITE": {
            "source": "g1_status",
            "evidence": "G1 routing-authority inference status",
        },
        "ROUTING": {
            "source": "routing_status",
            "evidence": "ROUTING routing authority inference status",
        },
        "RUNTIME_STATE_ID_STABLE": {
            "source": "runtime_state_id_stable",
            "evidence": "runtime state id did not change across inference",
        },
        "TRANSFORMER_ID_STABLE": {
            "source": "transformer_id_stable",
            "evidence": "transformer id did not change across inference",
        },
        "TEXT_ENCODER_ID_STABLE": {
            "source": "text_encoder_id_stable",
            "evidence": "text-encoder id did not change across inference",
        },
        "VAE_ID_STABLE": {
            "source": "vae_id_stable",
            "evidence": "vae id did not change across inference",
        },
        "EXACT_BLOCK_COVERAGE": {
            "source": "routing_block_order_valid",
            "evidence": "exact ROUTING block coverage (gate passes block order)",
        },
        "BLOCK_ORDER": {
            "source": "routing_block_order_valid",
            "evidence": "ROUTING observed block order equals planned order",
        },
        "NO_SKIP": {
            "source": "routing_no_skipped_blocks",
            "evidence": "no ROUTING block index skipped",
        },
        "NO_DUPLICATE_WITHIN_INVOCATION": {
            "source": "routing_no_duplicated_blocks",
            "evidence": "no ROUTING block index duplicated within invocation",
        },
        "GPU0_PARTICIPATION": {
            "source": "routing_gpu0_participation",
            "evidence": "ROUTING observed cuda:0 participation",
        },
        "GPU1_PARTICIPATION": {
            "source": "routing_gpu1_participation",
            "evidence": "ROUTING observed cuda:1 participation",
        },
        "TRANSFER_0_TO_1": {
            "source": "routing_transfer_0_to_1_valid",
            "evidence": "scoped cross_device_transfer events adjudicated by split-boundary cardinality",
        },
        "TRANSFORMER_RETURN_1_TO_0": {
            "source": "routing_transformer_return_valid",
            "evidence": "scoped transformer_output_return_transfer events by count+endpoints",
        },
        "POST_HEAD_DEVICE": {
            "source": "post_head_device_valid",
            "evidence": "expected post/head cross-check + audited return witness",
        },
        "VAE_INPUT_TRANSFER": {
            "source": "routing_vae_input_transfer_valid",
            "evidence": "scoped vae input transfer events by count+endpoints",
        },
        "CPU_FALLBACK_FORBIDDEN_CONFIG": {
            "source": "cpu_fallback_forbidden_config",
            "evidence": "authority configuration forbids CPU fallback (derived from RunContract)",
        },
        "NO_CPU_FALLBACK_OBSERVED": {
            "source": "no_cpu_fallback_observed",
            "evidence": "live placement + observed placement-validation + scoped telemetry",
        },
        "NO_NAN": {
            "source": "routing_no_nan",
            "evidence": "observed_facts NO_NAN of the ROUTING phase",
        },
        "NO_INF": {
            "source": "routing_no_inf",
            "evidence": "observed_facts NO_INF of the ROUTING phase",
        },
        "OUTPUT_VALID": {
            "source": "routing_output_valid",
            "evidence": "combined output gate (path + image + no-NaN + no-Inf)",
        },
        "G3_G6_EXECUTED": {
            "source": "g3_g6_executed",
            "evidence": "machine fact whether the G3/G6 future phase has executed",
        },
        "FINAL_VERDICT": {
            "source": "status.routing_status",
            "evidence": "final combined verdict (driver status + ROUTING status)",
        },
    }
    informational = {
        "SESSION_ID": {
            "source": "session_id",
            "evidence": "fresh session identifier",
        },
        "SESSION_STATE": {
            "source": "session_state",
            "evidence": "session state name",
        },
        "EVIDENCE_ROOT": {
            "source": "evidence_root",
            "evidence": "static evidence root path",
        },
        "MODEL_LOAD_COUNT": {
            "source": "model_load_count",
            "evidence": "model load count observed by the fresh session",
        },
        "SPLIT_BLOCK": {
            "source": "split_block",
            "evidence": "split-block K from the L0 memory plan",
        },
        "NUM_BLOCKS": {
            "source": "num_blocks",
            "evidence": "live discovered transformer block count",
        },
        "G1_INFERENCE_ID": {
            "source": "g1_inference_id",
            "evidence": "G1 inference identifier",
        },
        "ROUTING_INFERENCE_ID": {
            "source": "routing_inference_id",
            "evidence": "ROUTING inference identifier",
        },
        "EXPECTED_TRANSFORMER_INVOCATIONS": {
            "source": "expected_transformer_invocations",
            "evidence": "expected ROUTING transformer invocation cardinality",
        },
        "TRANSFER_0_TO_1_COUNT": {
            "source": "routing_transfer_0_to_1_event_count/routing_transfer_0_to_1_expected_count",
            "evidence": "observed vs expected cuda0->cuda1 transfer count",
        },
        "TRANSFORMER_RETURN_1_TO_0_COUNT": {
            "source": "routing_transformer_return_event_count/routing_transformer_return_expected_count",
            "evidence": "observed vs expected transformer output return count",
        },
        "POST_HEAD_DEVICE_EXPECTED": {
            "source": "post_head_device_expected",
            "evidence": "expected post/head device from split-boundary authority",
        },
        "POST_HEAD_DEVICE_OBSERVED": {
            "source": "post_head_device_observed",
            "evidence": "observed post/head device from scoped return telemetry",
        },
        "VAE_INPUT_TRANSFER_COUNT": {
            "source": "routing_vae_input_transfer_event_count/routing_vae_input_transfer_expected_count",
            "evidence": "observed vs expected vae input transfer count",
        },
    }
    entries: Dict[str, Dict[str, str]] = {}
    for label, entry in authority.items():
        entry["kind"] = "authority"
        entry["default_if_missing"] = "FAIL"
        entries[label] = entry
    for label, entry in informational.items():
        entry["kind"] = "informational"
        entries[label] = entry
    return entries


# ---------------------------------------------------------------------------
# qualified human-summary provenance (runbook 2026-09-18, sections 34-36)
# ---------------------------------------------------------------------------

DIAGNOSTIC_NAME = "DUAL_T4_RUNTIME_QUALIFICATION"

HUMAN_SUMMARY_KEYS = [
    "RUNTIME_BASELINE_INTEGRITY",
    "BOOTSTRAP_REQUIREMENTS_LOCK",
    "BOOTSTRAP_WHEELHOUSE_INTEGRITY",
    "BOOTSTRAP_LOCAL_SITE_FRESH",
    "BOOTSTRAP_LOCAL_PROVISION",
    "BOOTSTRAP_LOCAL_VERSION_VERIFY",
    "BOOTSTRAP_LOCAL_ORIGIN_VERIFY",
    "UPSTREAM_BOOTSTRAP",
    "P0",
    "G0",
    "L0",
    "SDPA",
    "G1",
    "ROUTING",
    "MODEL_LOAD_COUNT",
    "G1_INFERENCE_ID",
    "ROUTING_INFERENCE_ID",
    "FINAL_VERDICT",
]

MACHINE_FACT_AUTHORITY_KEYS = list(HUMAN_SUMMARY_KEYS)


def runtime_provenance_coverage() -> Dict[str, Any]:
    """Exact-set coverage of qualified machine-fact authority keys vs human summary.

    Every human-summary line must map to exactly one machine-fact key and vice
    versa (no missing, no extra, exact equality).  This mirrors the runbook
    summary-provenance requirement for the qualified diagnostic.
    """
    human = set(HUMAN_SUMMARY_KEYS)
    machine = set(MACHINE_FACT_AUTHORITY_KEYS)
    return {
        "equal": human == machine,
        "missing": sorted(human - machine),
        "extra": sorted(machine - human),
    }


def render_human_summary(machine: Dict[str, Any]) -> Dict[str, Any]:
    """Render the qualified human-summary lines from the machine-fact dict.

    Fail-closed: any missing or empty machine fact renders ``FAIL``, never a
    blank pass.  FINAL_VERDICT is PASS only when every qualified authority gate fact
    is exactly PASS.
    """
    rendered: Dict[str, Any] = {}
    for key in HUMAN_SUMMARY_KEYS:
        value = machine.get(key)
        rendered[key] = value if value not in (None, "") else "FAIL"
    gates = [
        "RUNTIME_BASELINE_INTEGRITY",
        "BOOTSTRAP_REQUIREMENTS_LOCK",
        "BOOTSTRAP_WHEELHOUSE_INTEGRITY",
        "BOOTSTRAP_LOCAL_SITE_FRESH",
        "BOOTSTRAP_LOCAL_PROVISION",
        "BOOTSTRAP_LOCAL_VERSION_VERIFY",
        "BOOTSTRAP_LOCAL_ORIGIN_VERIFY",
        "UPSTREAM_BOOTSTRAP",
        "P0",
        "G0",
        "L0",
        "SDPA",
        "G1",
        "ROUTING",
    ]
    complete = set(HUMAN_SUMMARY_KEYS) <= set(machine)
    all_pass = all(str(machine.get(g, "")).upper() == "PASS" for g in gates)
    rendered["FINAL_VERDICT"] = "PASS" if (complete and all_pass) else "FAIL"
    return rendered


# ---------------------------------------------------------------------------
# Block routing adjudication (kept for the BLOCK_ROUTING gate only)
# ---------------------------------------------------------------------------


def adjudicate_block_routing(
    observed_blocks: Dict[int, str],
    plan_blocks: Dict[int, str],
    num_blocks_live: int,
) -> Dict[str, Any]:
    """BLOCK_ROUTING gate over the COMPLETE block-index -> device mapping.

    PASS requires both:
      1. the observed block index set covers exactly ``[0, num_blocks_live)``;
      2. the full ``{index: device}`` mapping equals the L0 memory-aware plan.
    """
    order_valid = (
        sorted(observed_blocks) == list(range(num_blocks_live))
        and len(observed_blocks) == num_blocks_live
    )
    mapping_equal = set(observed_blocks.items()) == set(plan_blocks.items())
    ok = bool(order_valid and mapping_equal)
    mismatches: List[Dict[str, Any]] = []
    for idx in sorted(set(observed_blocks) | set(plan_blocks)):
        obs = observed_blocks.get(idx)
        plan = plan_blocks.get(idx)
        if str(obs) != str(plan):
            mismatches.append({"block": idx, "observed": obs, "planned": plan})
    return {
        "pass": ok,
        "order_valid": bool(order_valid),
        "mapping_equal": bool(mapping_equal),
        "observed": {int(k): str(v) for k, v in observed_blocks.items()},
        "planned": {int(k): str(v) for k, v in plan_blocks.items()},
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def phase_block_integrity(facts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Surface the ALREADY-derived phase block-integrity facts."""
    f = facts or {}
    return {
        "block_order_valid": bool(f.get("BLOCK_ORDER_VALID")),
        "no_duplicated_blocks": bool(f.get("NO_DUPLICATED_BLOCKS")),
        "no_skipped_blocks": bool(f.get("NO_SKIPPED_BLOCKS")),
        "cross_inference_clean": bool(f.get("CROSS_INFERENCE_CLEAN")),
        "gpu0_participation": bool(f.get("GPU0_PARTICIPATION")),
        "gpu1_participation": bool(f.get("GPU1_PARTICIPATION")),
    }


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


class DualT4QualificationDriver:
    """Collects gate results and machine-readable summary fields for qualified."""

    def __init__(self, evidence_root: Path) -> None:
        self.evidence_root = evidence_root
        self.session: Any = None
        self.summary: Dict[str, Any] = {
            "diagnostic_name": "DUAL_T4_RUNTIME_QUALIFICATION",
            "evidence_root": str(evidence_root),
            "status": "NOT_RUN",
            "first_failed_gate": None,
            "failure_code": None,
            "error": None,
            "exception_type": None,
            "exception_message": None,
            "session_id": None,
            "session_state": None,
            "bootstrap_requirements": None,
            "bootstrap_requirements_sha256": None,
            "bootstrap_wheelhouse_manifest_sha256": None,
            "bootstrap_wheelhouse_artifacts": None,
            "bootstrap_dependency_states_before": None,
            "bootstrap_dependency_states_after": None,
            "bootstrap_site": None,
            "bootstrap_pip_result": None,
            "bootstrap_requirements_lock_gate": None,
            "bootstrap_wheelhouse_integrity_gate": None,
            "bootstrap_dependency_preflight_gate": None,
            "bootstrap_dependency_provisioning_gate": None,
            "bootstrap_dependency_postverify_gate": None,
            "p0_status": None,
            "g0_hardware_status": None,
            "g0_session_status": None,
            "g0_status": None,
            "model_load_count": None,
            "split_block": None,
            "num_blocks": None,
            "expected_transformer_invocations": None,
            "runtime_state_id": None,
            "transformer_id": None,
            "text_encoder_id": None,
            "vae_id": None,
            "g1_inference_id": None,
            "g1_output_path": None,
            "g1_evidence_paths": None,
            "g1_status": None,
            "routing_inference_id": None,
            "routing_output_path": None,
            "routing_evidence_paths": None,
            "routing_status": None,
            "runtime_state_id_stable": False,
            "transformer_id_stable": False,
            "text_encoder_id_stable": False,
            "vae_id_stable": False,
            "routing_block_order_valid": False,
            "routing_no_skipped_blocks": False,
            "routing_no_duplicated_blocks": False,
            "routing_cross_inference_clean": False,
            "routing_gpu0_participation": False,
            "routing_gpu1_participation": False,
            "routing_transfer_0_to_1_event_count": None,
            "routing_transfer_0_to_1_expected_count": None,
            "routing_transfer_0_to_1_valid": False,
            "routing_transformer_return_event_count": None,
            "routing_transformer_return_expected_count": None,
            "routing_transformer_return_valid": False,
            "post_head_device_expected": None,
            "post_head_device_observed": None,
            "post_head_device_valid": False,
            "vae_device_expected": None,
            "routing_vae_input_transfer_event_count": None,
            "routing_vae_input_transfer_expected_count": None,
            "routing_vae_input_transfer_valid": False,
            "cpu_fallback_config_observed_value": None,
            "cpu_fallback_forbidden_config": False,
            "no_cpu_fallback_observed": False,
            "routing_no_nan": None,
            "routing_no_inf": None,
            "routing_output_valid": False,
            "g3_g6_executed": False,
        }
        self.gates: List[Dict[str, Any]] = []
        import torch

        self.summary["torch_version"] = torch.__version__
        self.summary["cuda_available"] = bool(torch.cuda.is_available())
        self.summary["gpu_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )

    def record_gate(self, name: str, ok: bool, detail: Any = None) -> None:
        entry: Dict[str, Any] = {"gate": name, "status": "PASS" if ok else "FAIL"}
        if detail is not None:
            entry["detail"] = detail
        self.gates.append(entry)
        self.summary["gates"] = self.gates
        print(f"[GATE] {name}={entry['status']}")

    def require(self, name: str, condition: bool, message: str) -> None:
        self.record_gate(name, condition, message)
        if not condition:
            self.summary["status"] = "FAIL"
            self.summary["first_failed_gate"] = name
            self.summary["failure_code"] = "AUTHORITY_GATE_FAILED"
            self.summary["error"] = message
            raise AuthorityGateFailure(name, message)

    # -- failure-evidence collection (best-effort, NEVER runs inference) -----

    def _existing_phase_evidence(self, phase: str) -> Dict[str, Any]:
        """Already-existing evidence only; never triggers a new inference."""
        key_map = {
            "G1": ("g1_inference_id", "g1_output_path", "g1_evidence_paths", "g1_status"),
            "ROUTING": ("routing_inference_id", "routing_output_path", "routing_evidence_paths", "routing_status"),
        }
        id_k, op_k, ep_k, st_k = key_map.get(phase, (None, None, None, None))
        if id_k and self.summary.get(ep_k):
            return {
                "inference_id": self.summary.get(id_k),
                "output_path": self.summary.get(op_k),
                "evidence_paths": self.summary.get(ep_k) or {},
                "status": self.summary.get(st_k),
                "telemetry_path": (self.summary.get(ep_k) or {}).get("telemetry"),
            }
        result = None
        if self.session is not None:
            try:
                result = self.session.results.get(phase)
            except Exception:
                result = None
        if result is None:
            return {}
        paths = result.evidence_paths or {}
        return {
            "inference_id": result.inference_id,
            "output_path": result.output_path,
            "evidence_paths": paths,
            "status": result.status,
            "telemetry_path": paths.get("telemetry"),
        }

    def collect_failure_evidence(self, exc: Exception) -> None:
        """Preserve rich machine-readable evidence on EVERY failure."""
        try:
            s = self.summary
            s["status"] = "FAIL"
            s["exception_type"] = type(exc).__name__
            s["exception_message"] = str(exc)
            if isinstance(exc, AuthorityGateFailure):
                s["first_failed_gate"] = exc.gate
                s["failure_code"] = s.get("failure_code") or "AUTHORITY_GATE_FAILED"
            else:
                if not s.get("first_failed_gate"):
                    s["first_failed_gate"] = "UNKNOWN"
                s["failure_code"] = s.get("failure_code") or "UNHANDLED"
            if not s.get("error"):
                s["error"] = str(exc)

            session = self.session
            s["session_id"] = getattr(session, "session_id", None)
            s["session_state"] = getattr(session, "state_name", None)
            s["model_load_count"] = getattr(session, "model_load_count", None)
            state = session.state if session is not None else None
            s["runtime_state_id"] = id(state) if state is not None else None
            s["transformer_id"] = (
                id(getattr(state, "transformer", None))
                if state is not None and getattr(state, "transformer", None) is not None
                else None
            )
            s["text_encoder_id"] = (
                id(getattr(state, "text_encoder", None))
                if state is not None and getattr(state, "text_encoder", None) is not None
                else None
            )
            s["vae_id"] = (
                id(getattr(state, "vae", None))
                if state is not None and getattr(state, "vae", None) is not None
                else None
            )

            for phase in ("G1", "ROUTING"):
                ev = self._existing_phase_evidence(phase)
                id_k, op_k, ep_k, st_k = {
                    "G1": ("g1_inference_id", "g1_output_path", "g1_evidence_paths", "g1_status"),
                    "ROUTING": ("routing_inference_id", "routing_output_path", "routing_evidence_paths", "routing_status"),
                }[phase]
                for key, ev_key in ((id_k, "inference_id"), (op_k, "output_path"), (ep_k, "evidence_paths"), (st_k, "status")):
                    if ev.get(ev_key) is not None and s.get(key) is None:
                        s[key] = ev[ev_key]
                if s.get(ep_k) is None:
                    s[ep_k] = ev.get("evidence_paths") or {}
        except Exception:
            pass

    def run(self) -> Dict[str, Any]:
        # === qualified gate order (runbook section 57) ===
        # 1) qualified runtime baseline integrity: FIRST gate, before anything else.
        baseline = verify_runtime_baseline(REPO)
        self.summary["RUNTIME_BASELINE_INTEGRITY"] = baseline["status"]
        self.summary["RUNTIME_BASELINE_SHA256"] = baseline.get("manifest_sha256")
        self.summary["RUNTIME_BASELINE_MANIFEST_ROWS"] = len(baseline.get("entries", []))
        self.summary["RUNTIME_BASELINE_CANDIDATE_FILES"] = sorted(
            e.get("path") for e in baseline.get("entries", [])
        )
        self.require(
            "RUNTIME_BASELINE_INTEGRITY",
            baseline["status"] == "PASS",
            str(baseline.get("errors") or "runtime baseline not self-consistent"),
        )
        print(
            "RUNTIME_BASELINE_INTEGRITY=PASS rows=%s sha256=%s"
            % (len(baseline.get("entries", [])), baseline.get("manifest_sha256"))
        )

        # 2) Freeze the SDPA attention backend AFTER the baseline gate, BEFORE
        #    the local bootstrap activation and any upstream mage_flow import.
        freeze_sdpa_env()

        # 3) Clean local bootstrap from a fresh site (offline, exact lock).
        #    prepare_bootstrap_environment creates a fresh local target,
        #    installs the exact bootstrap lock from the local wheelhouse into
        #    it, activates it, and returns all four local bootstrap gates plus
        #    the lock/wheelhouse integrity authoritative digests.
        try:
            bootstrap_env = prepare_bootstrap_environment(
                project_root=REPO,
            )
        except BootstrapEnvironmentError as exc:
            self.require("BOOTSTRAP_LOCAL_SITE_FRESH", False, str(exc))
            raise
        self.summary["bootstrap_site"] = bootstrap_env.get("target")
        self.summary["bootstrap_pins"] = bootstrap_env.get("pins")
        self.summary["bootstrap_requirements_sha256"] = bootstrap_env.get("lock_sha256")
        self.summary["bootstrap_wheelhouse_manifest_sha256"] = bootstrap_env.get("manifest_sha256")
        self.summary["BOOTSTRAP_REQUIREMENTS_LOCK"] = bootstrap_env.get("BOOTSTRAP_REQUIREMENTS_LOCK", "FAIL")
        self.summary["BOOTSTRAP_WHEELHOUSE_INTEGRITY"] = bootstrap_env.get("BOOTSTRAP_WHEELHOUSE_INTEGRITY", "FAIL")
        for gate, detail in (
            ("BOOTSTRAP_LOCAL_SITE_FRESH", "fresh local bootstrap target created"),
            ("BOOTSTRAP_LOCAL_PROVISION", "offline install of exact bootstrap lock"),
            ("BOOTSTRAP_LOCAL_VERSION_VERIFY", "post-provision exact-version verify"),
            ("BOOTSTRAP_LOCAL_ORIGIN_VERIFY", "bootstrap dependency local-origin verify"),
        ):
            value = bootstrap_env.get(gate, "FAIL")
            self.summary[gate] = value
            self.require(gate, value == "PASS", detail)
        self.require("BOOTSTRAP_REQUIREMENTS_LOCK", self.summary["BOOTSTRAP_REQUIREMENTS_LOCK"] == "PASS", str(bootstrap_env.get("pins")))
        self.require("BOOTSTRAP_WHEELHOUSE_INTEGRITY", self.summary["BOOTSTRAP_WHEELHOUSE_INTEGRITY"] == "PASS", "wheelhouse manifest/wheels unverified")
        print(
            "BOOTSTRAP_ENVIRONMENT=PASS site=%s lock_sha256=%s manifest_sha256=%s wheel_sha256=%s"
            % (
                bootstrap_env.get("target"),
                bootstrap_env.get("lock_sha256"),
                bootstrap_env.get("manifest_sha256"),
                bootstrap_env.get("wheel_sha256"),
            )
        )

        # --- R3 pinned upstream source bootstrap -----------------------------
        try:
            upstream = bootstrap_upstream_mage()
        except UpstreamMageBootstrapError as exc:
            self.require("UPSTREAM_BOOTSTRAP", False, f"{exc.failure_code}: {exc}")
            raise
        self.summary["upstream_bootstrap"] = {
            "source_root": upstream.source_root,
            "package_file": upstream.package_file,
            "upstream_repository": upstream.upstream_repository,
            "upstream_commit": upstream.upstream_commit,
            "package_version": upstream.package_version,
        }
        self.require("UPSTREAM_BOOTSTRAP", True, "pinned vendor mage_flow bootstrapped")
        print("UPSTREAM_BOOTSTRAP=PASS source=%s version=%s" % (
            upstream.source_root, upstream.package_version,
        ))

        import torch

        # --- P0 import/root gate -------------------------------------------
        p0_ok = all(
            (
                REPO.is_dir(),
                VaeInputBridge is not None,
                DualDeviceForwardAdapter is not None,
                callable(attach_dual_adapter),
                callable(gpu_session.attach_dual_adapter),
                callable(gpu_session.detach_dual_adapter),
            )
        )
        self.summary["p0_status"] = "PASS" if p0_ok else "FAIL"
        self.require("P0_FRESH_KERNEL_PREFLIGHT", p0_ok, "corrective modules/imports missing")
        print("P0_OK repo=%s" % REPO)

        # --- G0 hardware preflight -----------------------------------------
        inv0 = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                major, minor = torch.cuda.get_device_capability(i)
                inv0.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_vram_bytes": int(props.total_memory),
                        "compute_capability": f"{major}.{minor}",
                    }
                )
        self.summary["gpu_inventory"] = inv0
        hw_ok = (
            bool(torch.cuda.is_available())
            and int(torch.cuda.device_count()) == 2
            and all("T4" in str(d.get("name", "")) for d in inv0)
            and all(str(d.get("compute_capability")) == "7.5" for d in inv0)
        )
        self.summary["g0_hardware_status"] = "PASS" if hw_ok else "FAIL"
        self.require("G0_HARDWARE", hw_ok, "require two Tesla T4 (sm_75) devices")
        print("G0_HARDWARE=OK inventory=%s" % json.dumps(inv0))

        # --- Authority objects ----------------------------------------------
        contract = default_run_contract()
        source = pinned_source_authority()
        _require(source.mage_commit_sha == PINNED_MAGE_SHA, "SOURCE_PIN", "mage SHA mismatch")
        _require(source.model_revision == PINNED_MODEL_REVISION, "SOURCE_PIN", "model revision mismatch")
        expected_transformer_invocations = int(getattr(contract, "steps", 0))
        self.summary["expected_transformer_invocations"] = expected_transformer_invocations
        if expected_transformer_invocations <= 0:
            self.require("CONTRACT_STEPS", False, "contract.steps must be positive")
        self.require("CONTRACT_STEPS", True, "contract.steps positive")

        # --- config provenance (qualification sec 7): derived, NEVER literal True ------
        cfc = derive_cpu_fallback_config(contract)
        self.summary["cpu_fallback_config_observed_value"] = cfc["cpu_fallback_config_observed_value"]
        self.summary["cpu_fallback_forbidden_config"] = cfc["cpu_fallback_forbidden_config"]
        self.require(
            "CPU_FALLBACK_FORBIDDEN_CONFIG",
            bool(cfc["cpu_fallback_forbidden_config"]),
            "contract does not forbid CPU fallback",
        )

        evidence_root = self.evidence_root
        session = StageBSession(
            contract=contract,
            source_authority=source.to_dict(),
            evidence_root=str(evidence_root),
        )
        self.session = session
        self.summary["session_id"] = session.session_id
        self.summary["session_state"] = session.state_name
        self.summary["mage_commit"] = source.mage_commit_sha
        self.summary["model_id"] = source.model_identifier
        self.summary["model_revision"] = source.model_revision
        _require(
            session.model_load_count == 0 and session.state is None,
            "FRESH_SESSION",
            "session must be fresh",
        )
        print("SESSION_OK id=%s evidence=%s" % (session.session_id, evidence_root))

        # --- G0 via the session (project authority) -------------------------
        g0 = session.run_g0_preflight()
        self.summary["g0_session_status"] = g0.status
        self.summary["g0_inventory"] = (g0.details or {}).get("inventory")
        self.require("G0_SESSION", g0.status == "PASS", "G0 session preflight failed")
        combined_g0 = derive_p0_g0_summary(
            self.summary["p0_status"],
            self.summary["g0_hardware_status"],
            self.summary["g0_session_status"],
        )
        self.summary["g0_status"] = combined_g0["g0_status"]
        self.require(
            "G0",
            combined_g0["g0_status"] == "PASS",
            "G0 (hardware + session) not PASS",
        )
        print("G0_SESSION=PASS")

        # --- Local model path (authority prefers the Kaggle attachment) -----
        slug = contract.model.split("/")[-1]
        candidates = [
            EXPECTED_LOCAL_MODEL,
            f"/kaggle/input/{slug}/pytorch/default/1",
            f"/kaggle/input/{slug}",
        ]
        resolver = ModelPathResolver(candidates=candidates, required_rel_files=REQUIRED_MODEL_FILES)
        resolution = resolver.resolve(fallback_id=None)
        model_path = resolution["selected_path"]
        source_kind = resolution["model_source"]
        self.summary["model_path"] = model_path
        self.require(
            "MODEL_PATH",
            bool(model_path) and source_kind == "LOCAL_ATTACHMENT",
            f"local Kaggle model required; got {source_kind}",
        )

        # --- L0 dual-T4 spread load (exactly once) --------------------------
        state = session.load_once(model_path)
        l0 = session.results.get("L0")
        memory_plan = getattr(state, "memory_plan", None) or {}
        placement = getattr(state, "placement_validation", None) or {}
        l0_facts = (l0.observed_facts if l0 is not None else {}) or {}

        self.summary["model_load_count"] = session.model_load_count
        self.summary["session_state"] = session.state_name
        self.summary["runtime_state_id"] = id(state)
        self.summary["transformer_id"] = id(state.transformer)
        self.summary["text_encoder_id"] = id(state.text_encoder)
        self.summary["vae_id"] = id(state.vae)
        self.summary["memory_plan"] = memory_plan
        self.summary["split_block"] = memory_plan.get("split_block")
        self.summary["l0_status"] = l0.status if l0 is not None else "NOT_RUN"

        blocks_live, _block_attr = discover_transformer_blocks(state.transformer)
        persist_live_block_count(self.summary, blocks_live)
        num_blocks_live = int(self.summary["num_blocks"])
        plan_blocks = {
            int(k): str(v)
            for k, v in (memory_plan.get("block_devices") or {}).items()
        }

        # Live-derived authority endpoints (never hard-coded, never reconstructed).
        actual_vae_device = derive_vae_device(state.vae)
        self.summary["vae_device_expected"] = actual_vae_device
        split_block = memory_plan.get("split_block")
        boundary = derive_split_boundary_endpoints(memory_plan.get("block_devices") or {}, split_block)
        self.summary["cross_transfer_boundary_from_block"] = boundary["from_block"]
        self.summary["cross_transfer_boundary_to_block"] = boundary["to_block"]
        self.summary["cross_transfer_expected_from"] = boundary["expected_from"]
        self.summary["cross_transfer_expected_to"] = boundary["expected_to"]
        self.require(
            "CROSS_TRANSFER_BOUNDARY",
            boundary["validated"],
            boundary["reason"] or "split-boundary not validated",
        )
        cross_from = boundary["expected_from"]
        cross_to = boundary["expected_to"]

        # Plan-fallback latent anchor / post-head expected for the G1
        # prerequisite adjudication (live capture happens after ROUTING).
        plan_latent = derive_latent_anchor_device({}, memory_plan)
        latent_anchor_g1 = plan_latent["anchor_device"]
        self.summary["latent_anchor_plan_fallback"] = plan_latent
        plan_post = derive_post_head_device_expected(memory_plan, None)
        post_head_expected_g1 = plan_post["expected"]
        self.summary["post_head_plan_fallback"] = plan_post

        spread = len(set(plan_blocks.values())) == 2
        self.require(
            "L0",
            (
                (l0 is not None and l0.status == "PASS")
                and session.model_load_count == 1
                and memory_plan.get("status") == "PASS"
                and (memory_plan.get("single_device") or {}).get("status") == "FAIL"
                and bool(l0_facts.get("NO_FULL_MODEL_CUDA0_MATERIALIZATION"))
                and bool(l0_facts.get("SPREAD_LOAD_PLAN_PASS"))
                and bool(l0_facts.get("SPREAD_LOAD_OBSERVED_PLACEMENT_PASS"))
                and spread
                and memory_plan.get("split_policy") == "memory_aware"
            ),
            "L0 dual-T4 spread load acceptance failed",
        )
        print(
            "L0=PASS split_block=%s num_blocks=%s no_full_cuda0=%s"
            % (
                split_block,
                num_blocks_live,
                bool(l0_facts.get("NO_FULL_MODEL_CUDA0_MATERIALIZATION")),
            )
        )
        _write_json(
            Path(evidence_root) / "l0-adjudication.json",
            {
                "l0_status": l0.status,
                "model_load_count": session.model_load_count,
                "memory_plan_status": memory_plan.get("status"),
                "single_device_status": (memory_plan.get("single_device") or {}).get("status"),
                "no_full_model_cuda0_materialization": bool(
                    l0_facts.get("NO_FULL_MODEL_CUDA0_MATERIALIZATION")
                ),
                "split_block": split_block,
                "split_policy": memory_plan.get("split_policy"),
                "num_blocks_live": num_blocks_live,
                "plan_blocks": plan_blocks,
                "placement_status": placement.get("status"),
                "vae_device_expected": actual_vae_device,
                "cross_transfer_boundary_from_block": boundary["from_block"],
                "cross_transfer_boundary_to_block": boundary["to_block"],
                "cross_transfer_expected_from": cross_from,
                "cross_transfer_expected_to": cross_to,
            },
        )

        # --- SDPA gate ------------------------------------------------------
        sdpa = assert_sdpa_frozen()
        self.summary["sdpa"] = sdpa
        self.require(
            "SDPA",
            sdpa.get("status") == "PASS"
            and str(os.environ.get("VF_HF_ATTN_IMPL", "")).lower() == "sdpa",
            f"SDPA not provable: {sdpa}",
        )
        sdpa_backend = (sdpa.get("backend") or {}).get("backend")
        print("SDPA=PASS backend=%s" % sdpa_backend)

        # --- Identity capture (pre-G1) --------------------------------------
        identity_before = {
            "runtime_state": id(session.state),
            "model": id(session.state.model),
            "transformer": id(session.state.transformer),
            "text_encoder": id(session.state.text_encoder),
            "vae": id(session.state.vae),
        }
        param_sample_before = {
            "text_encoder": _first_param_ids(session.state.text_encoder),
            "vae": _first_param_ids(session.state.vae),
            "transformer_block0": _first_param_ids(blocks_live[0]),
            "transformer_img_in": _first_param_ids(
                getattr(session.state.transformer, "img_in", None)
            ),
        }

        # ==================================================================
        # G1: exactly one prerequisite dual-T4 mixed inference
        # ==================================================================
        g1 = session.run_g1_baseline()
        self.summary["g1_inference_id"] = g1.inference_id
        self.summary["g1_output_path"] = g1.output_path
        self.summary["g1_evidence_paths"] = g1.evidence_paths or {}
        self.summary["g1_status"] = g1.status
        self.summary["session_state"] = session.state_name
        self.require(
            "G1_PREREQUISITE",
            g1.status == "PASS"
            and g1.exit_code == 0
            and g1.output_path is not None
            and g1.observed_facts is not None,
            f"G1 prerequisite failed: {g1.failure_code}",
        )
        print("G1_OK inference_id=%s output=%s" % (g1.inference_id, g1.output_path))

        # --- Post-G1 identity check ----------------------------------------
        identity_after_g1 = {
            "runtime_state": id(session.state),
            "model": id(session.state.model),
            "transformer": id(session.state.transformer),
            "text_encoder": id(session.state.text_encoder),
            "vae": id(session.state.vae),
        }
        param_sample_after_g1 = {
            "text_encoder": _first_param_ids(session.state.text_encoder),
            "vae": _first_param_ids(session.state.vae),
            "transformer_block0": _first_param_ids(blocks_live[0]),
            "transformer_img_in": _first_param_ids(
                getattr(session.state.transformer, "img_in", None)
            ),
        }
        g1_identity_preserved = (
            identity_after_g1 == identity_before
            and param_sample_after_g1 == param_sample_before
            and session.model_load_count == 1
        )
        self.require(
            "IDENTITY_PRESERVED_G1",
            g1_identity_preserved,
            "post-G1 object identity drift",
        )
        print("IDENTITY_G1=PASS model_load_count=%s" % session.model_load_count)

        # --- G1 routing telemetry adjudication -------------------------------
        g1_telemetry_path = Path(g1.evidence_paths.get("telemetry") or "")
        _require(g1_telemetry_path.is_file(), "G1_TELEMETRY", f"missing: {g1_telemetry_path}")
        g1_telemetry = parse_telemetry(str(g1_telemetry_path))

        g1_blocked = _records_for(g1_telemetry, session.session_id, g1.inference_id, "block_forward")
        g1_cross_events = _records_for(g1_telemetry, session.session_id, g1.inference_id, "cross_device_transfer")
        g1_return_events = _records_for(g1_telemetry, session.session_id, g1.inference_id, "transformer_output_return_transfer")
        g1_vae_events = _records_for(g1_telemetry, session.session_id, g1.inference_id, "vae_input_transfer")

        g1_observed_blocks: Dict[int, str] = {}
        for ev in g1_blocked:
            g1_observed_blocks[int(ev.get("block"))] = str(ev.get("device"))
        g1_routing = adjudicate_block_routing(g1_observed_blocks, plan_blocks, num_blocks_live)
        self.require("G1_BLOCK_ROUTING", g1_routing["pass"], "G1 block routing mismatch")
        g1_cross_adj = adjudicate_transfer_cardinality(
            g1_cross_events,
            from_device=cross_from,
            to_device=cross_to,
            expected_count=int(contract.steps),
        )
        self.require(
            "G1_CROSS_TRANSFER",
            g1_cross_adj["pass"],
            "G1 cross_device_transfer cardinality invalid",
        )
        g1_post_head_adj = adjudicate_post_head_device(
            g1_return_events,
            expected_device=post_head_expected_g1,
            expected_count=int(contract.steps),
            anchor_device=latent_anchor_g1,
        )
        self.require(
            "G1_TRANSFORMER_RETURN",
            g1_post_head_adj["pass"],
            "G1 transformer return/post-head witness invalid",
        )
        g1_vae_adj = adjudicate_transfer_cardinality(
            g1_vae_events,
            from_device=latent_anchor_g1,
            to_device=actual_vae_device,
            expected_count=1,
        )
        self.require(
            "G1_VAE_INPUT_TRANSFER",
            g1_vae_adj["pass"],
            "G1 VAE input transfer invalid",
        )

        g1_integrity = phase_block_integrity(g1.observed_facts)
        self.require(
            "G1_ACCEPTANCE",
            g1_integrity["block_order_valid"]
            and g1_integrity["gpu0_participation"]
            and g1_integrity["gpu1_participation"],
            "G1 acceptance failed",
        )
        print("G1_ADJUDICATION=PASS")

        # ==================================================================
        # ROUTING: routing authority
        # ==================================================================
        routing = session.run_routing_validation()
        self.summary["routing_inference_id"] = routing.inference_id
        self.summary["routing_output_path"] = routing.output_path
        self.summary["routing_evidence_paths"] = routing.evidence_paths or {}
        self.summary["routing_status"] = routing.status
        self.summary["session_state"] = session.state_name
        self.require(
            "ROUTING_ROUTING_AUTHORITY",
            routing.status == "PASS"
            and routing.exit_code == 0
            and routing.output_path is not None
            and routing.observed_facts is not None,
            f"ROUTING routing authority failed: {routing.failure_code}",
        )
        print("ROUTING_OK inference_id=%s output=%s" % (routing.inference_id, routing.output_path))

        # --- qualification sec 17.1/17.3/17.7/17.8: capture LIVE authority placements --
        live_placements = collect_live_authority_placements(session.state)
        self.summary["live_authority_placements"] = live_placements
        live_adj = adjudicate_live_authority_placements(live_placements)
        self.summary["live_placement_adjudication"] = live_adj
        self.require(
            "LIVE_PLACEMENT",
            live_adj["pass"],
            "live authority placement not CUDA-only",
        )
        pv_observed = collect_observed_placement_validation(placement)
        self.summary["placement_validation_observed"] = pv_observed
        self.require(
            "PLACEMENT_VALIDATION",
            pv_observed["pass"],
            "observed placement-validation not PASS/CUDA",
        )
        live_vs_plan = derive_live_vs_plan_placement_match(live_placements, memory_plan)
        self.summary["live_vs_plan_placement_match"] = live_vs_plan
        self.require(
            "LIVE_VS_PLAN",
            live_vs_plan["match"],
            "live placement disagrees with the L0 plan",
        )
        latent = derive_latent_anchor_device(live_placements, memory_plan)
        latent_anchor = latent["anchor_device"]
        self.summary["latent_anchor_device_expected"] = latent_anchor
        self.summary["latent_anchor_device_source"] = latent["source"]
        self.require(
            "LATENT_ANCHOR_DEVICE",
            not latent["unproven"],
            latent["reason"] or "latent anchor unproven",
        )
        post_derivation = derive_post_head_device_expected(
            memory_plan, live_placements.get("transformer_post")
        )
        post_head_expected = post_derivation["expected"]
        self.summary["post_head_expected_cross_check"] = post_derivation
        self.summary["post_head_device_expected"] = post_head_expected
        self.require(
            "POST_HEAD_EXPECTED",
            post_derivation["valid"],
            post_derivation["reason"] or "post/head expected invalid",
        )
        print(
            "LIVE_PLACEMENT=PASS latent_anchor=%s post_head_expected=%s"
            % (latent_anchor, post_head_expected)
        )

        # --- Post-ROUTING identity check ----------------------------------------
        identity_after_routing = {
            "runtime_state": id(session.state),
            "model": id(session.state.model),
            "transformer": id(session.state.transformer),
            "text_encoder": id(session.state.text_encoder),
            "vae": id(session.state.vae),
        }
        param_sample_after_routing = {
            "text_encoder": _first_param_ids(session.state.text_encoder),
            "vae": _first_param_ids(session.state.vae),
            "transformer_block0": _first_param_ids(blocks_live[0]),
            "transformer_img_in": _first_param_ids(
                getattr(session.state.transformer, "img_in", None)
            ),
        }
        self.summary["runtime_state_id_stable"] = (
            identity_after_routing["runtime_state"] == identity_before["runtime_state"]
            and session.model_load_count == 1
        )
        self.summary["transformer_id_stable"] = (
            identity_after_routing["transformer"] == identity_before["transformer"]
            and param_sample_after_routing["transformer_block0"] == param_sample_before["transformer_block0"]
        )
        self.summary["text_encoder_id_stable"] = (
            identity_after_routing["text_encoder"] == identity_before["text_encoder"]
            and param_sample_after_routing["text_encoder"] == param_sample_before["text_encoder"]
        )
        self.summary["vae_id_stable"] = (
            identity_after_routing["vae"] == identity_before["vae"]
            and param_sample_after_routing["vae"] == param_sample_before["vae"]
        )
        identity_preserved_routing = (
            self.summary["runtime_state_id_stable"]
            and self.summary["transformer_id_stable"]
            and self.summary["text_encoder_id_stable"]
            and self.summary["vae_id_stable"]
        )
        self.require("IDENTITY_PRESERVED_ROUTING", identity_preserved_routing, "post-ROUTING identity drift")
        print("IDENTITY_ROUTING=PASS")

        # --- ROUTING routing telemetry adjudication -------------------------------
        routing_telemetry_path = Path(routing.evidence_paths.get("telemetry") or "")
        _require(routing_telemetry_path.is_file(), "ROUTING_TELEMETRY", f"missing: {routing_telemetry_path}")
        routing_telemetry = parse_telemetry(str(routing_telemetry_path))

        routing_blocked = _records_for(routing_telemetry, session.session_id, routing.inference_id, "block_forward")
        routing_cross_events = _records_for(routing_telemetry, session.session_id, routing.inference_id, "cross_device_transfer")
        routing_return_events = _records_for(routing_telemetry, session.session_id, routing.inference_id, "transformer_output_return_transfer")
        routing_vae_events = _records_for(routing_telemetry, session.session_id, routing.inference_id, "vae_input_transfer")

        # GPU participation
        routing_gpu0 = any(str(e.get("device")) == "cuda:0" for e in routing_blocked)
        routing_gpu1 = any(str(e.get("device")) == "cuda:1" for e in routing_blocked)
        self.summary["routing_gpu0_participation"] = bool(routing_gpu0)
        self.summary["routing_gpu1_participation"] = bool(routing_gpu1)
        self.require("ROUTING_GPU0_PARTICIPATION", routing_gpu0, "ROUTING GPU0 absent")
        self.require("ROUTING_GPU1_PARTICIPATION", routing_gpu1, "ROUTING GPU1 absent")

        # Block routing (BLOCK_ROUTING gate; never feeds TRANSFER/POST_HEAD)
        routing_observed_blocks: Dict[int, str] = {}
        for ev in routing_blocked:
            routing_observed_blocks[int(ev.get("block"))] = str(ev.get("device"))
        routing_routing = adjudicate_block_routing(routing_observed_blocks, plan_blocks, num_blocks_live)
        self.summary["routing_block_adjudication"] = routing_routing
        self.require("ROUTING_BLOCK_ROUTING", routing_routing["pass"], "ROUTING block routing mismatch")

        # Cross-device transfer 0->1 (split-boundary endpoints, strict count)
        cross_adj = adjudicate_transfer_cardinality(
            routing_cross_events,
            from_device=cross_from,
            to_device=cross_to,
            expected_count=int(contract.steps),
        )
        self.summary["routing_transfer_0_to_1_event_count"] = cross_adj["event_count"]
        self.summary["routing_transfer_0_to_1_expected_count"] = cross_adj["expected_count"]
        self.summary["routing_transfer_0_to_1_valid"] = cross_adj["pass"]
        self.require(
            "ROUTING_TRANSFER_0_TO_1",
            cross_adj["pass"],
            "ROUTING cross_device_transfer cardinality invalid",
        )

        # Transformer return via post/head witness (strict cardinality)
        return_adj = adjudicate_transfer_cardinality(
            routing_return_events,
            from_device=post_head_expected,
            to_device=latent_anchor,
            expected_count=int(contract.steps),
        )
        self.summary["routing_transformer_return_event_count"] = return_adj["event_count"]
        self.summary["routing_transformer_return_expected_count"] = return_adj["expected_count"]
        self.summary["routing_transformer_return_valid"] = return_adj["pass"]
        self.require(
            "ROUTING_TRANSFORMER_RETURN_1_TO_0",
            return_adj["pass"],
            "ROUTING transformer return transfer invalid",
        )

        # Post/head device proof (expected cross-check + audited witness)
        post_head_adj = adjudicate_post_head_device(
            routing_return_events,
            expected_device=post_head_expected,
            expected_count=int(contract.steps),
            anchor_device=latent_anchor,
        )
        observed_form = (
            post_head_expected
            if post_head_adj["pass"]
            else "MISMATCH(%s)"
            % ("|".join(post_head_adj["observed_from_devices"]) or "NO_RETURN_EVENTS")
        )
        self.summary["post_head_device_observed"] = observed_form
        self.summary["post_head_device_valid"] = post_head_adj["pass"]
        self.require(
            "ROUTING_POST_HEAD_DEVICE",
            post_head_adj["pass"],
            "ROUTING post/head device witness invalid",
        )

        # VAE input transfer (strict cardinality == 1)
        vae_adj = adjudicate_transfer_cardinality(
            routing_vae_events,
            from_device=latent_anchor,
            to_device=actual_vae_device,
            expected_count=1,
        )
        self.summary["routing_vae_input_transfer_event_count"] = vae_adj["event_count"]
        self.summary["routing_vae_input_transfer_expected_count"] = vae_adj["expected_count"]
        self.summary["routing_vae_input_transfer_valid"] = vae_adj["pass"]
        self.require(
            "ROUTING_VAE_INPUT_TRANSFER",
            vae_adj["pass"],
            f"ROUTING VAE input transfer to {actual_vae_device} invalid",
        )

        # qualification sec 6: observed NO-CPU-FALLBACK over live + observed + telemetry
        scoped_records: List[Dict[str, Any]] = []
        for ev in routing_blocked:
            scoped_records.append(ev)
        scoped_records.extend(routing_cross_events)
        scoped_records.extend(routing_return_events)
        scoped_records.extend(routing_vae_events)
        ncf = adjudicate_no_cpu_fallback(live_placements, pv_observed, scoped_records)
        self.summary["no_cpu_fallback_observed"] = ncf["no_cpu_fallback_observed"]
        self.summary["no_cpu_fallback_adjudication"] = ncf["no_cpu_fallback_adjudication"]
        self.require(
            "NO_CPU_FALLBACK_OBSERVED",
            ncf["no_cpu_fallback_observed"],
            "CPU fallback observed in live placement, observed validation, or telemetry",
        )

        # ROUTING phase block integrity (per-invocation multi-step)
        routing_integrity = phase_block_integrity(routing.observed_facts)
        self.summary["routing_block_order_valid"] = routing_integrity["block_order_valid"]
        self.summary["routing_no_skipped_blocks"] = routing_integrity["no_skipped_blocks"]
        self.summary["routing_no_duplicated_blocks"] = routing_integrity["no_duplicated_blocks"]
        self.summary["routing_cross_inference_clean"] = routing_integrity["cross_inference_clean"]
        self.summary["routing_block_integrity"] = routing_integrity
        self.require("ROUTING_EXACT_BLOCK_COVERAGE", routing_integrity["block_order_valid"], "ROUTING block order invalid")
        self.require("ROUTING_NO_SKIPPED_BLOCKS", routing_integrity["no_skipped_blocks"], "ROUTING skipped blocks")
        self.require("ROUTING_NO_DUPLICATED_BLOCKS", routing_integrity["no_duplicated_blocks"], "ROUTING duplicated blocks")
        self.require("ROUTING_CROSS_INFERENCE_CLEAN", routing_integrity["cross_inference_clean"], "ROUTING cross-inference contamination")
        print("ROUTING_ROUTING=PASS")

        # --- ROUTING output acceptance -------------------------------------------
        routing_image = validate_image(routing.output_path)
        routing_no_nan = bool((routing.observed_facts or {}).get("NO_NAN"))
        routing_no_inf = bool((routing.observed_facts or {}).get("NO_INF"))
        output_acceptance = build_output_acceptance(
            image_valid=bool(routing_image.get("valid")),
            no_nan=routing_no_nan,
            no_inf=routing_no_inf,
            path_exists=bool(routing.output_path and Path(routing.output_path).is_file()),
        )
        self.summary["routing_image_validation"] = routing_image
        self.summary["routing_no_nan"] = routing_no_nan
        self.summary["routing_no_inf"] = routing_no_inf
        self.summary["routing_output_valid"] = output_acceptance["valid"]
        self.require(
            "ROUTING_OUTPUT_VALID",
            bool(output_acceptance["valid"]),
            f"ROUTING output invalid: {routing_image}",
        )

        # --- ROUTING phase acceptance (whole trajectory) ---------------------------
        routing_acceptance_ok = (routing.acceptance or {}).get("status") == "PASS"
        self.require("ROUTING_ACCEPTANCE", routing_acceptance_ok, "ROUTING phase acceptance not PASS")

        # --- Final adjudication -----------------------------------------------
        self.summary["status"] = "PASS"
        self.summary["g3_g6_executed"] = False
        session.save_status(str(Path(evidence_root) / "session-status.json"))
        _write_summary_best_effort(self, self.summary, evidence_root)
        _print_human_summary(self.summary)
        print("FINAL_VERDICT=PASS")
        return self.summary


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def _print_human_summary(s: Dict[str, Any]) -> None:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    routing_ok = s.get("status") == "PASS" and s.get("routing_status") == "PASS"
    verdict = "PASS" if routing_ok else "FAIL"
    print("=" * 60)
    print("=== ROUTING ROUTING AUTHORITY qualified SUMMARY (historical/provenance-corrective preserved; multistep block-integrity corrective) ===")
    print(f"SESSION_ID={s.get('session_id')}")
    print(f"SESSION_STATE={s.get('session_state')}")
    print(f"EVIDENCE_ROOT={s.get('evidence_root')}")
    print(f"RUNTIME_BASELINE_INTEGRITY={_human_passthrough(s, 'RUNTIME_BASELINE_INTEGRITY')}")
    print(f"BOOTSTRAP_REQUIREMENTS_LOCK={_human_passthrough(s, 'BOOTSTRAP_REQUIREMENTS_LOCK')}")
    print(f"BOOTSTRAP_WHEELHOUSE_INTEGRITY={_human_passthrough(s, 'BOOTSTRAP_WHEELHOUSE_INTEGRITY')}")
    print(f"BOOTSTRAP_LOCAL_SITE_FRESH={_human_passthrough(s, 'BOOTSTRAP_LOCAL_SITE_FRESH')}")
    print(f"BOOTSTRAP_LOCAL_PROVISION={_human_passthrough(s, 'BOOTSTRAP_LOCAL_PROVISION')}")
    print(f"BOOTSTRAP_LOCAL_VERSION_VERIFY={_human_passthrough(s, 'BOOTSTRAP_LOCAL_VERSION_VERIFY')}")
    print(f"BOOTSTRAP_LOCAL_ORIGIN_VERIFY={_human_passthrough(s, 'BOOTSTRAP_LOCAL_ORIGIN_VERIFY')}")
    print(f"UPSTREAM_BOOTSTRAP={mark((s.get('upstream_bootstrap') or {}).get('source_root') is not None)}")
    print(f"P0={_human_passthrough(s, 'p0_status')}")
    print(f"G0={mark(s.get('g0_status') == 'PASS')}")
    print(f"L0={mark(s.get('l0_status') == 'PASS')}")
    print(f"SDPA={mark((s.get('sdpa') or {}).get('status') == 'PASS')}")
    print(f"MODEL_LOAD_COUNT={s.get('model_load_count')}")
    print(f"SPLIT_BLOCK={s.get('split_block')}")
    print(f"NUM_BLOCKS={s.get('num_blocks')}")
    print(f"G1={mark(s.get('g1_status') == 'PASS')}")
    print(f"G1_INFERENCE_ID={s.get('g1_inference_id')}")
    print(f"ROUTING={mark(routing_ok)}")
    print(f"ROUTING_INFERENCE_ID={s.get('routing_inference_id')}")
    print(f"RUNTIME_STATE_ID_STABLE={human_fact(s, 'runtime_state_id_stable')}")
    print(f"TRANSFORMER_ID_STABLE={human_fact(s, 'transformer_id_stable')}")
    print(f"TEXT_ENCODER_ID_STABLE={human_fact(s, 'text_encoder_id_stable')}")
    print(f"VAE_ID_STABLE={human_fact(s, 'vae_id_stable')}")
    print(f"EXACT_BLOCK_COVERAGE={human_fact(s, 'routing_block_order_valid')}")
    print(f"BLOCK_ORDER={human_fact(s, 'routing_block_order_valid')}")
    print(f"NO_SKIP={human_fact(s, 'routing_no_skipped_blocks')}")
    print(f"NO_DUPLICATE_WITHIN_INVOCATION={human_fact(s, 'routing_no_duplicated_blocks')}")
    print(f"GPU0_PARTICIPATION={human_fact(s, 'routing_gpu0_participation')}")
    print(f"GPU1_PARTICIPATION={human_fact(s, 'routing_gpu1_participation')}")
    print(f"EXPECTED_TRANSFORMER_INVOCATIONS={s.get('expected_transformer_invocations')}")
    print(f"TRANSFER_0_TO_1_COUNT={s.get('routing_transfer_0_to_1_event_count')}/{s.get('routing_transfer_0_to_1_expected_count')}")
    print(f"TRANSFER_0_TO_1={human_fact(s, 'routing_transfer_0_to_1_valid')}")
    print(f"TRANSFORMER_RETURN_1_TO_0_COUNT={s.get('routing_transformer_return_event_count')}/{s.get('routing_transformer_return_expected_count')}")
    print(f"TRANSFORMER_RETURN_1_TO_0={human_fact(s, 'routing_transformer_return_valid')}")
    print(f"POST_HEAD_DEVICE_EXPECTED={s.get('post_head_device_expected')}")
    print(f"POST_HEAD_DEVICE_OBSERVED={s.get('post_head_device_observed')}")
    print(f"POST_HEAD_DEVICE={human_fact(s, 'post_head_device_valid')}")
    print(f"VAE_INPUT_TRANSFER_COUNT={s.get('routing_vae_input_transfer_event_count')}/{s.get('routing_vae_input_transfer_expected_count')}")
    print(f"VAE_INPUT_TRANSFER={human_fact(s, 'routing_vae_input_transfer_valid')}")
    print(f"CPU_FALLBACK_FORBIDDEN_CONFIG={human_fact(s, 'cpu_fallback_forbidden_config')}")
    print(f"NO_CPU_FALLBACK_OBSERVED={human_fact(s, 'no_cpu_fallback_observed')}")
    print(f"NO_NAN={human_fact(s, 'routing_no_nan')}")
    print(f"NO_INF={human_fact(s, 'routing_no_inf')}")
    print(f"OUTPUT_VALID={human_fact(s, 'routing_output_valid')}")
    print(f"G3_G6_EXECUTED={human_g3_g6(s)}")
    print(f"FINAL_VERDICT={verdict}")
    print("=" * 60)


def _write_summary_best_effort(
    driver: DualT4QualificationDriver, summary: Dict[str, Any], evidence_root: Path
) -> Optional[Path]:
    summary_path = Path(evidence_root) / "corrective-summary.json"
    summary["corrective_summary_path"] = str(summary_path)
    try:
        _write_json(summary_path, summary)
        return summary_path
    except Exception:
        if summary.get("status") == "PASS":
            raise
        return None


def _run_driver() -> int:
    evidence_root = (
        REPO
        / "evidence"
        / "gpu"
        / ("routing-routing-authority-qualified-fresh-%s-%s" % (time.strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8]))
    )
    driver = DualT4QualificationDriver(evidence_root)
    summary = None
    try:
        summary = driver.run()
    except AuthorityGateFailure as exc:
        driver.collect_failure_evidence(exc)
    except Exception as exc:
        driver.collect_failure_evidence(exc)
    if summary is None:
        summary = driver.summary
    _write_summary_best_effort(driver, summary, evidence_root)
    _print_human_summary(summary)
    if summary.get("status") != "PASS":
        print(f"FINAL_VERDICT=FAIL first_failed_gate={summary.get('first_failed_gate')}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run_driver())