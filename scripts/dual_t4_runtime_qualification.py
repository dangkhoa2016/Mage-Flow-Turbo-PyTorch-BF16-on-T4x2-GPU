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
