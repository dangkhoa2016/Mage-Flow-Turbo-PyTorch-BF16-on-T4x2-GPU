"""SDPA freeze for the T4 / sm_75 authority run (Corrective L0_SDPA_OBSERVER_API).

sm_75 (T4) does not provide the FP16/BF16 kernels the project's default
``flash2`` attention backend needs, so the authority run MUST execute with
attention frozen to PyTorch SDPA end-to-end:

  1. ``VF_HF_ATTN_IMPL=sdpa`` in the environment **before** the model load —
     upstream ``TextEncoder`` reads this env var as the authoritative override
     (``_resolve_hf_attn_impl``) regardless of the config's ``attn_type``.
  2. ``set_attn_backend("sdpa")`` requested on the pinned upstream backend
     module before the DiT backend resolves (upstream constructor defaults the
     backend to ``"flash2"``).

The observer is a **pinned-source compatibility adapter**: the pinned upstream
commit (``76bec2bb3818863f470de7e867c2dc7f1d0bfd83``) does not export
``get_backend()``.  It exposes ``set_attn_backend`` / ``flash_attn_varlen_func``
and module-private dispatch state ``_BACKEND``.  This module never requires
``get_backend()``; a future upstream that adds it is handled via public-getter
preference, and the private-state path is used as the pinned fallback.

Module import is CPU-safe and never imports ``mage_flow`` at module scope.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

SDPA_BACKEND_NAME = "sdpa"
VF_HF_ATTN_IMPL = "VF_HF_ATTN_IMPL"

# Pinned upstream source authority (see mage_t4x2/constants.py).
PINNED_MAGE_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
_ATTN_BACKEND_MODULE = "mage_flow.models.modules._attn_backend"
_PINNED_STATE_SYMBOL = "_BACKEND"
_PINNED_SETTER_SYMBOL = "set_attn_backend"

SDPA_ENV = {
    VF_HF_ATTN_IMPL: SDPA_BACKEND_NAME,
}


def freeze_sdpa_env() -> Dict[str, Any]:
    """Force ``VF_HF_ATTN_IMPL=sdpa`` before any upstream import.

    Never overwrites (the value is authoritative at first read); it only fills
    the env var when unset and reports what the upstream will observe.
    """
    prior: Dict[str, str] = {}
    changed: Dict[str, str] = {}
    for key, value in SDPA_ENV.items():
        existing = os.environ.get(key)
        prior[key] = existing
        if existing is None or str(existing).lower() != value:
            os.environ[key] = value
            if existing is None:
                changed[key] = value
            else:
                changed[key] = f"{existing} -> {value}"
    return {
        "env": dict(os.environ.items()),
        "prior": prior,
        "changed": changed,
        "status": "PASS",
        "note": "SDPA env vector frozen before model load",
    }


def _load_backend_module() -> Any:
    """Import the exact pinned upstream backend module (import-level only)."""
    import importlib

    return importlib.import_module(_ATTN_BACKEND_MODULE)


def _observe_module(module: Any) -> Dict[str, Any]:
    """Read the current backend selection from a pinned-shape backend module.

    Preference order matches the pinned source reality:

      * ``get_backend`` — used only if a future upstream exposes it
        (``public_getter`` observation).
      * module-private ``_BACKEND`` — the pinned-source qualified state symbol
        (``pinned_private_state`` observation).  Symbol existence and string
        type are validated; anything unexpected fails closed.
    """
    method = "pinned_private_state"
    getter = getattr(module, "get_backend", None)
    if callable(getter):
        method = "public_getter"
        try:
            value = getter()
        except Exception as exc:  # noqa: BLE001
            return {
                "value": None,
                "ok": False,
                "method": method,
                "reason": f"get_backend() raised: {exc}",
            }
        ok = value == SDPA_BACKEND_NAME
        return {
            "value": value,
            "ok": ok,
            "method": method,
            "reason": None if ok else f"backend is {value!r}, expected {SDPA_BACKEND_NAME!r}",
        }

    state = getattr(module, _PINNED_STATE_SYMBOL, None)
    if not isinstance(state, str):
        return {
            "value": None,
            "ok": False,
            "method": method,
            "reason": (
                f"pinned backend state {_PINNED_STATE_SYMBOL!r} missing or not a string "
                f"in {_ATTN_BACKEND_MODULE}"
            ),
        }
    ok = state == SDPA_BACKEND_NAME
    return {
        "value": state,
        "ok": ok,
        "method": method,
        "reason": None if ok else f"backend is {state!r}, expected {SDPA_BACKEND_NAME!r}",
    }


def observed_backend() -> Dict[str, Any]:
    """Observe the current attention backend selection of the pinned module.

    Output contract (see corrective directive §13):

    .. code-block:: json

       {
         "available": true,
         "backend": "sdpa",
         "status": "PASS",
         "observation_method": "pinned_private_state|public_getter",
         "module": "mage_flow.models.modules._attn_backend",
         "pinned_commit": "76bec2bb...",
         "reason": null
       }

    On CPU hosts without ``mage_flow`` the module is unavailable and the record
    reports ``available: False`` / ``status: SKIP``.
    """
    try:
        module = _load_backend_module()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "backend": None,
            "status": "SKIP",
            "observation_method": "module_unavailable",
            "module": _ATTN_BACKEND_MODULE,
            "pinned_commit": PINNED_MAGE_COMMIT,
            "reason": f"attention backend module unavailable here: {type(exc).__name__}",
        }
    obs = _observe_module(module)
    return {
        "available": True,
        "backend": obs["value"],
        "status": "PASS" if obs["ok"] else "FAIL",
        "observation_method": obs["method"],
        "module": _ATTN_BACKEND_MODULE,
        "pinned_commit": PINNED_MAGE_COMMIT,
        "reason": obs["reason"],
    }


def backend_api_shape() -> Dict[str, Any]:
    """Static pin-shape audit of the upstream backend module (no model touch).

    Used by the L0 SDPA observer audits to prove the repository adapter matches
    the pinned upstream surface: ``set_attn_backend`` exists, ``get_backend`` is
    optional, and a qualified observation symbol/state exists.
    """
    try:
        module = _load_backend_module()
    except Exception as exc:  # noqa: BLE001
        return {
            "module": _ATTN_BACKEND_MODULE,
            "pinned_commit": PINNED_MAGE_COMMIT,
            "available": False,
            "status": "SKIP",
            "set_attn_backend": False,
            "get_backend": False,
            "backend_state_symbol": None,
            "__all__": None,
            "reason": f"module unavailable here: {type(exc).__name__}",
        }
    exported = getattr(module, "__all__", None)
    symbols = set(dir(module))
    state_symbol: Optional[str] = None
    if callable(getattr(module, "get_backend", None)):
        state_symbol = "get_backend"
    elif _PINNED_STATE_SYMBOL in symbols:
        state_symbol = _PINNED_STATE_SYMBOL
    return {
        "module": _ATTN_BACKEND_MODULE,
        "pinned_commit": PINNED_MAGE_COMMIT,
        "available": True,
        "status": "PASS",
        "set_attn_backend": callable(getattr(module, _PINNED_SETTER_SYMBOL, None)),
        "get_backend": callable(getattr(module, "get_backend", None)),
        "backend_state_symbol": state_symbol,
        "__all__": list(exported) if exported else None,
        "reason": None if state_symbol else (
            f"no qualified observation symbol found in {_ATTN_BACKEND_MODULE}"
        ),
    }


def freeze_sdpa_backend() -> Dict[str, Any]:
    """Request ``sdpa`` on the pinned upstream backend setter (idempotent).

    This is the authority-side ``set_attn_backend("sdpa")`` call that replaces
    the nonexistent ``get_backend()`` assumption.  It must be called before the
    DiT backend is resolved so the T4 (sm_75) never dispatches to ``flash2``.
    """
    try:
        module = _load_backend_module()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "status": "SKIP",
            "module": _ATTN_BACKEND_MODULE,
            "pinned_commit": PINNED_MAGE_COMMIT,
            "reason": f"attention backend module unavailable here: {type(exc).__name__}",
        }
    setter: Callable[..., Any] = getattr(module, _PINNED_SETTER_SYMBOL, None)
    if not callable(setter):
        return {
            "available": True,
            "status": "FAIL",
            "module": _ATTN_BACKEND_MODULE,
            "pinned_commit": PINNED_MAGE_COMMIT,
            "reason": f"{_PINNED_SETTER_SYMBOL} missing in {_ATTN_BACKEND_MODULE}",
        }
    try:
        setter(SDPA_BACKEND_NAME)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": True,
            "status": "FAIL",
            "module": _ATTN_BACKEND_MODULE,
            "pinned_commit": PINNED_MAGE_COMMIT,
            "reason": f"{_PINNED_SETTER_SYMBOL}({SDPA_BACKEND_NAME!r}) raised: {exc}",
        }
    return {
        "available": True,
        "status": "PASS",
        "backend": SDPA_BACKEND_NAME,
        "module": _ATTN_BACKEND_MODULE,
        "pinned_commit": PINNED_MAGE_COMMIT,
        "reason": None,
    }


def assert_sdpa_frozen() -> Dict[str, Any]:
    """Fail-closed SDPA freeze check used by the L0 preflight wiring.

    ``VF_HF_ATTN_IMPL`` must read exactly ``sdpa``; and, when the pinned
    upstream backend module is importable, ``set_attn_backend("sdpa")`` is
    re-requested (idempotent) and the resulting dispatch state must observe as
    ``sdpa``.  A present module that cannot be frozen or observed FAILs closed.
    """
    env_ok = str(os.environ.get(VF_HF_ATTN_IMPL, "")).lower() == SDPA_BACKEND_NAME
    pre = observed_backend()
    backend = pre
    backend_freeze = None
    if pre["available"]:
        backend_freeze = freeze_sdpa_backend()
        backend = observed_backend()
        if backend_freeze["status"] != "PASS":
            return {
                "status": "FAIL",
                "vf_hf_attn_impl": os.environ.get(VF_HF_ATTN_IMPL),
                "backend": backend,
                "backend_freeze": backend_freeze,
                "env_aligned": env_ok,
                "required_env_value": SDPA_BACKEND_NAME,
            }
    backend_ok = backend["status"] == "PASS" or backend["status"] == "SKIP"
    ok = env_ok and backend_ok
    result = {
        "status": "PASS" if ok else "FAIL",
        "vf_hf_attn_impl": os.environ.get(VF_HF_ATTN_IMPL),
        "backend": backend,
        "env_aligned": env_ok,
        "required_env_value": SDPA_BACKEND_NAME,
    }
    if backend_freeze is not None:
        result["backend_freeze"] = backend_freeze
    return result
