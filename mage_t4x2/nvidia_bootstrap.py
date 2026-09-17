"""NVIDIA user-mode library preflight bootstrap (Corrective G0_LOAD_OOM).

Kaggle T4 kernels require the NVIDIA user-mode libraries
(``libcuda.so`` / ``libcudart.so`` / ``libcublas*`` / ``libcudnn*``) to be
resolvable before :mod:`torch` first initializes CUDA. If they are missing from
``LD_LIBRARY_PATH``, a model load that later calls ``torch.cuda.*`` OOMs or
dies with a driver/library error — indistinguishable-from / adjacent to the
captured G0 load failure. The authority therefore asserts library presence and
prepends the observed NVIDIA lib dir **before** any CUDA initialization.

Import-safety: no torch import at module scope; pure ``os`` logic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

NVIDIA_LIB_DIRS: tuple = (
    "/usr/local/nvidia/lib64",
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu/nvidia",
    "/usr/lib/x86_64-linux-gnu",
)

NVIDIA_LIB_NAMES: tuple = (
    "libcuda.so",
    "libcudart.so",
    "libcublas.so",
    "libcublasLt.so",
    "libcudnn.so",
    "libnvJitLink.so",
    "libnvrtc.so",
)


def _env_lib_paths() -> List[str]:
    raw = os.environ.get("LD_LIBRARY_PATH", "") or ""
    return [p for p in raw.split(os.pathsep) if p]


def find_nvidia_libs() -> Dict[str, Any]:
    """Locate NVIDIA user-mode libraries on the filesystem."""
    present: dict = {}
    for directory in NVIDIA_LIB_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in NVIDIA_LIB_NAMES:
            for candidate in (name, name + ".1"):
                path = os.path.join(directory, candidate)
                if os.path.isfile(path):
                    present.setdefault(name, {"path": path, "basename": os.path.basename(path)})
                    break
    return present


def ensure_nvidia_libs(require_core: bool = True) -> Dict[str, Any]:
    """Assert core NVIDIA libraries are resolvable; prepend dirs to the path.

    Returns a machine-readable record with ``status`` PASS/FAIL. Never fails a
    GPU-less CPU environment (there is nothing to load there) unless
    ``require_core`` is set and the caller is on a CUDA-capable box; on pure CPU
    hosts this returns PASS with an explicit ``cuda_context: absent`` note.
    """
    import torch  # late import; CPU-safe gate only, no CUDA init

    present = find_nvidia_libs()
    env_paths = _env_lib_paths()
    missing: List[str] = []

    core = ("libcuda.so", "libcudart.so")
    if torch.cuda.is_available() or require_core:
        for name in core:
            if name not in present:
                missing.append(name)

    prepended: List[str] = []
    for directory in NVIDIA_LIB_DIRS:
        if directory in env_paths:
            continue
        if os.path.isdir(directory) and any(
            os.path.isfile(os.path.join(directory, n)) for n in NVIDIA_LIB_NAMES
        ):
            prepended.append(directory)
    if prepended:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(prepended + env_paths)

    status = "FAIL" if missing else "PASS"
    return {
        "status": status,
        "present": present,
        "missing": missing,
        "prepended_dirs": prepended,
        "ld_library_path": _env_lib_paths(),
        "cuda_context": "present" if torch.cuda.is_available() else "absent",
        "note": (
            "NVIDIA user-mode libraries verified before CUDA init"
            if not missing
            else "missing NVIDIA user-mode libraries; do not start a GPU load"
        ),
    }


def preflight_nvidia_libs() -> Dict[str, Any]:
    """Fail-closed wrapper used by ``preflight_gpu``."""
    report = ensure_nvidia_libs(require_core=True)
    if report["status"] != "PASS":
        raise RuntimeError(
            "NVIDIA_LIB_MISSING: " + "; ".join(f"{n} not found" for n in report["missing"])
        )
    return report