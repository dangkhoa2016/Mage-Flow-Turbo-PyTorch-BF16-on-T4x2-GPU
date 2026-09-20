"""Runtime environment discovery.

Every function here is lazy: nothing in this module (nor in the package import
path) initializes CUDA. CUDA is only touched by explicit calls from GPU-phase
code.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List, Optional


def python_version() -> str:
    return platform.python_version()


def python_implementation() -> str:
    return platform.python_implementation()


def platform_info() -> str:
    return platform.platform()


def torch_version() -> Optional[str]:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None


def torch_cuda_build() -> Optional[str]:
    try:
        import torch

        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def cuda_is_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def cuda_device_count() -> int:
    if not cuda_is_available():
        return 0
    import torch

    return int(torch.cuda.device_count())


def installed_deps() -> Dict[str, str]:
    import importlib.metadata

    wanted = [
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "numpy",
        "pillow",
        "einops",
        "sentencepiece",
        "tokenizers",
        "pytest",
    ]
    out: Dict[str, str] = {}
    for name in wanted:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "NOT_INSTALLED"
    return out


def _probe_cuda_device(i: int) -> Dict[str, Any]:
    import torch

    name = torch.cuda.get_device_name(i)
    major, minor = torch.cuda.get_device_capability(i)
    total = torch.cuda.get_device_properties(i).total_memory
    props = torch.cuda.get_device_properties(i)
    return {
        "index": i,
        "name": name,
        "compute_capability": f"{major}.{minor}",
        "major": major,
        "minor": minor,
        "total_vram_bytes": int(total),
        "multi_processor_count": int(props.multi_processor_count),
    }


def cuda_inventory() -> List[Dict[str, Any]]:
    """Return detailed inventory of CUDA devices.

    Returns an empty list when CUDA is unavailable. Only call this from an
    explicit runtime function, never at import time.
    """
    if not cuda_is_available():
        return []
    count = cuda_device_count()
    return [_probe_cuda_device(i) for i in range(count)]


def environment_summary() -> Dict[str, Any]:
    """Flat JSON-friendly snapshot used for environment.json evidence."""
    return {
        "python_version": python_version(),
        "python_implementation": python_implementation(),
        "platform": platform_info(),
        "kernel": getattr(sys, "platform", ""),
        "torch_version": torch_version(),
        "torch_cuda_version": torch_cuda_build(),
        "torch_cuda_available": cuda_is_available(),
        "torch_cuda_device_count": cuda_device_count() if cuda_is_available() else 0,
        "deps": installed_deps(),
    }
