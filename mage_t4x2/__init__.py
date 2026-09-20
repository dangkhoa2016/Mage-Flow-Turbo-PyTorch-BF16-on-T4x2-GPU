"""Mage-Flow-Turbo PyTorch BF16 + dual-T4 inference qualification package.

CPU-first preparation stage (Stage A) modules.

IMPORTANT: importing this package MUST NOT initialize CUDA and MUST NOT require a
GPU. All CUDA probing lives inside explicit runtime functions in
``mage_t4x2.environment``.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
