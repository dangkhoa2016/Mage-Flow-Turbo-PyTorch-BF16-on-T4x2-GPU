"""Precision contract definition for the all-BF16 qualification target.

The contract language::

    All major trainable/inference model modules — text encoder, Mage DiT/Transformer,
    and VAE — must have floating parameters and persistent floating buffers
    materialized as ``torch.bfloat16``, unless an explicitly documented upstream
    buffer is required to remain another dtype. Any exception must be enumerated
    and cannot be silently accepted.

Three classifications exist: ``BF16_REQUIRED``, ``BF16_TRACKED``, ``NON_FLOATING``.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Optional, Tuple

from . import constants as C

from .constants import (
    BF16_REQUIRED_STR,
    BF16_TRACKED_STR,
    NON_FLOATING_STR,
)

KNOWN_FLOAT_DTYPE_NAMES = ("float16", "float32", "float64", "bfloat16")


@dataclasses.dataclass(frozen=True)
class PrecisionClassification:
    """One of the three contract classifications."""

    name: str

    @property
    def requires_bf16(self) -> bool:
        return self.name == BF16_REQUIRED_STR

    @property
    def tracks_float(self) -> bool:
        return self.name in (BF16_REQUIRED_STR, BF16_TRACKED_STR)

    def __str__(self) -> str:
        return self.name


BF16_REQUIRED = PrecisionClassification(BF16_REQUIRED_STR)
BF16_TRACKED = PrecisionClassification(BF16_TRACKED_STR)
NON_FLOATING = PrecisionClassification(NON_FLOATING_STR)

_ALL_BY_NAME = {c.name: c for c in (BF16_REQUIRED, BF16_TRACKED, NON_FLOATING)}


def classification(name: str) -> PrecisionClassification:
    try:
        return _ALL_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown precision classification {name!r}") from exc


DEFAULT_PRECISION_CONTRACT: Dict[str, str] = {
    "text_encoder": BF16_REQUIRED_STR,
    "transformer": BF16_REQUIRED_STR,
    "vae": BF16_REQUIRED_STR,
    "scheduler": BF16_TRACKED_STR,
    "latents": BF16_TRACKED_STR,
    "prompt_embeddings": BF16_TRACKED_STR,
    "helper_tensors": BF16_TRACKED_STR,
    "tokenizer": NON_FLOATING_STR,
}


def classify_component(name: str, contract: Optional[Dict[str, str]] = None) -> PrecisionClassification:
    table = contract or DEFAULT_PRECISION_CONTRACT
    if name not in table:
        raise ValueError(
            f"component {name!r} is not declared in the precision contract; "
            "refusing to silently classify"
        )
    return classification(table[name])


@dataclasses.dataclass(frozen=True)
class AllowedException:
    """An explicitly enumerated, documented exception to the BF16 requirement.

    Examples: an upstream boolean/position buffer that must stay integer, or a
    numerically-required FP32 scheduler scalar that is explicitly allowed.
    """

    exception_id: str
    component: str
    tensor_name: str  # parameter/buffer path, e.g. "vae.decoder.alpha"
    dtype: str  # "torch.float32", "torch.int64", ...
    reason: str

    def matches(self, component: str, tensor_name: str, dtype: str) -> bool:
        return (
            self.component == component
            and self.tensor_name == tensor_name
            and self.dtype == dtype
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ExceptionRegistry:
    """Registry of explicit exceptions. Exceptions must be enumerated or the
    BF16 claim is invalid."""

    def __init__(self, entries: Optional[Iterable[AllowedException]] = None) -> None:
        self._entries: List[AllowedException] = list(entries or [])

    def add(self, entry: AllowedException) -> None:
        self._entries.append(entry)

    def find(self, component: str, tensor_name: str, dtype: str) -> Optional[AllowedException]:
        for entry in self._entries:
            if entry.matches(component, tensor_name, dtype):
                return entry
        return None

    def to_list(self) -> List[dict]:
        return [e.to_dict() for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)


def declare_exceptions() -> ExceptionRegistry:
    """The default explicit exception list validated during CPU Stage A.

    Empty for now: upstream Mage already materializes all three major components
    in bf16. Any discovered FP32/scalar exception must be added here explicitly.
    """
    return ExceptionRegistry([])


def dtype_short_name(dtype) -> str:
    return f"torch.{str(dtype).replace('torch.', '')}"


def is_float_dtype(dtype) -> bool:
    try:
        import torch

        if isinstance(dtype, torch.dtype):
            name = str(dtype)
            return name.startswith(("torch.float", "torch.bfloat", "torch.float8", "torch.float4"))
    except Exception:
        pass
    return str(dtype).startswith(("torch.float", "torch.bfloat", "torch.float8", "torch.float4"))


def is_integer_dtype(dtype) -> bool:
    try:
        import torch

        if isinstance(dtype, torch.dtype):
            return dtype.is_floating_point is False and dtype.is_complex is False
    except Exception:
        pass
    name = str(dtype)
    return any(k in name for k in ("int", "bool", "uint", "long", "short"))


def is_bfloat16(dtype) -> bool:
    return str(dtype) in ("torch.bfloat16", "torch.bfloat16") or getattr(dtype, "__name__", "") == "bfloat16"


def is_known_float(dtype) -> bool:
    name = str(dtype).replace("torch.", "")
    return name in KNOWN_FLOAT_DTYPE_NAMES