"""Single source of truth for the public dual-T4 canonical demo contract.

The public qualification runner, notebook, tests, and documentation consume
this contract only. Non-canonical public inputs fail closed and are never
reported as qualification PASS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

MODEL_ID = "dangkhoa2016/mage-flow-community-mage-flow-turbo"
MODEL_REVISION = "65bb3500f0da9df6a41ec6383716fc02cf014773"
UPSTREAM_MAGE_REPOSITORY = "https://github.com/microsoft/Mage"
UPSTREAM_MAGE_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
UPSTREAM_MAGE_TREE = "946b91bcb2cac75e6cfe8399f0f7f330a2280adf"

CANONICAL_PROMPT = "a red fox in a snowy forest at golden hour, high detail"

CANONICAL_SEED = 42
CANONICAL_STEPS = 4
CANONICAL_CFG = 1.0
CANONICAL_WIDTH = 512
CANONICAL_HEIGHT = 512
CANONICAL_NUM_BLOCKS = 12
CANONICAL_SPLIT_BLOCK = 1
CANONICAL_ATTENTION_BACKEND = "sdpa"
CANONICAL_DTYPE = "bfloat16"

_OVERRIDABLE_INPUTS = (
    "model_id",
    "model_revision",
    "upstream_mage_commit",
    "upstream_mage_tree",
    "attention_backend",
    "dtype",
    "prompt",
    "width",
    "height",
    "steps",
    "cfg_scale",
    "seed",
    "num_transformer_blocks",
    "split_block",
)


class PublicDemoContractError(RuntimeError):
    """Raised when a public demo input or model source differs from the canonical contract."""


@dataclass(frozen=True)
class PublicDemoContract:
    """Immutable canonical inputs and dual-T4 topology for the public demo."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    upstream_mage_commit: str = UPSTREAM_MAGE_COMMIT
    upstream_mage_tree: str = UPSTREAM_MAGE_TREE
    attention_backend: str = CANONICAL_ATTENTION_BACKEND
    dtype: str = CANONICAL_DTYPE
    prompt: str = CANONICAL_PROMPT
    width: int = CANONICAL_WIDTH
    height: int = CANONICAL_HEIGHT
    steps: int = CANONICAL_STEPS
    cfg_scale: float = CANONICAL_CFG
    seed: int = CANONICAL_SEED
    num_transformer_blocks: int = CANONICAL_NUM_BLOCKS
    split_block: int = CANONICAL_SPLIT_BLOCK

    @property
    def model_path(self) -> Path:
        return (
            Path("/kaggle/input/models")
            / self.model_id
            / "pytorch"
            / "default"
            / "1"
        )

    @property
    def working_root(self) -> Path:
        return Path("/kaggle/working")

    @property
    def expected_block_devices(self) -> Dict[int, str]:
        return {
            index: ("cuda:0" if index < self.split_block else "cuda:1")
            for index in range(self.num_transformer_blocks)
        }

    def errors(self) -> List[str]:
        """Return every mismatch between this instance and the canonical contract."""
        errors: List[str] = []
        if self.model_id != MODEL_ID:
            errors.append(f"non-canonical model_id: {self.model_id!r}")
        if self.model_revision != MODEL_REVISION:
            errors.append(f"non-canonical model_revision: {self.model_revision!r}")
        if self.upstream_mage_commit != UPSTREAM_MAGE_COMMIT:
            errors.append(f"non-canonical upstream_mage_commit: {self.upstream_mage_commit!r}")
        if self.upstream_mage_tree != UPSTREAM_MAGE_TREE:
            errors.append(f"non-canonical upstream_mage_tree: {self.upstream_mage_tree!r}")
        if self.attention_backend != CANONICAL_ATTENTION_BACKEND:
            errors.append(
                f"public Tesla T4 path requires SDPA, got {self.attention_backend!r}"
            )
        if self.dtype != CANONICAL_DTYPE:
            errors.append(f"public demo requires BF16 dtype/materialization, got {self.dtype!r}")
        if self.prompt != CANONICAL_PROMPT:
            errors.append("public demo prompt is pinned to the canonical prompt")
        if (self.width, self.height) != (CANONICAL_WIDTH, CANONICAL_HEIGHT):
            errors.append(
                f"public demo resolution is pinned to {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}, "
                f"got {self.width}x{self.height}"
            )
        if self.steps != CANONICAL_STEPS:
            errors.append(
                f"public demo denoising steps are pinned to {CANONICAL_STEPS}, got {self.steps}"
            )
        if self.cfg_scale != CANONICAL_CFG:
            errors.append(f"public demo CFG is pinned to {CANONICAL_CFG}, got {self.cfg_scale}")
        if self.seed != CANONICAL_SEED:
            errors.append(f"public demo seed is pinned to {CANONICAL_SEED}, got {self.seed}")
        if self.num_transformer_blocks != CANONICAL_NUM_BLOCKS:
            errors.append(
                f"public demo expects {CANONICAL_NUM_BLOCKS} transformer blocks, "
                f"got {self.num_transformer_blocks}"
            )
        if self.split_block != CANONICAL_SPLIT_BLOCK:
            errors.append(
                f"public demo split_block must be {CANONICAL_SPLIT_BLOCK}, got {self.split_block}"
            )

        block_count = self.num_transformer_blocks
        if not isinstance(block_count, int) or block_count <= 0:
            errors.append("num_transformer_blocks must be a positive integer")
            return errors
        split_block = self.split_block
        if not isinstance(split_block, int) or not (1 <= split_block <= block_count):
            errors.append("split_block must be an integer within the transformer block range")
            return errors
        devices = self.expected_block_devices
        if devices.get(0) != "cuda:0":
            errors.append("transformer block 0 must be on cuda:0")
        if any(devices[index] != "cuda:1" for index in range(1, block_count)):
            errors.append("transformer blocks 1..N-1 must be on cuda:1")
        return errors

    def validate(self) -> None:
        errors = self.errors()
        if errors:
            raise PublicDemoContractError("NONCANONICAL_PUBLIC_CONTRACT: " + "; ".join(errors))


CANONICAL = PublicDemoContract()


def canonical_model_path() -> Path:
    """The exact owner-qualified Kaggle model attachment for the public demo."""
    return CANONICAL.model_path


def public_demo_input_mismatches(**overrides: object) -> List[str]:
    """Report every supplied public demo input that differs from the canonical contract."""
    mismatches: List[str] = []
    for name, value in overrides.items():
        if name not in _OVERRIDABLE_INPUTS:
            mismatches.append(f"unknown public demo input: {name}")
        elif value is not None and value != getattr(CANONICAL, name):
            mismatches.append(
                f"non-canonical {name}: got {value!r}, expected {getattr(CANONICAL, name)!r}"
            )
    return mismatches


def require_canonical_public_demo_inputs(**overrides: object) -> None:
    """Fail closed unless every supplied public demo input equals the canonical value."""
    mismatches = public_demo_input_mismatches(**overrides)
    if mismatches:
        raise PublicDemoContractError("NONCANONICAL_PUBLIC_INPUT: " + "; ".join(mismatches))


def require_public_demo_model_path(selected: str) -> str:
    """Reject any selected model path that is not the canonical owner-qualified attachment.

    The owner namespace is never dropped, basename-only paths are rejected, and
    no remote fallback is accepted for the canonical public qualification run.
    """
    canonical = canonical_model_path().resolve()
    if Path(selected).resolve() != canonical:
        raise PublicDemoContractError(
            f"model path must be the canonical owner-qualified attachment: {canonical}"
        )
    return selected


def require_public_demo_model_source(resolution: Mapping[str, Any]) -> None:
    """Reject any model resolution that is not the owner-qualified local attachment."""
    from mage_t4x2.model_provenance import MODEL_SOURCE_LOCAL

    source = resolution.get("model_source")
    if source != MODEL_SOURCE_LOCAL:
        raise PublicDemoContractError(
            f"public demo requires the owner-qualified local model attachment, "
            f"got {source!r}; remote fallback is forbidden"
        )