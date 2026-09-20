from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_ID = "dangkhoa2016/mage-flow-community-mage-flow-turbo"
MODEL_REVISION = "65bb3500f0da9df6a41ec6383716fc02cf014773"
UPSTREAM_MAGE_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"
UPSTREAM_MAGE_TREE = "946b91bcb2cac75e6cfe8399f0f7f330a2280adf"
R2G_RUNTIME_BASELINE_SHA256 = (
    "b829fdea0b568ff80ff74d1c58344d7f289d781d42029f2b42a34a8d9c266466"
)


@dataclass(frozen=True)
class PublicDemoContract:
    """Immutable public-demo inputs and dual-T4 topology."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    upstream_mage_commit: str = UPSTREAM_MAGE_COMMIT
    upstream_mage_tree: str = UPSTREAM_MAGE_TREE
    attention_backend: str = "sdpa"
    dtype: str = "bfloat16"
    width: int = 512
    height: int = 512
    steps: int = 4
    cfg_scale: float = 1.0
    seed: int = 42
    num_transformer_blocks: int = 12
    split_block: int = 1

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
    def expected_block_devices(self) -> dict[int, str]:
        return {
            index: ("cuda:0" if index < self.split_block else "cuda:1")
            for index in range(self.num_transformer_blocks)
        }

    def validate(self) -> None:
        if self.attention_backend != "sdpa":
            raise ValueError("public Tesla T4 path requires SDPA")
        if self.dtype != "bfloat16":
            raise ValueError("public demo requires BF16 dtype/materialization")
        if (self.width, self.height) != (512, 512):
            raise ValueError("public demo resolution is pinned to 512x512")
        if self.steps != 4:
            raise ValueError("public demo denoising steps are pinned to 4")
        if self.cfg_scale != 1.0:
            raise ValueError("public demo CFG is pinned to 1.0")
        if self.num_transformer_blocks != 12:
            raise ValueError("public demo expects 12 transformer blocks")
        if self.split_block != 1:
            raise ValueError("public demo split_block must be 1")

        devices = self.expected_block_devices
        if devices[0] != "cuda:0":
            raise ValueError("transformer block 0 must be on cuda:0")
        if any(devices[index] != "cuda:1" for index in range(1, 12)):
            raise ValueError("transformer blocks 1..11 must be on cuda:1")


def required_model_files() -> tuple[str, ...]:
    return (
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    )
