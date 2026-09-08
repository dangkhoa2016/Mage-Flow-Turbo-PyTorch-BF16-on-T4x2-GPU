"""Frozen minimal inference run contract parsing.

The contract is JSON/YAML-serializable and machine readable. It is the single
authoritative description of one qualification T2I run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import constants as C

_REQUIRED_FIELDS = (
    "model",
    "framework",
    "task",
    "resolution",
    "seed",
    "steps",
    "cfg",
    "accelerator",
    "cpu_fallback",
    "stable_diffusion_cpp",
    "sd_cli",
)

_FORBIDDEN_TRUE_FLAGS = ("cpu_fallback", "stable_diffusion_cpp", "sd_cli")


@dataclasses.dataclass(frozen=True)
class RunContract:
    model: str
    framework: str
    task: str
    resolution: Iterable[int]
    seed: int
    steps: int
    cfg: float
    accelerator: str
    cpu_fallback: bool
    stable_diffusion_cpp: bool
    sd_cli: bool

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.model:
            errors.append("model is empty")
        if self.framework != "upstream Mage" and "PyTorch" not in self.framework:
            errors.append(f"framework must mention upstream Mage/PyTorch: {self.framework!r}")
        if self.task != "text-to-image":
            errors.append(f"task must be text-to-image: {self.task!r}")
        h, w = list(self.resolution)
        if h % 16 != 0 or w % 16 != 0:
            errors.append(f"resolution must be multiple of 16: {h}x{w}")
        if self.seed < 0 or self.seed >= 2**32:
            errors.append("seed out of range")
        if self.steps <= 0:
            errors.append("steps must be positive")
        if self.cfg <= 0:
            errors.append("cfg must be positive")
        if self.cpu_fallback:
            errors.append("cpu_fallback must be False (forbidden in authority runs)")
        if self.stable_diffusion_cpp:
            errors.append("stable_diffusion_cpp must be False (forbidden)")
        if self.sd_cli:
            errors.append("sd_cli must be False (forbidden)")
        if "T4" not in self.accelerator or "x2" not in self.accelerator:
            errors.append(f"accelerator must name dual T4: {self.accelerator!r}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunContract":
        missing = [f for f in _REQUIRED_FIELDS if f not in data]
        if missing:
            raise ValueError(f"run contract missing fields: {missing}")
        return cls(
            model=str(data["model"]),
            framework=str(data["framework"]),
            task=str(data["task"]),
            resolution=_tuple2(data["resolution"]),
            seed=int(data["seed"]),
            steps=int(data["steps"]),
            cfg=float(data["cfg"]),
            accelerator=str(data["accelerator"]),
            cpu_fallback=bool(data["cpu_fallback"]),
            stable_diffusion_cpp=bool(data["stable_diffusion_cpp"]),
            sd_cli=bool(data["sd_cli"]),
        )

    @classmethod
    def from_json(cls, path: str) -> "RunContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml(cls, path: str) -> "RunContract":
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError("PyYAML required to read YAML contracts") from exc
        return cls.from_dict(dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))))


def _tuple2(value: Any) -> Iterable[int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"resolution must be a 2-tuple, got {value!r}")


def default_run_contract() -> RunContract:
    return RunContract(
        model=C.MODEL_ID,
        framework=C.FRAMEWORK,
        task=C.TASK,
        resolution=C.RESOLUTION,
        seed=C.SEED,
        steps=C.STEPS,
        cfg=C.CFG,
        accelerator=C.ACCELERATOR,
        cpu_fallback=C.CPU_FALLBACK,
        stable_diffusion_cpp=C.STABLE_DIFFUSION_CPP_ALLOWED,
        sd_cli=C.SD_CLI_ALLOWED,
    )


def resolve_contract(value: Any) -> RunContract:
    if isinstance(value, RunContract):
        return value
    if isinstance(value, str):
        if value.endswith(".json"):
            return RunContract.from_json(value)
        if value.endswith(".yaml") or value.endswith(".yml"):
            return RunContract.from_yaml(value)
        return RunContract.from_json(value)
    if isinstance(value, dict):
        return RunContract.from_dict(value)
    raise TypeError(f"cannot resolve run contract from {type(value)!r}")
