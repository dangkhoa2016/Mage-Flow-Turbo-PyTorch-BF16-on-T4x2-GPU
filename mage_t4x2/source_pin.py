"""Upstream source and dependency authority pinning.

Records the exact upstream Mage commit, the exact model identifier/revision and
the exact dependency versions that the qualification environment must use.  It
also provides the git source-freeze bootstrap helper that DOCUMENTS the working
tree state without ever fabricating a commit.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from . import constants as C
from . import environment as env

_DEPENDENCY_KEYS = (
    "python",
    "torch",
    "torchvision",
    "torchaudio",
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
)


@dataclasses.dataclass(frozen=True)
class SourceAuthority:
    mage_repo_url: str
    mage_commit_sha: str
    model_identifier: str
    model_revision: str
    python_version_target: str
    pytorch_version_target: str
    transformers_version_target: str
    diffusers_version_target: str
    accelerate_version_target: str
    safetensors_version_target: str
    cuda_expectation: str
    pinned_at: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.mage_repo_url.startswith("https://"):
            errors.append("mage_repo_url must be https")
        if len(self.mage_commit_sha) != 40:
            errors.append("mage_commit_sha must be a 40-char SHA1")
        if self.model_revision and len(self.model_revision) != 40:
            errors.append("model_revision must be a 40-char SHA1 when set")
        for field, value, banned in (
            ("python_version_target", self.python_version_target, [">=", "<", "*", "latest"]),
            ("pytorch_version_target", self.pytorch_version_target, [">=", "<", "*", "latest"]),
            ("transformers_version_target", self.transformers_version_target, [">=", "<", "*", "latest"]),
            ("diffusers_version_target", self.diffusers_version_target, [">=", "<", "*", "latest"]),
            ("accelerate_version_target", self.accelerate_version_target, [">=", "<", "*", "latest"]),
            ("safetensors_version_target", self.safetensors_version_target, [">=", "<", "*", "latest"]),
        ):
            if value is None or not str(value).strip():
                errors.append(f"{field} must be pinned to an exact version")
            elif any(b in str(value) for b in banned):
                errors.append(f"{field} must not use moving reference {value!r}")
        if self.mage_commit_sha in ("main", "HEAD", "latest"):
            errors.append("mage_commit_sha is a moving reference")
        return errors


def _exact(raw: str) -> str:
    return str(raw).split("+")[0]


def current_dependency_authority() -> Dict[str, str]:
    deps = env.installed_deps()
    mapping = {
        "python": env.python_version(),
        "torch": _exact(deps.get("torch", "NOT_INSTALLED")),
        "torchvision": _exact(deps.get("torchvision", "NOT_INSTALLED")),
        "torchaudio": _exact(deps.get("torchaudio", "NOT_INSTALLED")),
        "transformers": _exact(deps.get("transformers", "NOT_INSTALLED")),
        "diffusers": _exact(deps.get("diffusers", "NOT_INSTALLED")),
        "accelerate": _exact(deps.get("accelerate", "NOT_INSTALLED")),
        "safetensors": _exact(deps.get("safetensors", "NOT_INSTALLED")),
        "huggingface-hub": _exact(deps.get("huggingface-hub", "NOT_INSTALLED")),
        "numpy": _exact(deps.get("numpy", "NOT_INSTALLED")),
        "pillow": _exact(deps.get("pillow", "NOT_INSTALLED")),
        "einops": _exact(deps.get("einops", "NOT_INSTALLED")),
        "sentencepiece": _exact(deps.get("sentencepiece", "NOT_INSTALLED")),
        "tokenizers": _exact(deps.get("tokenizers", "NOT_INSTALLED")),
        "pytest": _exact(deps.get("pytest", "NOT_INSTALLED")),
    }
    return mapping


def pinned_source_authority() -> SourceAuthority:
    deps = current_dependency_authority()
    return SourceAuthority(
        mage_repo_url=C.UPSTREAM_MAGE_REPO_URL,
        mage_commit_sha=C.UPSTREAM_MAGE_COMMIT_SHA,
        model_identifier=C.MODEL_ID,
        model_revision=C.MODEL_REVISION,
        python_version_target=deps["python"],
        pytorch_version_target=deps["torch"],
        transformers_version_target=deps["transformers"],
        diffusers_version_target=deps["diffusers"],
        accelerate_version_target=deps["accelerate"],
        safetensors_version_target=deps["safetensors"],
        cuda_expectation="NVIDIA Tesla T4 x2 (sm_75); BF16 as PyTorch dtype materialization",
        pinned_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def render_dependency_authority_text(deps: Dict[str, str]) -> str:
    lines = [
        "# Mage-Flow-Turbo-PyTorch-BF16-T4x2 — frozen dependency authority",
        "# Generated during CPU Stage A. No moving references are allowed.",
        "",
    ]
    for key in _DEPENDENCY_KEYS:
        lines.append(f"{key}=={deps.get(key, 'NOT_INSTALLED')}")
    lines.append("")
    lines.append("# CUDA note: torch wheel variant resolved and recorded at GPU G0.")
    return "\n".join(lines)


def write_dependency_authority(path: str) -> Dict[str, str]:
    deps = current_dependency_authority()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_dependency_authority_text(deps), encoding="utf-8")
    return deps


def repo_git_state() -> Dict[str, Any]:
    """Source-freeze bootstrap: record the repository git state explicitly.

    Git commands run from the repository directory, never from ``/kaggle/working``.
    If the working directory is not a Git clone this records
    ``GIT_REPO_AVAILABLE=NO`` and does NOT fabricate a commit; the pinned
    upstream/model hashes already recorded are the source authority.
    """
    repo = Path(__file__).resolve().parent.parent
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return {
            "GIT_REPO_AVAILABLE": "NO",
            "GIT_HEAD": None,
            "GIT_WORKTREE_STATUS": None,
            "SOURCE_COMMANDS_CWD": str(repo),
            "note": "working directory is not a Git clone; pinned source-authority hashes are authoritative",
        }

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    return {
        "GIT_REPO_AVAILABLE": "YES",
        "GIT_HEAD": head.stdout.strip() if head.returncode == 0 else None,
        "GIT_WORKTREE_STATUS": status.stdout.strip() if status.returncode == 0 else None,
        "SOURCE_COMMANDS_CWD": str(repo),
        "note": "git source-freeze executed from the repository directory",
    }