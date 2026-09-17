"""Source/model provenance verification.

Separates:
    upstream source identity (repository + commit SHA),
    model repository identity (HF id + revision),
    local Kaggle attached files (preload-fast manifest),
    runtime-loaded files (final authority hash mode, optional).

The manifest builder is CPU-testable with small fixture directories. Weights are
never hashed by default in fast mode (only configs + name/size inventory).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import hashing

MODEL_SOURCE_LOCAL = "LOCAL_ATTACHMENT"
MODEL_SOURCE_REMOTE_FALLBACK = "REMOTE_FALLBACK"

CONFIG_RELATIVE_NAMES = (
    "model_index.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "vae/config.json",
    "scheduler/scheduler_config.json",
)

WEIGHT_RELATIVE_PATTERN = (
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)


def _is_weight_file(rel_path: str) -> bool:
    return rel_path.endswith(".safetensors") or rel_path.endswith(".bin")


def build_model_manifest(
    model_path: str,
    source: str = MODEL_SOURCE_LOCAL,
    fast: bool = True,
    hash_weights: bool = False,
) -> Dict[str, Any]:
    """Build a manifest of the attached model directory.

    fast=True: small config SHA-256 + weight filename/size inventory (no weight
    hashing). hash_weights=True enables the final authority hash mode.
    """
    root = Path(model_path)
    files: Dict[str, Dict[str, Any]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = str(full.relative_to(root))
            stat = full.stat()
            entry: Dict[str, Any] = {
                "path": rel,
                "size": stat.st_size,
                "mtime": round(stat.st_mtime, 3),
                "config_sha256": None,
                "weight_hash": None,
            }
            is_config = rel in CONFIG_RELATIVE_NAMES or rel.endswith(".json")
            if is_config and fast:
                entry["config_sha256"] = hashing.sha256_file(str(full))
            if _is_weight_file(rel) and hash_weights:
                entry["weight_hash"] = hashing.sha256_file(str(full))
            files[rel] = entry

    total_bytes = sum(f["size"] for f in files.values())
    weight_shards = [
        {"name": k, "size": v["size"]}
        for k, v in sorted(files.items())
        if _is_weight_file(k)
    ]
    manifest: Dict[str, Any] = {
        "model_path": str(root.resolve()),
        "source": source,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "fast" if fast and not hash_weights else "authority_hash",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "config_hashes": {
            k: v["config_sha256"]
            for k, v in sorted(files.items())
            if v["config_sha256"] is not None
        },
        "weight_shard_names": [s["name"] for s in weight_shards],
        "weight_shard_sizes": {s["name"]: s["size"] for s in weight_shards},
        "weight_shard_hashes": {
            k: v["weight_hash"] for k, v in sorted(files.items())
            if v["weight_hash"] is not None
        },
        "files": files,
    }
    return manifest


def model_identifier_from_manifest(manifest: Dict[str, Any]) -> Optional[str]:
    """Read the model identity from model_index.json captured in the manifest."""
    cfg_path = "model_index.json"
    cfg = manifest.get("config_hashes")
    if cfg_path not in cfg:
        return None
    # config_hashes is a hash *of* the file; identity must be read from source.
    # We store identity separately when available; fall back to path slug.
    return manifest.get("model_identifier")


def write_model_manifest(manifest: Dict[str, Any], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


class ModelPathResolver:
    """Deterministic resolver over candidate Kaggle input roots."""

    def __init__(self, candidates: List[str], required_rel_files: List[str]) -> None:
        self.candidates = [str(c) for c in candidates]
        self.required_rel_files = list(required_rel_files)

    def resolve(self, fallback_id: Optional[str] = None) -> Dict[str, Any]:
        checked: List[Dict[str, Any]] = []
        selected: Optional[str] = None
        reason: Optional[str] = None
        for candidate in self.candidates:
            missing = [
                rel
                for rel in self.required_rel_files
                if not os.path.isfile(os.path.join(candidate, rel))
            ]
            present = {
                rel: rel not in missing
                for rel in self.required_rel_files
            }
            ok = not missing
            checked.append(
                {
                    "path": candidate,
                    "is_dir": os.path.isdir(candidate),
                    "files": present,
                    "ok": ok,
                }
            )
            if ok and selected is None:
                selected = candidate
                reason = "first candidate with all required files present"
        if selected is None:
            selected = fallback_id
            reason = "no local attachment satisfied; fallback to remote HF id" if fallback_id else "no candidate found"
            source = MODEL_SOURCE_REMOTE_FALLBACK if fallback_id else MODEL_SOURCE_REMOTE_FALLBACK
        else:
            source = MODEL_SOURCE_LOCAL
        return {
            "candidate_paths_checked": checked,
            "selected_path": selected,
            "selection_reason": reason,
            "model_source": source,
            "required_files_present": selected is not None and selected != fallback_id if fallback_id else selected is not None,
        }


def model_path_resolution(
    resolver: ModelPathResolver,
    fallback_id: Optional[str] = None,
) -> Dict[str, Any]:
    return resolver.resolve(fallback_id)