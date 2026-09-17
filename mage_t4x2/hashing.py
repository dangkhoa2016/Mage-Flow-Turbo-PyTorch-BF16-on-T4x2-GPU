"""SHA-256 hashing helpers and evidence manifest generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(directory: str, names: List[str]) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for name in names:
        path = Path(directory) / name
        if path.is_file():
            manifest[name] = sha256_file(str(path))
    return manifest


def write_manifest_json(directory: str, manifest: Dict[str, str], out_name: str = "manifest.json") -> str:
    path = Path(directory) / out_name
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def write_manifest_sha256(directory: str, manifest: Dict[str, str], out_name: str = "manifest.sha256") -> str:
    lines = [f"{digest}  {name}" for name, digest in sorted(manifest.items())]
    path = Path(directory) / out_name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_manifest(directory: str, names: List[str]) -> Dict[str, str]:
    manifest = file_manifest(directory, names)
    write_manifest_json(directory, manifest)
    write_manifest_sha256(directory, manifest)
    return manifest