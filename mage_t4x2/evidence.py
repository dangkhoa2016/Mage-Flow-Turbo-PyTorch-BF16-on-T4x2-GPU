"""Machine-readable evidence run directory writer.

GPU run layout (per directive section 13)::

    evidence/gpu/<RUN_ID>/
        run-contract.json  environment.json  gpu-inventory.json
        source-authority.json  dependency-authority.txt  dtype-before.json
        dtype-after.json  device-map.json  telemetry.jsonl  runtime.log
        output.png  output.sha256  image-validation.json
        acceptance.json  manifest.sha256

GPU-specific documents default to ``NOT_RUN`` during CPU Stage A.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import hashing


def _now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def generate_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def write_json_doc(path: str, data: Any) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class EvidenceRun:
    """Writers for a single run directory. CPU stage uses ``stage()`` scoped
    writers to produce the criterion documents without requiring GPU facts."""

    def __init__(self, run_dir: str, source_authority: Optional[Dict[str, Any]] = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.created_at = _now_utc()
        self._source_authority = source_authority or {}

    # -- individual documents -------------------------------------------------
    def write_run_contract(self, contract: Dict[str, Any]) -> str:
        return write_json_doc(str(self.run_dir / "run-contract.json"), contract)

    def write_environment(self, environment: Dict[str, Any]) -> str:
        return write_json_doc(str(self.run_dir / "environment.json"), environment)

    def write_gpu_inventory(self, inventory: Optional[List[Dict[str, Any]]]) -> str:
        data = inventory if inventory is not None else {"state": "NOT_RUN", "gpus": []}
        return write_json_doc(str(self.run_dir / "gpu-inventory.json"), data)

    def write_source_authority(self, authority: Optional[Dict[str, Any]] = None) -> str:
        data = authority or self._source_authority
        return write_json_doc(str(self.run_dir / "source-authority.json"), data)

    def write_dependency_authority_text(self, text: str) -> str:
        path = self.run_dir / "dependency-authority.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def write_dtype(self, which: str, report: Dict[str, Any]) -> str:
        assert which in ("before", "after")
        return write_json_doc(str(self.run_dir / f"dtype-{which}.json"), report)

    def write_device_map(self, device_map: Dict[str, Any]) -> str:
        return write_json_doc(str(self.run_dir / "device-map.json"), device_map)

    def write_telemetry_path(self, jsonl_here: str) -> str:
        dst = self.run_dir / "telemetry.jsonl"
        shutil.copyfile(jsonl_here, dst)
        return str(dst)

    def write_runtime_log(self, text: str) -> str:
        path = self.run_dir / "runtime.log"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def copy_output_image(self, src: str, name: str = "output.png") -> Optional[str]:
        src_path = Path(src)
        dst = self.run_dir / name
        if not src_path.is_file():
            return None
        if src_path.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        manifest_path = self.run_dir / f"{name}.sha256"
        manifest_path.write_text(hashing.sha256_file(str(dst)) + "\n", encoding="utf-8")
        return str(dst)

    def write_image_validation(self, report: Dict[str, Any]) -> str:
        return write_json_doc(str(self.run_dir / "image-validation.json"), report)

    def write_acceptance(self, report: Dict[str, Any]) -> str:
        return write_json_doc(str(self.run_dir / "acceptance.json"), report)

    def write_failure(self, failure: Dict[str, Any]) -> str:
        return write_json_doc(str(self.run_dir / "failure.json"), failure)

    # -- manifest -------------------------------------------------------------
    def finalize_manifest(self) -> Dict[str, str]:
        names = sorted(p.name for p in self.run_dir.iterdir() if p.is_file())
        return hashing.build_manifest(str(self.run_dir), names)

    def list_documents(self) -> List[str]:
        return sorted(p.name for p in self.run_dir.iterdir() if p.is_file())


def bundle_run(run_dir: str, out_archive: str) -> str:
    import tarfile

    with tarfile.open(out_archive, "w:gz") as tar:
        tar.add(run_dir, arcname=os.path.basename(run_dir.rstrip("/")))
    return out_archive