from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mage_t4x2 import constants as C
from mage_t4x2.bootstrap_environment import prepare_bootstrap_environment
from mage_t4x2.upstream_bootstrap import UpstreamMageProvenance, bootstrap_upstream_mage

from .contract import (
    UPSTREAM_MAGE_COMMIT,
    UPSTREAM_MAGE_REPOSITORY,
    UPSTREAM_MAGE_TREE,
)

UPSTREAM_REPOSITORY = UPSTREAM_MAGE_REPOSITORY
UPSTREAM_COMMIT = UPSTREAM_MAGE_COMMIT
UPSTREAM_MAGE_FLOW_TREE = UPSTREAM_MAGE_TREE
LOGURU_WHEEL = "loguru-0.7.3-py3-none-any.whl"
LOGURU_WHEEL_BYTES = 61595
LOGURU_WHEEL_SHA256 = "31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c"


class PublicBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicBootstrapResult:
    runtime_root: str
    upstream_root: str
    wheelhouse_dir: str
    bootstrap_target: str
    bootstrap_env: dict
    upstream: UpstreamMageProvenance


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise PublicBootstrapError(
            f"command failed rc={proc.returncode}: {' '.join(argv)}\n{proc.stdout[-4000:]}"
        )
    return proc.stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def public_runtime_root(project_root: Path, run_id: str) -> Path:
    return project_root.parent / ".mage-flow-public-runtime" / run_id


def verify_checkout_identity(
    checkout: Path,
    *,
    expected_commit: str = UPSTREAM_COMMIT,
    expected_tree: str = UPSTREAM_MAGE_FLOW_TREE,
) -> dict:
    head = _run(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    tree = _run(["git", "-C", str(checkout), "rev-parse", "HEAD:mage_flow"])
    if head != expected_commit:
        raise PublicBootstrapError(f"upstream HEAD mismatch: expected {expected_commit}, got {head}")
    if tree != expected_tree:
        raise PublicBootstrapError(f"mage_flow tree mismatch: expected {expected_tree}, got {tree}")
    return {"status": "PASS", "head": head, "mage_flow_tree": tree}


def checkout_upstream_mage(project_root: Path, runtime_root: Path) -> Path:
    checkout = runtime_root / "upstream-mage"
    if checkout.exists():
        raise PublicBootstrapError(f"stale upstream checkout exists: {checkout}")
    _run(["git", "clone", "--no-checkout", UPSTREAM_REPOSITORY, str(checkout)])
    _run(["git", "-C", str(checkout), "checkout", "--detach", UPSTREAM_COMMIT])
    verify_checkout_identity(checkout)

    authority_manifest = project_root / "authority" / "upstream-mage-runtime-provenance.json"
    if not authority_manifest.is_file():
        raise PublicBootstrapError(f"upstream provenance manifest missing: {authority_manifest}")
    shutil.copy2(authority_manifest, checkout / "UPSTREAM_SOURCE_PROVENANCE.json")
    return checkout


def verify_wheel_artifact(path: Path) -> dict:
    if not path.is_file():
        raise PublicBootstrapError(f"bootstrap wheel missing: {path}")
    size = path.stat().st_size
    digest = _sha256(path)
    if size != LOGURU_WHEEL_BYTES:
        raise PublicBootstrapError(
            f"bootstrap wheel size mismatch: expected {LOGURU_WHEEL_BYTES}, got {size}"
        )
    if digest != LOGURU_WHEEL_SHA256:
        raise PublicBootstrapError(
            f"bootstrap wheel sha256 mismatch: expected {LOGURU_WHEEL_SHA256}, got {digest}"
        )
    return {"status": "PASS", "size": size, "sha256": digest}


def download_bootstrap_wheel(runtime_root: Path) -> Path:
    wheelhouse = runtime_root / "wheelhouse"
    if wheelhouse.exists():
        raise PublicBootstrapError(f"stale wheelhouse exists: {wheelhouse}")
    wheelhouse.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(wheelhouse),
            "loguru==0.7.3",
        ]
    )
    wheel = wheelhouse / LOGURU_WHEEL
    verify_wheel_artifact(wheel)
    unexpected = sorted(p.name for p in wheelhouse.iterdir() if p.name != LOGURU_WHEEL)
    if unexpected:
        raise PublicBootstrapError(f"unexpected bootstrap wheelhouse files: {unexpected}")
    return wheelhouse


def prepare_public_bootstrap(project_root: Path, run_id: str) -> PublicBootstrapResult:
    root = project_root.resolve()
    runtime_root = public_runtime_root(root, run_id)
    if runtime_root.exists():
        raise PublicBootstrapError(f"stale public runtime root exists: {runtime_root}")

    try:
        runtime_root.mkdir(parents=True)
        checkout = checkout_upstream_mage(root, runtime_root)
        wheelhouse = download_bootstrap_wheel(runtime_root)
        bootstrap_target = runtime_root / "bootstrap-site"
        bootstrap_env = prepare_bootstrap_environment(
            project_root=root,
            requirements_path=root / "requirements-bootstrap.lock",
            wheelhouse_dir=wheelhouse,
            wheelhouse_manifest=root / "vendor" / "bootstrap-wheelhouse-manifest.json",
            target=bootstrap_target,
        )
        upstream = bootstrap_upstream_mage(str(checkout))
    except PublicBootstrapError:
        raise
    except Exception as exc:
        raise PublicBootstrapError(f"{type(exc).__name__}: {exc}") from exc

    return PublicBootstrapResult(
        runtime_root=str(runtime_root),
        upstream_root=str(checkout),
        wheelhouse_dir=str(wheelhouse),
        bootstrap_target=str(bootstrap_target),
        bootstrap_env=bootstrap_env,
        upstream=upstream,
    )
