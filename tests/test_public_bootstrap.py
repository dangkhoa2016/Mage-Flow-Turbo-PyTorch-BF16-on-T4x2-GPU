from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from public_demo.bootstrap import (
    LOGURU_WHEEL_BYTES,
    LOGURU_WHEEL_SHA256,
    PublicBootstrapError,
    public_runtime_root,
    verify_checkout_identity,
    verify_wheel_artifact,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def test_public_runtime_root_is_outside_checkout(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    target = public_runtime_root(project, "run-1")
    assert target == tmp_path / ".mage-flow-public-runtime" / "run-1"


def test_verify_checkout_identity_uses_head_and_mage_flow_tree(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "mage_flow").mkdir()
    (repo / "mage_flow" / "__init__.py").write_text("VALUE = 1\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "."])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "fixture"])
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD:mage_flow")
    report = verify_checkout_identity(repo, expected_commit=head, expected_tree=tree)
    assert report == {"status": "PASS", "head": head, "mage_flow_tree": tree}


def test_checkout_identity_fails_closed_on_wrong_commit(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "mage_flow").mkdir()
    (repo / "mage_flow" / "__init__.py").write_text("VALUE = 1\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "."])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "fixture"])
    tree = _git(repo, "rev-parse", "HEAD:mage_flow")
    with pytest.raises(PublicBootstrapError, match="HEAD mismatch"):
        verify_checkout_identity(repo, expected_commit="0" * 40, expected_tree=tree)


def test_wheel_artifact_verifier_is_hash_and_size_strict(tmp_path, monkeypatch):
    wheel = tmp_path / "wheel.whl"
    wheel.write_bytes(b"x" * LOGURU_WHEEL_BYTES)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setattr("public_demo.bootstrap.LOGURU_WHEEL_SHA256", digest)
    assert verify_wheel_artifact(wheel)["status"] == "PASS"
    wheel.write_bytes(wheel.read_bytes() + b"x")
    with pytest.raises(PublicBootstrapError, match="size mismatch"):
        verify_wheel_artifact(wheel)


def test_known_wheel_pin_shape_is_stable():
    assert LOGURU_WHEEL_BYTES == 61595
    assert LOGURU_WHEEL_SHA256 == "31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c"


def test_prepare_public_bootstrap_wraps_internal_failures(tmp_path, monkeypatch):
    from public_demo import bootstrap as module

    project = tmp_path / "project"
    (project / "authority").mkdir(parents=True)
    monkeypatch.setattr(module, "checkout_upstream_mage", lambda *_: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(PublicBootstrapError, match="ValueError: boom"):
        module.prepare_public_bootstrap(project, "run-1")
