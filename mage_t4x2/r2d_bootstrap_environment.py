"""R2D clean bootstrap environment (mage_t4x2/r2d_bootstrap_environment.py).

Implements the R2D bootstrap dependency authority policy (static runbook
2026-09-17, sections 20-29, 37, 50):

  * ALWAYS create a fresh project-local bootstrap site
  * ALWAYS provision the exact bootstrap lock from the local wheelhouse into it
    (offline pip semantics: --no-index / --find-links / --target, no network)
  * ALWAYS prepend that local site before Mage import
  * ALWAYS verify the bootstrap dependency module origin resolves inside that
    local site
  * a strict meta-path import guard makes global ``loguru`` unusable inside the
    masked clean-room subprocess proof

Even when a globally installed ``loguru==0.7.3`` exists the R2D authority clean
bootstrap gate relies only on the masked proof; global package state remains
informational only.

CPU-only. Importing this module never initializes CUDA.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .bootstrap_dependencies import (
    BootstrapDependencyError,
    BootstrapPin,
    load_bootstrap_requirements,
    load_wheelhouse_manifest,
    reject_network_commands,
)

R2D_BOOTSTRAP_SITE_REL = ".runtime/r2d-bootstrap-site"
STALE_BOOTSTRAP_SITE = "STALE_BOOTSTRAP_SITE"
FRESH_TARGET_MARKER = ".R2D_FRESH_TARGET_CREATED"


class R2DError(RuntimeError):
    """R2D bootstrap environment failure (fail-closed)."""


# ---------------------------------------------------------------------------
# strict loguru import guard (runbook sections 25-26)
# ---------------------------------------------------------------------------

class R2DLoguruGuardFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that intercepts ``loguru`` imports.

    Resolution is only allowed from the approved local bootstrap paths; any
    other origin (global site-packages, user site, another project) raises
    ModuleNotFoundError. With no approved paths, ALL ``loguru`` resolution is
    blocked — this is the pre-provision mask.
    """

    def __init__(self, allowed_paths: Sequence[Any] = ()) -> None:
        self.allowed = [str(Path(p).resolve()) for p in allowed_paths if p]

    def _resolve_from_base(self, fullname: str, base: str) -> Optional[Any]:
        rel = fullname.replace(".", "/")
        base_path = Path(base)
        py_file = base_path / f"{rel}.py"
        pkg_dir = base_path / rel
        if str(py_file.parent).startswith(str(base_path)) and py_file.is_file():
            spec = importlib.machinery.ModuleSpec(
                fullname,
                importlib.machinery.SourceFileLoader(fullname, str(py_file)),
                origin=str(py_file),
            )
            spec.has_location = True
            return spec
        if str(pkg_dir.parent).startswith(str(base_path)) and (pkg_dir / "__init__.py").is_file():
            init_py = pkg_dir / "__init__.py"
            spec = importlib.machinery.ModuleSpec(
                fullname,
                importlib.machinery.SourceFileLoader(fullname, str(init_py)),
                origin=str(init_py),
                is_package=True,
            )
            spec.has_location = True
            spec.submodule_search_locations = [str(pkg_dir)]
            return spec
        return None

    def find_spec(self, fullname: str, path: Optional[List[str]] = None,
                  target: Any = None) -> Optional[Any]:
        if fullname != "loguru" and not fullname.startswith("loguru."):
            return None
        for base in self.allowed:
            if not str(Path(base)).startswith("/"):
                continue
            spec = self._resolve_from_base(fullname, base)
            if spec is not None:
                return spec
        raise ModuleNotFoundError(
            f"No module named {fullname!r}: loguru must resolve from the "
            "approved local bootstrap site, not global site-packages"
        )


def install_loguru_mask(allowed_paths: Sequence[Any] = ()) -> R2DLoguruGuardFinder:
    """Install the strict loguru import guard at sys.meta_path[0].

    Any already-imported ``loguru``/``loguru.*`` modules are evicted so the
    guard is effective. Returns the guard for later uninstall.
    """
    guard = R2DLoguruGuardFinder(allowed_paths)
    sys.meta_path.insert(0, guard)
    for name in [m for m in list(sys.modules) if m == "loguru" or m.startswith("loguru.")]:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return guard


def uninstall_loguru_mask(guard: Optional[R2DLoguruGuardFinder] = None) -> None:
    """Remove the guard from sys.meta_path and invalidate import caches."""
    if guard is not None and guard in sys.meta_path:
        sys.meta_path.remove(guard)
    importlib.invalidate_caches()


# ---------------------------------------------------------------------------
# bootstrap input authority
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
