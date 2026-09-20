"""Corrective G0_LOAD_OOM — static authority guards over the load path.

Fail-closed checks that re-encode the root cause of the captured
``CUDA_OOM_GPU0 / MODEL_LOAD_ONCE`` failure: no authority load path may ever
call whole-model ``.to(device)`` or load the MageFlow pipeline onto CUDA.

All guards are pure-AST/string checks: import-safe on CPU, no torch import.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GPU_SESSION = _REPO_ROOT / "scripts" / "gpu_session.py"

_SINGLE_T4_RETIRED_NAMES = (
    "apply_single_t4_mixed",
    "apply_single_t4_all_bf16",
)

_LOCKED_FILES = (
    "mage_t4x2/backend_contract.py",
    "mage_t4x2/stage_b_session.py",
    "mage_t4x2/phase_backend.py",
)


_SPREAD_LOADER = _REPO_ROOT / "mage_t4x2" / "spread_loader.py"


def _spread_loader_builds_on_cpu() -> bool:
    text = _SPREAD_LOADER.read_text(encoding="utf-8")
    return 'builder(model_path, "cpu")' in text or 'builder(model_path, \'cpu\')' in text


def _functions_ast() -> Dict[str, ast.FunctionDef]:
    tree = ast.parse(_GPU_SESSION.read_text(encoding="utf-8"))
    found: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
    return found


def _calls_to_module_toplevel(fn: ast.FunctionDef, name: str) -> List[ast.Call]:
    calls: List[ast.Call] = []

    class _Walker(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                calls.append(node)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == name:
                calls.append(node)
            self.generic_visit(node)

    _Walker().visit(fn)
    return calls


def guard_dual_spread_routing() -> List[str]:
    """`load_runtime_state` must route through the dual-T4 spread load."""
    issues: List[str] = []
    fns = _functions_ast()
    route = fns.get("load_runtime_state")
    if route is None:
        return ["gpu_session.load_runtime_state missing"]
    if not _calls_to_module_toplevel(route, "load_runtime_state_dual_t4_spread"):
        issues.append("load_runtime_state does not route to load_runtime_state_dual_t4_spread")
    return issues


def guard_cpu_stage_load() -> List[str]:
    """Every `load_from_repo` call must stage on device=\"cpu\" (never device=\"cuda\")."""
    issues: List[str] = []
    fns = _functions_ast()
    loader = fns.get("load_runtime_state_dual_t4_spread")
    if loader is None:
        return ["load_runtime_state_dual_t4_spread missing"]

    device_bound_cpu = False

    class _Bind(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            nonlocal device_bound_cpu
            if any(isinstance(t, ast.Name) and t.id == "device" for t in node.targets):
                if isinstance(node.value, ast.Constant) and node.value.value == "cpu":
                    device_bound_cpu = True
            self.generic_visit(node)

    _Bind().visit(loader)

    for call in _calls_to_module_toplevel(loader, "load_from_repo"):
        device_arg = None
        resolved = False
        if call.keywords:
            for kw in call.keywords:
                if kw.arg == "device":
                    if isinstance(kw.value, ast.Constant):
                        device_arg = kw.value.value
                        resolved = True
                    elif isinstance(kw.value, ast.Name) and kw.value.id == "device":
                        if device_bound_cpu or _spread_loader_builds_on_cpu():
                            device_arg = "cpu"
                        else:
                            issues.append(
                                f"load_from_repo device binding not statically 'cpu': {ast.unparse(call)[:120]}"
                            )
                        resolved = True
        if not resolved and len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
            device_arg = call.args[1].value
            resolved = True
        if not resolved or device_arg not in ("cpu",):
            issues.append(
                f"load_from_repo must load onto 'cpu', found {device_arg!r}: {ast.unparse(call)[:120]}"
            )
    return issues


def guard_no_whole_model_to_device() -> List[str]:
    """No `model.to(device)` (whole-pipeline) attribute call in the load path."""
    issues: List[str] = []
    fns = _functions_ast()
    for name in ("load_runtime_state_dual_t4_spread",):
        fn = fns.get(name)
        if fn is None:
            continue

        class _Walker(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "to" and isinstance(func.value, ast.Name):
                    issues.append(
                        f"whole-model .to() call on `{func.value.id}` in {name}: {ast.unparse(node)[:120]}"
                    )
                self.generic_visit(node)

        _Walker().visit(fn)
    return issues


def guard_single_t4_retired() -> List[str]:
    """The retired single-T4 gates must not appear in any locked authority file."""
    issues: List[str] = []
    for rel in _LOCKED_FILES:
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for name in _SINGLE_T4_RETIRED_NAMES:
            if name in text:
                issues.append(f"{rel} still references retired gate {name}")
    for name in _SINGLE_T4_RETIRED_NAMES:
        text = _GPU_SESSION.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if name in line:
                issues.append(f"gpu_session.py:{line_no} still references retired gate {name}")
    return issues


def commit_load_guards() -> str:
    return "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"


def run_all_load_guards() -> Dict[str, Any]:
    """Run every guard; returns issue groups. Empty groups == all PASS."""
    checks: Dict[str, List[str]] = {
        "dual_spread_routing": guard_dual_spread_routing(),
        "cpu_stage_load": guard_cpu_stage_load(),
        "no_whole_model_to_device": guard_no_whole_model_to_device(),
        "single_t4_retired": guard_single_t4_retired(),
    }
    return {
        "status": "PASS" if not any(checks.values()) else "FAIL",
        "checks": {k: {"status": "PASS" if not v else "FAIL", "issues": v} for k, v in checks.items()},
        "guard_target": str(_GPU_SESSION),
        "upstream_commit_pinned": commit_load_guards(),
    }
