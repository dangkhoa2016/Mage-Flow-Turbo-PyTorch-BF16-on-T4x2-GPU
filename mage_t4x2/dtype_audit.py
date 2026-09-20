"""Dtype audit over PyTorch modules.

Audits a module (or a dict of modules) against the precision contract and
reports: parameter count, buffer count, dtype histogram, non-floating count,
unexpected dtype list and an explicit exception record.

Fully CPU-testable with small synthetic modules — the real 20 GB model is not
required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import dtype_policy as policy
from .dtype_policy import ExceptionRegistry

import dataclasses


@dataclasses.dataclass(frozen=True)
class AuditResult:
    module: str
    parameter_count: int
    buffer_count: int
    float_parameter_count: int
    float_buffer_count: int
    dtype_histogram: Dict[str, int]
    non_floating_count: int
    unexpected_dtype_list: List[str]
    unexpected_names: List[str]
    allowed_exceptions: List[dict]
    classification: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _iter_attributes(module) -> Any:
    yield from module.named_parameters(recurse=True)
    yield from module.named_buffers(recurse=True)


def _dtype_label(dtype) -> str:
    return str(dtype)


def audit_module(
    module,
    name: str,
    classification,
    exceptions: Optional[ExceptionRegistry] = None,
    exceptional_module_dtypes: Optional[List[Dict[str, str]]] = None,
) -> AuditResult:
    """Audit one module against the precision contract.

    ``exceptional_module_dtypes`` maps tensor_name -> dtype for tensors that are
    allowed to remain non-bf16 (equivalent to declaring exceptions).
    """
    exceptions = exceptions or policy.declare_exceptions()
    histogram: Dict[str, int] = {}
    unexpected: List[str] = []
    unexpected_names: List[str] = []
    allowed_records: List[dict] = []
    non_floating = 0
    fp_params = 0
    fp_buffers = 0
    param_count = 0
    buffer_count = 0
    seen_names = set()

    param_names = {n for n, _ in module.named_parameters(recurse=True)}
    buffer_names = {n for n, _ in module.named_buffers(recurse=True)}

    for tensor_name, tensor in _iter_attributes(module):
        dtype = tensor.dtype
        label = _dtype_label(dtype)
        histogram[label] = histogram.get(label, 0) + 1
        is_param = tensor_name in param_names
        is_buf = tensor_name in buffer_names

        overrides: List[Dict[str, str]] = exceptional_module_dtypes or []
        override = next(
            (o for o in overrides if o.get("tensor_name") == tensor_name), None
        )

        if policy.is_integer_dtype(dtype):
            non_floating += 1
            continue

        if not policy.is_float_dtype(dtype):
            # Unknown (e.g. complex or mocked) dtype — treat as non-floating only if
            # it is clearly complex, else flag as unexpected.
            if "complex" in str(dtype):
                non_floating += 1
            else:
                unexpected.append(label)
                unexpected_names.append(tensor_name)
            continue

        if is_param:
            fp_params += 1
        else:
            fp_buffers += 1

        if override is not None:
            allowed_records.append(
                {
                    "exception_id": override.get("exception_id", "static_override"),
                    "component": name,
                    "tensor_name": tensor_name,
                    "dtype": label,
                    "reason": override.get("reason", "documented static override"),
                }
            )
            continue

        exception = exceptions.find(name, tensor_name, label)
        if exception is not None:
            allowed_records.append(exception.to_dict())
            continue

        if not policy.is_known_float(dtype):
            unexpected.append(f"{label}:unknown_float")
            unexpected_names.append(tensor_name)
            continue

        if classification.requires_bf16 and not policy.is_bfloat16(dtype):
            unexpected.append(label)
            unexpected_names.append(tensor_name)

    status = "PASS" if not unexpected else "FAIL"
    return AuditResult(
        module=name,
        parameter_count=len(param_names),
        buffer_count=len(buffer_names),
        float_parameter_count=fp_params,
        float_buffer_count=fp_buffers,
        dtype_histogram=histogram,
        non_floating_count=non_floating,
        unexpected_dtype_list=sorted(set(unexpected)),
        unexpected_names=sorted(set(unexpected_names)),
        allowed_exceptions=allowed_records,
        classification=str(classification),
        status=status,
    )


def audit_components(
    components: Dict[str, Any],
    contract: Optional[Dict[str, str]] = None,
    exceptions: Optional[ExceptionRegistry] = None,
) -> Dict[str, Any]:
    """Audit a dict of ``{component_name: nn.Module}`` and summarize."""
    contract = contract or policy.DEFAULT_PRECISION_CONTRACT
    results: Dict[str, Any] = {}
    statuses: List[str] = []
    for name, module in components.items():
        cls = policy.classify_component(name, contract)
        results[name] = audit_module(module, name, cls, exceptions).to_dict()
        statuses.append(results[name]["status"])
    overall = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    return {
        "overall_status": overall,
        "components": results,
    }


# ---------------------------------------------------------------------------
# Corrective A4.2 — 3-component runtime dtype audit schema
#
# The G3/G4 acceptance reducer requires ``dtype_after`` to encode text_encoder,
# transformer and vae separately. The real backend must never collapse the audit
# to a single transformer-only report.
# ---------------------------------------------------------------------------

AUTHORITY_DTYPE_COMPONENTS = ("text_encoder", "transformer", "vae")


def _dtype_histogram(tensors: Any) -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for _name, tensor in tensors:
        label = _dtype_label(tensor.dtype)
        histogram[label] = histogram.get(label, 0) + 1
    return histogram


def audit_component(
    module,
    name: str,
    classification,
    exceptions: Optional[ExceptionRegistry] = None,
) -> Dict[str, Any]:
    """Audit one component into the A4 3-component dtype schema.

    Reuses the authoritative ``audit_module`` histogram/exception logic and adds
    parameter-vs-buffer dtype histograms and the unexpected floating list.
    """
    if module is None:
        return {
            "status": "FAIL",
            "parameter_dtype_histogram": {},
            "buffer_dtype_histogram": {},
            "unexpected_floating_dtypes": [],
            "component": name,
            "note": "missing component; authority requires all three",
        }
    exceptions = exceptions or policy.declare_exceptions()
    result = audit_module(module, name, classification, exceptions=exceptions)
    param_hist = _dtype_histogram(module.named_parameters(recurse=True))
    buffer_hist = _dtype_histogram(module.named_buffers(recurse=True))
    return {
        "status": result.status,
        "parameter_dtype_histogram": param_hist,
        "buffer_dtype_histogram": buffer_hist,
        "unexpected_floating_dtypes": sorted(set(result.unexpected_dtype_list)),
        "component": name,
        "classification": result.classification,
        "float_parameter_count": result.float_parameter_count,
        "float_buffer_count": result.float_buffer_count,
        "allowed_exceptions": result.allowed_exceptions,
    }


def audit_runtime_dtypes(state: Any, exceptions: Optional[ExceptionRegistry] = None) -> Dict[str, Any]:
    """Audit the three authority components on a live runtime state.

    ``state`` must expose ``text_encoder``, ``transformer`` and ``vae``. Missing
    components fail closed (never PASS). Output schema matches the A4 contract.
    """
    exceptions = exceptions or policy.declare_exceptions()
    report: Dict[str, Any] = {}
    for name in AUTHORITY_DTYPE_COMPONENTS:
        module = getattr(state, name, None)
        if module is None:
            report[name] = audit_component(None, name, classification=None, exceptions=exceptions)
            continue
        cls = policy.classify_component(name, policy.DEFAULT_PRECISION_CONTRACT)
        report[name] = audit_component(module, name, cls, exceptions=exceptions)
    report["overall_status"] = (
        "PASS" if all(comp["status"] == "PASS" for comp in report.values()) else "FAIL"
    )
    report["authority_components"] = list(AUTHORITY_DTYPE_COMPONENTS)
    return report
