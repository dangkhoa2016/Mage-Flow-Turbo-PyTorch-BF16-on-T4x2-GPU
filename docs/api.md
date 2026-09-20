# Python and CLI API

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](api.vi.md)

This project does not expose a REST service. Its supported public interface is the Python public-demo runner plus the small CLI wrapper.

## Python API

Import:

```python
from public_demo.runner import run_public_demo
```

Signature:

```python
run_public_demo(
    project_root=None,
    model_path=None,
    prompt=CANONICAL_PROMPT,
    seed=None,
    steps=None,
    width=None,
    height=None,
    cfg_scale=None,
    num_blocks=12,
    split_block=1,
    output_root=None,
    run_id=None,
)
```

### Important behavior

- `project_root` must contain `mage_t4x2/` and `scripts/`.
- `prompt` must equal the canonical prompt or the runner fails closed.
- If `model_path` is not supplied, the runner resolves only the owner-qualified Kaggle path under `/kaggle/input/models/<owner>/<model>/pytorch/default/1`.
- The runner verifies the R2G baseline before live execution.
- SDPA is asserted before model execution, reasserted after model construction, and checked again immediately before inference.
- The live path requires exactly two Tesla T4 GPUs.
- The model load count is expected to remain one.
- CPU fallback, `stable-diffusion.cpp`, and `sd-cli` are outside the public contract.

### Return value

The function returns a dictionary. Stable top-level fields used by the notebook/CLI include:

| Field | Meaning |
|---|---|
| `status` | `PASS`, `FAIL`, or intermediate state while running |
| `run_id` | evidence run identifier |
| `first_failed_gate` | first fail-closed gate, or `None` on PASS |
| `error` | failure text, or `None` on PASS |
| `summary_path` | generated summary JSON |
| `verdict_lines` | ordered human-readable gate lines |

The summary also contains reduced hardware, routing, placement, precision, safety, and output evidence when live inference reaches those stages.

## CLI API

```bash
python scripts/public_kaggle_demo.py   [--project-root PATH]   [--model-path PATH]   [--output-root PATH]
```

Exit status:

- `0`: the runner did not finish with final `FAIL`.
- `1`: final status is `FAIL`.

The CLI intentionally does not expose prompt/seed/steps/CFG/resolution switches. Those are governed by the canonical public run contract.

## Internal interfaces

Modules under `mage_t4x2/` are qualification/runtime building blocks. They are public source code, but this repository does not promise them as a versioned general-purpose library API.
