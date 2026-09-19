# Testing and Validation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](testing.vi.md)

## Two validation classes

This repository distinguishes CPU/static validation from real GPU acceptance.

### CPU/static validation

CPU-safe tests can validate:

- source/bootstrap contracts;
- path and provenance rules;
- qualified manifest byte integrity;
- routing/telemetry reducers with fixtures;
- notebook structure;
- documentation contracts;
- fail-closed behavior that does not require loading the model.

These checks must not be described as proof that a live dual-T4 inference succeeded.

### Real GPU validation

Claims about actual two-GPU participation, model placement, cross-device transfers, BF16 live materialization, no CPU fallback, and generated image output require a real **Tesla T4 x2** session.

## Useful local commands

At minimum:

```bash
python -m compileall -q mage_t4x2 public_demo scripts
python -m pytest -q
```

Focused public-demo tests:

```bash
python -m pytest -q \
  tests/test_public_bootstrap.py \
  tests/test_public_contract.py \
  tests/test_public_demo_gates.py \
  tests/test_public_runner_static.py \
  tests/test_public_docs_contract.py
```

## Authority integrity

The public runner itself verifies `authority/runtime-baseline.json` before proceeding.

For independent inspection, call:

```python
from mage_t4x2.runtime_baseline import verify_runtime_baseline
print(verify_runtime_baseline())
```

Expected qualified state is `status == "PASS"` with no missing required files or hash/size mismatches.

## Publication validation

For a final public Kaggle artifact:

- start from a fresh notebook/kernel;
- run all cells once;
- avoid cleanup/retry cells that mutate failed state;
- preserve all outputs;
- preserve the evidence directory;
- record the Git commit/tag used by the notebook.

A technically successful run and a pristine publication artifact are related but distinct acceptance questions.
