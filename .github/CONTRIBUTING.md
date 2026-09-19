# Contributing

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for contributing to this Mage-Flow-Turbo PyTorch BF16 dual-T4 engineering project.

## Principles

Changes should remain:

- focused and reviewable;
- reproducible;
- explicit about whether they affect closed qualified authority;
- safe to validate without loading the model unless GPU behavior is the subject of the change.

Do not weaken a fail-closed gate only to make a demo pass.

## Development validation

For CPU/static changes, run at least:

```bash
python -m compileall -q mage_t4x2 public_demo scripts
python -m pytest -q
```

For public-demo-focused work:

```bash
python -m pytest -q \
  tests/test_public_bootstrap.py \
  tests/test_public_contract.py \
  tests/test_public_demo_gates.py \
  tests/test_public_runner_static.py \
  tests/test_public_docs_contract.py
```

CPU/static PASS is not evidence of a successful live dual-T4 inference.

## Bilingual documentation policy

Every project Markdown document should have an English/Vietnamese pair in the same directory:

- `name.md`
- `name.vi.md`

A documentation commit should change both sides together. Each pair must contain a language switcher near the top.

## Authority-sensitive changes

Files listed in `authority/runtime-baseline.json` are byte-qualified runtime authority files. If a change modifies one of those files, explicitly state that byte-equivalence with the closed qualified authority is no longer preserved and determine whether requalification is required.

## Pull requests

Describe:

- what changed and why;
- validation commands actually run;
- whether the change is CPU/static only or includes new T4x2 evidence;
- effect on source/model/dependency provenance;
- effect on public notebook or release workflow;
- compatibility or security impact.

Do not include model weights, secrets, Kaggle tokens, runtime caches, generated evidence archives, or personal credentials in Git.
