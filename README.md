# Mage-Flow-Turbo PyTorch BF16 on T4×2 GPU

<p align="center">
  <a href="https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-upstream-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-Tesla%20T4%20%C3%972-76B900?logo=nvidia&logoColor=white">
  <img alt="Precision" src="https://img.shields.io/badge/Precision-BF16%20materialization-2563EB">
  <img alt="Attention" src="https://img.shields.io/badge/Attention-SDPA-7C3AED">
  <img alt="Authority" src="https://img.shields.io/badge/Runtime-QUALIFIED-0A7E07">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-T4%20%C3%972-20BEFF?logo=kaggle&logoColor=white">
</p>

> 🌐 Language: **English** | [Tiếng Việt](README.vi.md)

A reproducible engineering project for running **Mage-Flow-Turbo** through the upstream **PyTorch** path on **Kaggle Tesla T4 ×2**, while preserving **one logical model load** and **one text-to-image trajectory across both GPUs** (Lane A — canonical qualification) and additionally running a **separate sustained multi-image production showcase lane** (Lane B) on its own single model load.

This is an independent engineering project around the upstream Microsoft Mage model/runtime. It is **not an official Microsoft release**.

---

## At a glance

| Area | Qualified public profile |
|---|---|
| Runtime | Upstream Mage / PyTorch |
| Accelerator | Exactly **2 × NVIDIA Tesla T4** |
| Precision | **BF16 dtype/materialization** |
| Attention | **SDPA** |
| Inference shape | One logical T2I trajectory |
| Model loads | Exactly **1** |
| Transformer split | Block 0 on `cuda:0`; blocks 1–11 on `cuda:1` |
| VAE | `cuda:1` |
| Demo resolution | `512×512` RGB (Lane A) |
| Steps / CFG / Seed | `4` / `1.0` / `42` |
| Showcase lane | Sustained multi-image (12–24 images, `512/768/1024`, seeds `1001+`) |
| Showcase verdict | `PUBLIC_SHOWCASE_FINAL_VERDICT` (fail-closed) |
| CPU fallback | Forbidden by the canonical contract |
| Quantization | None |
| sd.cpp / sd-cli | Not used |
| Runtime authority | **qualified CLOSED_PASS** |
| License | MIT for this repository's original code/docs |

The supported precision wording is:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

This project does **not** claim native BF16 Tensor Core acceleration on Tesla T4.

---

## Why this project exists

Mage-Flow-Turbo can be straightforward to load on a large accelerator, but a reproducible **single-trajectory dual-T4 execution path** is a different engineering problem.

This repository focuses on that problem explicitly:

- keep a **single logical model instance**;
- keep a **single prompt / latent / scheduler trajectory**;
- split the transformer across both T4 GPUs instead of running two independent jobs;
- make cross-device transfers observable;
- fail closed when hardware, source identity, attention backend, routing, precision, or output does not match the contract;
- preserve a byte-verifiable qualified runtime baseline;
- provide a public Kaggle workflow that does not depend on private ZIP bundles or hidden setup steps.

The goal is not merely to generate an image. The goal is to make the execution path **inspectable, reproducible, and reviewable**.

---

## What this repository demonstrates

- Upstream PyTorch execution, not `stable-diffusion.cpp` / `sd-cli`.
- BF16 dtype/materialization on the text encoder, transformer, and VAE.
- Exactly one model load per lane (two independent loads total: one for Lane A, one for Lane B — never shared).
- Exactly one logical T2I trajectory distributed across two T4 GPUs in the canonical Lane A.
- A second, sustained production showcase lane (Lane B) that keeps its model hot across a deterministic multi-prompt, multi-seed, multi-resolution workload (`512 / 768 / 1024`) with per-image timing, GPU allocated/reserved peaks, routing and SHA-256 output evidence, plus a labeled contact-sheet gallery.
- Transformer block 0 on `cuda:0`; blocks 1–11 and the output head on `cuda:1`.
- Explicit `cuda:0 -> cuda:1` transformer boundary transfers.
- Explicit `cuda:1 -> cuda:0` transformer result returns.
- VAE input transfer from `cuda:0` to `cuda:1`.
- SDPA attention with no `flash_attn` dependency for the T4 path.
- Fail-closed provenance, placement, routing, precision, and output checks.
- A public notebook that bootstraps source from Git rather than internal publication archives.
- A fail-closed showcase lane whose pretty gallery can never override a failed canonical qualification.
- CPU/static CI that validates contracts without pretending to replace live T4×2 evidence.

---

## End-to-end execution flow

```text
Fresh Kaggle Notebook
        |
        |  T4 ×2 + Internet ON
        v
Clone this repository
        |
        +--> verify public source ref
        |
        +--> verify frozen qualified runtime baseline
        |
        +--> fetch exact pinned upstream Mage commit/tree
        |
        +--> verify pinned bootstrap wheel
        |
        v
Resolve owner-qualified Kaggle model mount
        |
        v
Freeze SDPA before model construction
        |
        v
Load ONE model instance (Lane A)
        |
        v
Apply explicit dual-device placement
        |
        +--> text encoder ---------------- cuda:0
        +--> transformer block 0 --------- cuda:0
        +--> transformer blocks 1..11 ---- cuda:1
        +--> norm_out / proj_out --------- cuda:1
        +--> VAE ------------------------- cuda:1
        |
        v
Reassert SDPA
        |
        v
Run one 4-step T2I trajectory
        |
        v
Reduce telemetry + dtype + output evidence
        |
        v
Lane A PASS / FAIL with preserved evidence
        |
        v
LANE B  (production showcase)
        |
        v
Resolve showcase model path (owner-qualified only)
        |
        v
Load ONE more model instance (independent of Lane A)
        |
        v
Apply explicit dual-device placement (hot model)
        |
        v
Run deterministic multi-image schedule (512/768/1024)
        |
        v
Per-image telemetry + GPU peaks + SHA-256 + gallery
        |
        v
Lane B PASS / FAIL with preserved evidence
```

The showcase lane runs **after** and **independently of** the canonical Lane A; it can never rewrite Lane A evidence.

---

## Runtime topology

```text
Prompt
  |
Text encoder ------------------------ cuda:0
  |
Transformer block 0 ---------------- cuda:0
  |
activation transfer 0 -> 1
  |
Transformer blocks 1..11 ----------- cuda:1
norm_out / proj_out ----------------- cuda:1
  |
transformer result 1 -> 0
  |
latent / scheduler path ------------- cuda:0
  |
VAE input 0 -> 1
  |
VAE --------------------------------- cuda:1
  |
512×512 RGB image
```

For the canonical four-step demo (Lane A), the transformer sequence `[0..11]` is expected **four times**, once per denoising invocation. Lane B reuses the same explicit dual-T4 placement while the model stays hot across the whole showcase schedule.

---

## Quick start on Kaggle

The public notebook is designed so users do **not** need any internal ZIP, identity JSON, or private source bundle.

### 1. Create a fresh Kaggle Notebook

Configure:

- Accelerator: **GPU T4 ×2**
- Internet: **ON**
- Start from a fresh kernel.

### 2. Attach the required Kaggle Model

Attach:

```text
dangkhoa2016/mage-flow-community-mage-flow-turbo
```

Expected read-only mount:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

No additional dataset is required for the canonical text-to-image demo.

### 3. Open the public notebook

```text
notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb
```

The notebook clones this repository into `/kaggle/working`, verifies source identity, bootstraps the exact pinned upstream Mage source and dependency, resolves the Kaggle model mount, then executes the two lanes: the canonical **Lane A** qualification and the sustained **Lane B** production showcase.

Use **Run All exactly once** for a publication-quality run.

See [Kaggle public demo guide](docs/kaggle-public-demo.md) and [Kaggle setup notes](docs/kaggle.md).

---

## Python and CLI entry points

### Python

```python
from public_demo.runner import run_public_demo

result = run_public_demo(
    project_root="/path/to/repository",
)

print(result["status"])
print(result["summary_path"])
```

The canonical runner (`public_demo/runner.py`) intentionally accepts no prompt/profile overrides. Prompt, seed, steps, CFG, resolution, block/split topology, SDPA attention, BF16 dtype, model identity and the owner-qualified model path are all fixed by the canonical public contract in `public_demo/contract.py`; non-canonical inputs fail closed.

The showcase runner (`public_demo/showcase.py`) is the Lane B counterpart: it also accepts no overrides, performs its own single model load, and executes the deterministic showcase table (`s01..s24`, `512/768/1024`, seeds `1001..1024`) with fail-closed per-image and manifest evidence.

```python
from public_demo.showcase import run_public_showcase

showcase = run_public_showcase(project_root="/path/to/repository")
print(showcase["status"])      # PASS / FAIL
print(showcase["summary_path"])  # artifacts/public-showcase/<run_id>/showcase-summary.json
```

### CLI

```bash
python scripts/public_kaggle_demo.py \
  --project-root /path/to/repository \
  --output-root /path/to/output
```

The CLI prints ordered gate verdicts, the run ID, final status, and summary path.

See [Python and CLI API](docs/api.md) and [Usage](docs/usage.md).

---

## Lane A — canonical demo configuration

| Setting | Value |
|---|---|
| Model | `dangkhoa2016/mage-flow-community-mage-flow-turbo` |
| Resolution | `512×512` |
| Steps | `4` |
| CFG | `1.0` |
| Seed | `42` |
| Attention | `sdpa` |
| Precision | BF16 dtype/materialization |
| GPU | Tesla T4 ×2 |
| `split_block` | `1` |
| Transformer blocks | `12` |
| CPU fallback | disabled |
| Model load count | `1` |

## Lane B — showcase configuration

The showcase is its own lane with its own single model load and a deterministic, repository-controlled workload:

| Setting | Value |
|---|---|
| Cases | `s01..s24` (12 mandatory + 12 extension) |
| Mandatory minimum | `12` |
| Maximum | `24` |
| Resolutions | `512`, `768`, `1024` |
| Steps / CFG | `4` / `1.0` |
| Seeds | `1001..1024` |
| Target inference window | `300 s` |
| Outputs | `outputs/<case>-<res>x<res>-seed<seed>.png`, SHA-256 each |
| Telemetry | `telemetry/<case>.jsonl` |
| Summary | `showcase-summary.json`, `case-results.json`, `showcase-runtime.log`, manifest under `artifacts/public-showcase/<run_id>/` |
| Final key | `PUBLIC_SHOWCASE_FINAL_VERDICT` |

---

## Lane A acceptance matrix

The closed qualified authority established the following core facts:

| Acceptance item | Qualified status |
|---|---|
| qualified runtime authority | `CLOSED_PASS` |
| T4 ×2 preflight | PASS |
| Single model load | `MODEL_LOAD_COUNT=1` |
| Single T2I trajectory | PROVEN |
| GPU0 participation | PROVEN |
| GPU1 participation | PROVEN |
| Transformer invocations | 4 / 4 |
| 0 → 1 transformer boundary transfers | 4 / 4 |
| 1 → 0 transformer returns | 4 / 4 |
| VAE input 0 → 1 transfer | 1 / 1 |
| BF16 materialization | PASS |
| CPU fallback | NOT_OBSERVED |
| Output validation | PASS |
| Output profile | 512×512 RGB |

The public notebook is a reproducible demonstration surface; it does **not** rewrite the frozen authority evidence.

### Lane B acceptance summary

The showcase lane must satisfy:

```text
SHOWCASE_MODEL_LOAD_COUNT=1
ROUTING=PASS           (block order, participation, single lane instance, no CPU fallback)
MIN_CASES=PASS         (>= 12 executed cases)
OUTPUT_VALIDATION=PASS (every image valid, adapter detached after each case)
GALLERY=PASS
DIGEST_VERIFY=PASS     (every recorded SHA-256 matches the artifact)
MANIFEST_BUILD=PASS
PUBLIC_SHOWCASE_FINAL_VERDICT=PASS | FAIL | NOT_RUN
```

---

## Reproducibility chain

The project pins and verifies the identities that matter to the qualified path:

- Model ID: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- Model revision: `65bb3500f0da9df6a41ec6383716fc02cf014773`
- Upstream Mage repository: `https://github.com/microsoft/Mage`
- Upstream Mage commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- Upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`
- qualified runtime baseline manifest SHA-256: `b829fdea0b568ff80ff74d1c58344d7f289d781d42029f2b42a34a8d9c266466`
- Bootstrap dependency: `loguru==0.7.3`
- Bootstrap wheel SHA-256: `31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c`

The runtime baseline records byte size + SHA-256 for qualified runtime files. The public runner verifies that baseline before live inference.

See [Provenance and authority](docs/provenance.md).

---

## Safety and fail-closed behavior

The public path stops rather than silently weakening the contract when, among other cases:

- the project checkout is stale or does not match the selected source ref;
- upstream Mage commit/tree identity differs from the pin;
- the bootstrap wheel differs in size or SHA-256;
- the qualified runtime baseline does not match;
- exactly two Tesla T4 GPUs are not available;
- the owner-qualified Kaggle model attachment is missing;
- SDPA is not effective after model construction or before inference;
- model load count differs from one;
- block order or transfer cardinality differs from the contract;
- one of the GPUs does not participate as expected;
- CPU fallback is observed;
- BF16 live materialization does not match the qualified profile;
- the generated output is not a valid 512×512 RGB image;
- the showcase runs fewer than the mandatory case minimum, any showcase image fails validation, the routing facts are not all green, a recorded output digest mismatches, or the showcase manifest cannot be built.

This design intentionally prefers an explicit FAIL over a misleading PASS.

---

## Repository layout

```text
.
├── mage_t4x2/        qualified runtime, routing, placement and evidence modules
├── public_demo/      public Git/PyPI bootstrap, reproducible demo runner and showcase lane
├── scripts/          CLI, session orchestration and authority drivers
├── authority/        frozen runtime baselines and provenance manifests
├── notebooks/        bilingual public Kaggle production demo
├── docs/             architecture, usage, API, Kaggle, limits and provenance
├── tests/            CPU/static contract and public workflow tests
├── .github/          CI, issue forms, security, support and contribution policy
└── LICENSE           MIT license for this repository's original work
```

---

## Documentation

| Topic | English |
|---|---|
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Usage | [docs/usage.md](docs/usage.md) |
| Python & CLI API | [docs/api.md](docs/api.md) |
| Kaggle setup | [docs/kaggle.md](docs/kaggle.md) |
| Kaggle public demo | [docs/kaggle-public-demo.md](docs/kaggle-public-demo.md) |
| Limitations | [docs/limitations.md](docs/limitations.md) |
| Provenance / authority | [docs/provenance.md](docs/provenance.md) |
| Testing / validation | [docs/testing.md](docs/testing.md) |
| Contributing | [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) |
| Security | [.github/SECURITY.md](.github/SECURITY.md) |
| Support | [.github/SUPPORT.md](.github/SUPPORT.md) |

Every project Markdown guide has a corresponding Vietnamese version.

---

## Scope and non-goals

This repository is intentionally narrow.

It does **not** claim:

- support for arbitrary GPU topologies;
- qualification on P100, TPU, CPU-only, or unrelated GPUs;
- general compatibility with every Mage model/revision;
- a hosted REST service or production SLA;
- multi-user serving, autoscaling, or authentication;
- native BF16 Tensor Core acceleration on Tesla T4;
- that CPU/static CI proves real dual-T4 execution.

For the exact boundaries, see [Limitations](docs/limitations.md).

---

## Testing

CPU/static validation:

```bash
python -m compileall -q mage_t4x2 public_demo scripts
python -m pytest -q
```

CI validates static/public contracts across supported Python versions, but real GPU claims remain tied to real **Tesla T4 ×2** evidence.

See [Testing and validation](docs/testing.md).

---

## Contributing, security and support

Contributions are welcome when they preserve clear provenance and explicitly state whether a change affects qualified runtime bytes.

- [Contributing guide](.github/CONTRIBUTING.md)
- [Code of Conduct](.github/CODE_OF_CONDUCT.md)
- [Security policy](.github/SECURITY.md)
- [Support policy](.github/SUPPORT.md)

Security-sensitive findings should follow the private reporting path in the security policy rather than a public issue.

---

## License and upstream code

This repository's original code and documentation are licensed under the [MIT License](LICENSE).

Copyright (c) 2026 **Đăng Khoa <i.am@dangkhoa.dev>**.

Upstream Mage source is fetched from its pinned public Git commit rather than copied into this repository. The upstream source and model artifacts remain subject to their own licenses and terms; review them before redistribution or production use.
