# Mage-Flow-Turbo PyTorch BF16 on T4×2 GPU

> 🌐 Language: **English** | [Tiếng Việt](README.vi.md)

A reproducible engineering project for running **Mage-Flow-Turbo** through the upstream PyTorch path on **Kaggle Tesla T4 ×2**, using one logical text-to-image trajectory across both GPUs.

This is an independent engineering project around the upstream Microsoft Mage model/runtime. It is **not an official Microsoft release**.

## What this repository demonstrates

- Upstream PyTorch execution, not `stable-diffusion.cpp` / `sd-cli`.
- BF16 dtype/materialization on the text encoder, transformer, and VAE.
- Exactly one logical model load.
- Exactly one T2I trajectory across both Tesla T4 GPUs.
- Transformer block 0 on `cuda:0`; blocks 1–11 and the output head on `cuda:1`.
- Explicit `cuda:0 -> cuda:1` boundary transfers and `cuda:1 -> cuda:0` transformer returns.
- VAE input transfer from `cuda:0` to `cuda:1`.
- SDPA attention; no `flash_attn` dependency for the T4 path.
- Fail-closed evidence and routing checks.

The supported precision wording is:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

This project does not claim native BF16 Tensor Core acceleration on Tesla T4.

## Quick start on Kaggle

The public notebook is designed so users do **not** need any internal ZIP, identity JSON, or source bundle.

### 1. Create a fresh Kaggle Notebook

Configure:

- Accelerator: **GPU T4 ×2**
- Internet: **ON**

### 2. Attach the required Kaggle Model

Add:

`dangkhoa2016/mage-flow-community-mage-flow-turbo`

Expected read-only model path:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

No additional dataset is required for the text-to-image demo.

### 3. Open the public notebook

Repository notebook:

```text
notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb
```

The notebook automatically clones this repository at release tag `v1.0.0` into `/kaggle/working`, verifies the source ref, fetches the exact pinned upstream Mage source, verifies the bootstrap wheel, and then runs the demo.

Use a fresh kernel and **Run All exactly once**.

See [Kaggle public demo guide](docs/kaggle-public-demo.md) for the complete procedure.

## Runtime topology

```text
Prompt
  |
Text encoder ------------------------ cuda:0
  |
Transformer block 0 ---------------- cuda:0
  |
activation 0 -> 1
  |
Transformer blocks 1..11 ----------- cuda:1
norm_out / proj_out ----------------- cuda:1
  |
transformer output 1 -> 0
  |
latent / scheduler path ------------- cuda:0
  |
VAE input 0 -> 1
  |
VAE --------------------------------- cuda:1
  |
512×512 RGB image
```

For the canonical four-step demo, the transformer sequence `[0..11]` is expected four times—once per denoising invocation.

## Canonical demo configuration

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

## Pinned provenance

- Model revision: `65bb3500f0da9df6a41ec6383716fc02cf014773`
- Upstream Mage commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- Upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`
- R2G runtime baseline manifest SHA-256: `b829fdea0b568ff80ff74d1c58344d7f289d781d42029f2b42a34a8d9c266466`
- Bootstrap dependency: `loguru==0.7.3`
- Bootstrap wheel SHA-256: `31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c`

## R2G authority status

The closed R2G authority run established:

```text
R2G_GPU_RUNTIME_AUTHORITY=CLOSED_PASS
MODEL_LOAD_COUNT=1
DUAL_T4_SINGLE_INFERENCE_TRAJECTORY=PROVEN
GPU0_PARTICIPATION=PROVEN
GPU1_PARTICIPATION=PROVEN
BF16_MATERIALIZATION=PASS
CPU_FALLBACK=NOT_OBSERVED
OUTPUT_VALID=PASS
```

The public notebook is a reproducible demonstration surface; it does not rewrite the frozen authority evidence.

## Repository layout

```text
mage_t4x2/                 qualified runtime modules
public_demo/               public Git/PyPI bootstrap + demo runner
scripts/                   CLI and authority drivers
authority/                 runtime baseline and provenance manifests
notebooks/                 public Kaggle notebook
docs/                      bilingual usage documentation
tests/                     CPU/static contract tests
```

## Safety and failure behavior

The public path fails closed when, among other cases:

- the source checkout is stale or does not match the release ref;
- upstream Mage commit/tree identity differs from the pin;
- the bootstrap wheel differs in size or SHA-256;
- the R2G runtime baseline does not match;
- two Tesla T4 GPUs are not available;
- the owner-qualified Kaggle model attachment is missing;
- SDPA is not effective after model construction or before inference;
- model load count differs from one;
- block order, transfer cardinality, or GPU participation differs from the contract;
- CPU fallback is observed;
- the generated output is not a valid 512×512 RGB image.

## License and upstream code

Upstream Mage source is fetched from its pinned public Git commit at runtime rather than copied into this repository. Review the upstream repository and model licenses before redistribution or production use.
