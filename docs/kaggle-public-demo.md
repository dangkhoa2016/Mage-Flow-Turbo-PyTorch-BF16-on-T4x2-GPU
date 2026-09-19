# Kaggle public demo guide

> 🌐 Language: **English** | [Tiếng Việt](kaggle-public-demo.vi.md)

This guide describes the public, reproducible Kaggle path for the Mage-Flow-Turbo PyTorch BF16 dual-T4 demo, which runs **two independent lanes**: Lane A (canonical qualification) and Lane B (production showcase).

## Requirements

- Kaggle account with **Tesla T4 ×2** available.
- Notebook Internet access enabled.
- Kaggle Model attachment: `dangkhoa2016/mage-flow-community-mage-flow-turbo`.
- No additional dataset is required.

## Kaggle setup

1. Create a new Kaggle Notebook.
2. Open **Settings → Accelerator** and select **GPU T4 ×2**.
3. Enable **Internet**.
4. Use **Add Input → Models** and attach:
   `dangkhoa2016/mage-flow-community-mage-flow-turbo`.
5. Confirm the model is visible at:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

Do not manually copy the model into `/kaggle/working`.

## Source bootstrap

The notebook clones:

```text
https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU.git
```

and resolves the committed `SOURCE_REF` (default `v1.0.0`), which may be a release tag or an exact 40-character commit SHA, into:

```text
/kaggle/working/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU
```

The notebook verifies the Git origin, resolves the ref (tag or SHA), performs a detached checkout of the resolved commit, and records `HEAD`. No publication ZIP or identity JSON is needed.

## Runtime bootstrap

The public runner then:

1. Verifies `authority/runtime-baseline.json`.
2. Freezes the SDPA contract before upstream import.
3. Clones Microsoft Mage at the exact pinned commit.
4. Verifies the `mage_flow` Git tree identity.
5. Downloads `loguru==0.7.3` as a wheel.
6. Verifies the wheel byte size and SHA-256.
7. Creates a fresh local bootstrap site.
8. Verifies bootstrap dependency version and import origin.
9. Imports the pinned upstream Mage source.
10. Requires exactly two Tesla T4 GPUs.

## Lane A — canonical qualification (acceptance authority)

The canonical runner (`public_demo.runner.run_public_demo`) accepts **no** prompt/seed/resolution overrides and proves the frozen acceptance contract:

```text
Prompt: a red fox in a snowy forest at golden hour, high detail
Seed: 42
Steps: 4
CFG: 1.0
Resolution: 512x512
split_block: 1
num_blocks: 12
```

Expected PASS facts:

```text
GPU_COUNT=2
MODEL_LOAD_COUNT=1
GPU0_PARTICIPATION=PASS
GPU1_PARTICIPATION=PASS
EXPECTED_TRANSFORMER_INVOCATIONS=4
OBSERVED_TRANSFORMER_INVOCATIONS=4
TRANSFER_0_TO_1_COUNT=4
TRANSFORMER_RETURN_1_TO_0_COUNT=4
VAE_INPUT_TRANSFER_COUNT=1
BLOCK_ORDER_VALID=PASS
NO_SKIPPED_BLOCKS=PASS
NO_DUPLICATED_BLOCKS_WITHIN_INVOCATION=PASS
TEXT_ENCODER_BF16=PASS
TRANSFORMER_BF16=PASS
VAE_BF16=PASS
CPU_FALLBACK_OBSERVED=NO
OUTPUT_512X512=PASS
OUTPUT_RGB=PASS
PUBLIC_DEMO_FINAL_VERDICT=PASS
```

## Lane B — production showcase (sustained multi-image demo)

`public_demo.showcase.run_public_showcase` performs its **own single model load** (never sharing Lane A), applies the same explicit dual-T4 placement, keeps the model hot across the deterministic case table (`s01..s24`), and writes per-image evidence:

- seed-qualified outputs `outputs/<case>-<res>x<res>-seed<seed>.png`;
- telemetry per case under `telemetry/<case>.jsonl`;
- GPU allocated + reserved peaks per image on both T4s;
- routing facts reduced from telemetry (block order, participation, single lane instance, no CPU fallback);
- `showcase-summary.json`, `case-results.json`, `showcase-runtime.log` and a manifest, all under:
  `artifacts/public-showcase/<run_id>/`.

> The notebook's primary per-image presentation shows every generated PNG independently and prints the exact deterministic prompt immediately above its corresponding image, together with case id, category, resolution, seed, inference time and SHA-256. The compact prompt map remains available as an audit view.

Showcase constants (fixed by contract):

```text
resolutions: 512 / 768 / 1024
steps: 4
CFG: 1.0
mandatory minimum cases: 12
max cases: 24
target inference window: 300 s
verdict key: PUBLIC_SHOWCASE_FINAL_VERDICT
```

## Expected final verdicts

```text
PUBLIC_DEMO_FINAL_VERDICT=PASS
PUBLIC_SHOWCASE_FINAL_VERDICT=PASS
```

A pretty showcase can never override a failed canonical qualification.

## Publication discipline

For a clean public proof:

- start from a fresh Kaggle kernel;
- run all cells in order exactly once;
- do not manually preload the model;
- do not delete stale state and rerun inside the same proof attempt;
- if a cell fails, preserve that failed attempt and diagnose from `first_failed_gate`, `error`, `summary.json`, and `runtime.log`.

## Outputs

Runtime artifacts are written under `/kaggle/working` inside the cloned project/output tree. The Kaggle model remains read-only under `/kaggle/input`.