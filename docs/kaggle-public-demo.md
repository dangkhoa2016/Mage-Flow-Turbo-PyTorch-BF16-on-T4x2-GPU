# Kaggle public demo guide

> 🌐 Language: **English** | [Tiếng Việt](kaggle-public-demo.vi.md)

This guide describes the public, reproducible Kaggle path for the Mage-Flow-Turbo PyTorch BF16 dual-T4 demo.

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

at release ref:

```text
v1.0.0
```

into:

```text
/kaggle/working/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU
```

The checkout target must not already exist. The notebook verifies the Git origin and exact release tag and records the resolved commit SHA.

No publication ZIP or identity JSON is needed.

## Runtime bootstrap

The public runner then:

1. Verifies `authority/r2g-runtime-baseline.json`.
2. Freezes the SDPA contract before upstream import.
3. Clones Microsoft Mage at the exact pinned commit.
4. Verifies the `mage_flow` Git tree identity.
5. Downloads `loguru==0.7.3` as a wheel.
6. Verifies the wheel byte size and SHA-256.
7. Creates a fresh local bootstrap site.
8. Verifies bootstrap dependency version and import origin.
9. Imports the pinned upstream Mage source.
10. Requires exactly two Tesla T4 GPUs.
11. Resolves only the owner-qualified local Kaggle model path.
12. Loads the model exactly once.
13. Reasserts SDPA after model construction.
14. Applies explicit dual-T4 placement.
15. Reasserts SDPA immediately before inference.
16. Runs one T2I trajectory and derives acceptance from live telemetry.

## Canonical run

```text
Prompt: a red fox in a snowy forest at golden hour, high detail
Seed: 42
Steps: 4
CFG: 1.0
Resolution: 512x512
split_block: 1
num_blocks: 12
```

## Expected PASS facts

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

## Publication discipline

For a clean public proof:

- start from a fresh Kaggle kernel;
- run all cells in order exactly once;
- do not manually preload the model;
- do not delete stale state and rerun inside the same proof attempt;
- if a cell fails, preserve that failed attempt and diagnose from `first_failed_gate`, `error`, `summary.json`, and `runtime.log`.

## Outputs

Runtime artifacts are written under `/kaggle/working` inside the cloned project/output tree. The Kaggle model remains read-only under `/kaggle/input`.
