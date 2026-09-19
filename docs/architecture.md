# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](architecture.vi.md)

## Purpose

This repository runs Mage-Flow-Turbo through the upstream Mage/PyTorch path on exactly two Tesla T4 GPUs while preserving one logical model load and one text-to-image trajectory.

## Runtime layers

```text
Kaggle notebook
  |
  +-- public_demo/bootstrap.py
  |     +-- verifies the pinned upstream Mage commit/tree
  |     +-- downloads and verifies the pinned loguru wheel
  |     +-- prepares an isolated bootstrap site
  |
  +-- public_demo/runner.py
  |     +-- verifies the frozen qualified runtime baseline
  |     +-- freezes/reasserts SDPA
  |     +-- resolves the owner-qualified Kaggle model path
  |     +-- creates the canonical RunContract
  |
  +-- public_demo/live_execution.py
        +-- requires Tesla T4 x2
        +-- loads the model once
        +-- applies the explicit dual-device plan
        +-- executes one T2I trajectory
        +-- reduces telemetry into routing/precision/output evidence
```

The qualified runtime implementation lives under `mage_t4x2/`. The public demo layer orchestrates those modules without replacing the frozen authority artifacts under `authority/`.

## Device topology

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
512x512 RGB image
```

For the canonical four-step run, the transformer block sequence `[0..11]` is expected once per denoising invocation.

## Key modules

- `mage_t4x2/device_plan.py`: explicit component/block placement.
- `mage_t4x2/transformer_placement.py`: applies and validates complete transformer placement.
- `mage_t4x2/device_bridges.py` and `dual_device_forward.py`: cross-device boundaries and the single logical transformer path.
- `mage_t4x2/spread_loader.py`: qualified model-loading path.
- `mage_t4x2/sdpa_contract.py`: fail-closed SDPA contract.
- `mage_t4x2/dtype_audit.py`: BF16 materialization evidence.
- `mage_t4x2/evidence_reducers.py`: derives block-order, transfer, participation, and single-instance facts from telemetry.
- `mage_t4x2/runtime_baseline.py`: verifies the frozen successor runtime baseline.
- `scripts/gpu_session.py`: GPU-session orchestration used by the public execution layer.

## Fail-closed boundaries

The public path stops instead of silently weakening the contract when source identity, model path, T4x2 hardware, SDPA, placement, routing, BF16 materialization, CPU-fallback policy, or output validation does not match the expected evidence.

## Precision scope

The supported claim is:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

This architecture does not claim native BF16 Tensor Core acceleration on Tesla T4.
