# Limitations

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](limitations.vi.md)

## Hardware scope

The qualified public path targets **exactly two NVIDIA Tesla T4 GPUs**. Other GPUs, a single T4, CPU-only execution, P100, TPU, or different accelerator topologies are not covered by the public dual-T4 acceptance claim.

## Precision wording

The supported statement is:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

This does not mean Tesla T4 provides native BF16 Tensor Core acceleration.

## Fixed public demo profile

The public runner is a reproducibility/qualification surface, not a general image-generation API. The canonical demo fixes:

- prompt;
- 512x512 resolution;
- four denoising steps;
- CFG 1.0;
- seed 42;
- 12 transformer blocks;
- split block 1.

Changing these values may be useful for experimentation, but such runs are outside the exact public acceptance profile unless separately qualified.

## Model scope

The repository targets the pinned Mage-Flow-Turbo model ID/revision used by the authority evidence. It does not claim compatibility with arbitrary Mage models, Edit-Turbo, future model revisions, or unrelated diffusion architectures.

## Internet and external availability

The public Git bootstrap requires Internet access to obtain repository/upstream/dependency content. Availability of GitHub, PyPI and Kaggle model attachments is outside this repository's control.

## No service SLA

This project is an engineering demonstration and reproducibility repository. It does not provide:

- a hosted REST service;
- multi-tenant isolation;
- availability/SLA commitments;
- production autoscaling;
- authentication/authorization infrastructure;
- long-term compatibility guarantees for internal `mage_t4x2` modules.

## Authority versus new environments

CPU/static CI can detect many regressions, but it does not replace real T4x2 qualification for GPU claims. Likewise, a successful run on different hardware is not direct evidence for the qualified Kaggle T4x2 topology.

## Upstream and model licenses

This repository's MIT license covers this repository's own original code and documentation. Upstream Mage source and model artifacts remain subject to their own licenses and terms.
