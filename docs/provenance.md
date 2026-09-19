# Provenance and Authority

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](provenance.vi.md)

## Why provenance is explicit

The public demo is built around a closed routing/runtime qualification. Reproducibility therefore depends on identifying the exact project, upstream source, model, bootstrap dependency, and qualified runtime bytes instead of relying only on package names.

## Pinned identities

| Item | Identity |
|---|---|
| Model ID | `dangkhoa2016/mage-flow-community-mage-flow-turbo` |
| Model revision | `65bb3500f0da9df6a41ec6383716fc02cf014773` |
| Upstream Mage repository | `https://github.com/microsoft/Mage` |
| Upstream Mage commit | `76bec2bb3818863f470de7e867c2dc7f1d0bfd83` |
| Upstream `mage_flow` tree | `946b91bcb2cac75e6cfe8399f0f7f330a2280adf` |
| Bootstrap dependency | `loguru==0.7.3` |
| Bootstrap wheel SHA-256 | `31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c` |

## qualified runtime baseline

`authority/runtime-baseline.json` records the exact size and SHA-256 of the qualified successor runtime files.

`mage_t4x2.runtime_baseline.verify_runtime_baseline()` verifies:

- manifest schema/name/status;
- required runtime file set;
- duplicate/missing entries;
- byte size;
- SHA-256.

The public runner executes this gate before model bootstrap/live inference.

## Closed authority status

The prior qualified authority established the qualified routing/runtime facts, including:

```text
DUAL_T4_RUNTIME_AUTHORITY=CLOSED_PASS
MODEL_LOAD_COUNT=1
DUAL_T4_SINGLE_INFERENCE_TRAJECTORY=PROVEN
GPU0_PARTICIPATION=PROVEN
GPU1_PARTICIPATION=PROVEN
BF16_MATERIALIZATION=PASS
CPU_FALLBACK=NOT_OBSERVED
OUTPUT_VALID=PASS
```

A public notebook run demonstrates reproducibility; it does not retroactively rewrite the authority record.

## Evidence produced by new runs

The public runner creates per-run evidence including contract/environment/source authority, telemetry, dtype records, output validation, summary JSON and ordered gate verdicts.

A PASS should be interpreted only together with the matching evidence directory and source revision.

## Change policy

A source change that modifies files protected by the qualified runtime baseline can invalidate byte-equivalence with the closed authority. Such changes require explicit review and, where relevant, new qualification rather than silently preserving the old claim.
