# Provenance và Authority

> 🌐 Language / Ngôn ngữ: [English](provenance.md) | **Tiếng Việt**

## Vì sao provenance được ghi tường minh

Public demo được xây dựng quanh một routing/runtime qualification đã đóng. Vì vậy khả năng tái lập phụ thuộc vào việc định danh chính xác project, upstream source, model, bootstrap dependency và qualified runtime bytes thay vì chỉ dựa vào package name.

## Identity đã pin

| Hạng mục | Identity |
|---|---|
| Model ID | `dangkhoa2016/mage-flow-community-mage-flow-turbo` |
| Model revision | `65bb3500f0da9df6a41ec6383716fc02cf014773` |
| Upstream Mage repository | `https://github.com/microsoft/Mage` |
| Upstream Mage commit | `76bec2bb3818863f470de7e867c2dc7f1d0bfd83` |
| Upstream `mage_flow` tree | `946b91bcb2cac75e6cfe8399f0f7f330a2280adf` |
| Bootstrap dependency | `loguru==0.7.3` |
| Bootstrap wheel SHA-256 | `31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c` |

## R2G runtime baseline

`authority/r2g-runtime-baseline.json` lưu exact size và SHA-256 của qualified successor runtime files.

`mage_t4x2.r2g_runtime_baseline.verify_r2g_runtime_baseline()` xác minh:

- manifest schema/name/status;
- required runtime file set;
- duplicate/missing entry;
- byte size;
- SHA-256.

Public runner chạy gate này trước model bootstrap/live inference.

## Closed authority status

R2G authority trước đó đã xác lập các routing/runtime fact đã qualification, gồm:

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

Một public notebook run chứng minh reproducibility; nó không ghi lại authority record trong quá khứ.

## Evidence từ run mới

Public runner tạo evidence theo từng run gồm contract/environment/source authority, telemetry, dtype record, output validation, summary JSON và ordered gate verdicts.

PASS chỉ nên được diễn giải cùng evidence directory và source revision tương ứng.

## Chính sách thay đổi

Source change làm thay đổi file được R2G runtime baseline bảo vệ có thể làm mất byte-equivalence với closed authority. Các thay đổi đó cần review tường minh và, khi phù hợp, qualification mới thay vì âm thầm giữ claim cũ.
