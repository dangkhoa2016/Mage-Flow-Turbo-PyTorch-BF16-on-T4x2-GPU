# Mage-Flow-Turbo PyTorch BF16 trên GPU T4×2

> 🌐 Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Dự án engineering có khả năng tái lập để chạy **Mage-Flow-Turbo** bằng pipeline PyTorch upstream trên **Kaggle Tesla T4 ×2**, với một luồng text-to-image logic duy nhất đi qua cả hai GPU.

Đây là dự án engineering độc lập xây dựng quanh model/runtime Microsoft Mage upstream. Dự án **không phải bản phát hành chính thức của Microsoft**.

## Repository này chứng minh điều gì

- Chạy bằng PyTorch upstream, không dùng `stable-diffusion.cpp` / `sd-cli`.
- BF16 dtype/materialization cho text encoder, transformer và VAE.
- Chỉ một lần load model logic.
- Chỉ một T2I trajectory đi qua cả hai Tesla T4.
- Transformer block 0 trên `cuda:0`; block 1–11 và output head trên `cuda:1`.
- Transfer boundary `cuda:0 -> cuda:1` và transformer return `cuda:1 -> cuda:0` tường minh.
- VAE input được chuyển từ `cuda:0` sang `cuda:1`.
- Attention dùng SDPA; T4 path không phụ thuộc `flash_attn`.
- Evidence và routing check theo fail-closed.

Cách mô tả precision được hỗ trợ là:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

Dự án không tuyên bố Tesla T4 có native BF16 Tensor Core acceleration.

## Chạy nhanh trên Kaggle

Public notebook được thiết kế để người dùng **không cần** bất kỳ internal ZIP, identity JSON hay source bundle nào.

### 1. Tạo Kaggle Notebook mới

Cấu hình:

- Accelerator: **GPU T4 ×2**
- Internet: **ON**

### 2. Attach Kaggle Model bắt buộc

Thêm model:

`dangkhoa2016/mage-flow-community-mage-flow-turbo`

Đường dẫn model read-only dự kiến:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

Demo text-to-image không cần dataset bổ sung.

### 3. Mở public notebook

Notebook trong repository:

```text
notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb
```

Notebook tự clone repository này tại release tag `v1.0.0` vào `/kaggle/working`, xác minh source ref, lấy đúng upstream Mage source đã pin, kiểm tra bootstrap wheel, rồi mới chạy demo.

Dùng fresh kernel và **Run All đúng một lần**.

Xem [Hướng dẫn Kaggle public demo](docs/kaggle-public-demo.vi.md) để biết đầy đủ các bước.

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

Với demo canonical 4 steps, chuỗi transformer `[0..11]` phải xuất hiện bốn lần—mỗi lần tương ứng một denoising invocation.

## Cấu hình demo canonical

| Thiết lập | Giá trị |
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

## Provenance đã pin

- Model revision: `65bb3500f0da9df6a41ec6383716fc02cf014773`
- Upstream Mage commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- Upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`
- R2G runtime baseline manifest SHA-256: `b829fdea0b568ff80ff74d1c58344d7f289d781d42029f2b42a34a8d9c266466`
- Bootstrap dependency: `loguru==0.7.3`
- Bootstrap wheel SHA-256: `31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c`

## Trạng thái R2G authority

R2G authority đã đóng chứng minh:

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

Public notebook là bề mặt demo có khả năng tái lập; nó không ghi đè authority evidence đã đóng băng.

## Cấu trúc repository

```text
mage_t4x2/                 runtime modules đã qualification
public_demo/               public Git/PyPI bootstrap + demo runner
scripts/                   CLI và authority drivers
authority/                 runtime baseline và provenance manifests
notebooks/                 public Kaggle notebook
docs/                      tài liệu sử dụng song ngữ
tests/                     CPU/static contract tests
```

## Tài liệu

- [Kiến trúc](docs/architecture.vi.md)
- [Cách sử dụng](docs/usage.vi.md)
- [Python và CLI API](docs/api.vi.md)
- [Thiết lập Kaggle](docs/kaggle.vi.md)
- [Quy trình Kaggle public demo](docs/kaggle-public-demo.vi.md)
- [Giới hạn](docs/limitations.vi.md)
- [Provenance và authority](docs/provenance.vi.md)
- [Testing và validation](docs/testing.vi.md)
- [Đóng góp](.github/CONTRIBUTING.vi.md)
- [Bảo mật](.github/SECURITY.vi.md)
- [Hỗ trợ](.github/SUPPORT.vi.md)

## Safety và fail-closed

Public path dừng fail-closed nếu, ví dụ:

- source checkout cũ hoặc không đúng release ref;
- upstream Mage commit/tree khác pin;
- bootstrap wheel khác size hoặc SHA-256;
- R2G runtime baseline không khớp;
- không có đúng hai Tesla T4;
- thiếu owner-qualified Kaggle model attachment;
- SDPA không còn hiệu lực sau model construction hoặc trước inference;
- model load count khác một;
- block order, transfer cardinality hoặc GPU participation sai contract;
- quan sát thấy CPU fallback;
- output không phải ảnh RGB 512×512 hợp lệ.

## License và upstream code

Code và tài liệu nguyên bản của repository này được phát hành theo [MIT License](LICENSE).

Copyright (c) 2026 **Đăng Khoa <i.am@dangkhoa.dev>**.

Upstream Mage source được lấy trực tiếp từ public Git commit đã pin khi chạy, thay vì copy vào repository này. Upstream source và model artifact vẫn chịu license và điều khoản riêng; hãy kiểm tra trước khi redistribution hoặc production use.
