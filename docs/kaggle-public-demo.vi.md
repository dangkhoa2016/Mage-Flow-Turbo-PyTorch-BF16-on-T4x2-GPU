# Hướng dẫn Kaggle public demo

> 🌐 Ngôn ngữ: [English](kaggle-public-demo.md) | **Tiếng Việt**

Tài liệu này mô tả public Kaggle path có khả năng tái lập cho demo Mage-Flow-Turbo PyTorch BF16 dual-T4.

## Yêu cầu

- Tài khoản Kaggle có thể dùng **Tesla T4 ×2**.
- Bật Internet cho Notebook.
- Attach Kaggle Model: `dangkhoa2016/mage-flow-community-mage-flow-turbo`.
- Không cần dataset bổ sung.

## Thiết lập Kaggle

1. Tạo Kaggle Notebook mới.
2. Mở **Settings → Accelerator** và chọn **GPU T4 ×2**.
3. Bật **Internet**.
4. Dùng **Add Input → Models** và attach:
   `dangkhoa2016/mage-flow-community-mage-flow-turbo`.
5. Xác nhận model xuất hiện tại:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

Không copy thủ công model vào `/kaggle/working`.

## Source bootstrap

Notebook clone:

```text
https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU.git
```

tại release ref:

```text
v1.0.0
```

vào:

```text
/kaggle/working/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU
```

Checkout target phải chưa tồn tại. Notebook xác minh Git origin, đúng release tag và ghi lại commit SHA thực tế.

Không cần publication ZIP hoặc identity JSON.

## Runtime bootstrap

Sau đó public runner:

1. Xác minh `authority/r2g-runtime-baseline.json`.
2. Freeze SDPA contract trước khi import upstream.
3. Clone Microsoft Mage tại đúng commit đã pin.
4. Xác minh Git tree identity của `mage_flow`.
5. Tải wheel `loguru==0.7.3`.
6. Xác minh byte size và SHA-256 của wheel.
7. Tạo local bootstrap site mới.
8. Xác minh version và import origin của bootstrap dependency.
9. Import upstream Mage source đã pin.
10. Yêu cầu đúng hai Tesla T4.
11. Chỉ resolve owner-qualified local Kaggle model path.
12. Load model đúng một lần.
13. Reassert SDPA sau model construction.
14. Áp dụng explicit dual-T4 placement.
15. Reassert SDPA ngay trước inference.
16. Chạy một T2I trajectory và suy ra acceptance từ telemetry thực tế.

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

## Các fact PASS dự kiến

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

## Kỷ luật publication

Để có public proof sạch:

- bắt đầu từ fresh Kaggle kernel;
- chạy tất cả cell đúng thứ tự và đúng một lần;
- không preload model thủ công;
- không xóa stale state rồi chạy lại trong cùng proof attempt;
- nếu cell lỗi, giữ nguyên failed attempt và chẩn đoán từ `first_failed_gate`, `error`, `summary.json` và `runtime.log`.

## Output

Runtime artifacts được ghi dưới `/kaggle/working` trong project/output tree đã clone. Kaggle model vẫn read-only dưới `/kaggle/input`.
