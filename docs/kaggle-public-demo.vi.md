# Hướng dẫn public demo trên Kaggle

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](kaggle-public-demo.md)

Hướng dẫn này mô tả đường dẫn Kaggle công khai, tái lập được cho demo Mage-Flow-Turbo PyTorch BF16 trên T4 kép, chạy **hai lane độc lập**: Lane A (kiểm định chuẩn) và Lane B (showcase production).

## Yêu cầu

- Tài khoản Kaggle có **Tesla T4 ×2**.
- Notebook bật Internet.
- Đính kèm Kaggle Model: `dangkhoa2016/mage-flow-community-mage-flow-turbo`.
- Không cần dataset bổ sung.

## Thiết lập Kaggle

1. Tạo Kaggle Notebook mới.
2. Mở **Settings → Accelerator** và chọn **GPU T4 ×2**.
3. Bật **Internet**.
4. Dùng **Add Input → Models** và đính kèm:
   `dangkhoa2016/mage-flow-community-mage-flow-turbo`.
5. Xác nhận model hiển thị tại:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

Không tự sao chép model vào `/kaggle/working`.

## Bootstrap nguồn (Source bootstrap)

Notebook clone:

```text
https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU.git
```

rồi phân giải `SOURCE_REF` đã commit (mặc định `v1.0.0`), có thể là release tag hoặc SHA commit 40 ký tự chính xác, vào:

```text
/kaggle/working/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU
```

Notebook xác minh Git origin, phân giải ref (tag hoặc SHA), checkout detached commit đã phân giải và ghi lại `HEAD`. Không cần publication ZIP hay identity JSON.

## Bootstrap runtime

Runner công khai sau đó:

1. Xác minh `authority/runtime-baseline.json`.
2. Đóng băng SDPA contract trước khi import upstream.
3. Clone Microsoft Mage tại commit pin chính xác.
4. Xác minh định danh cây `mage_flow` trong Git.
5. Tải `loguru==0.7.3` dạng wheel.
6. Xác minh kích thước byte và SHA-256 của wheel.
7. Tạo local bootstrap site mới.
8. Xác minh phiên bản dependency bootstrap và nguồn import.
9. Import upstream Mage source đã pin.
10. Yêu cầu đúng hai GPU Tesla T4.

## Lane A — kiểm định chuẩn (acceptance authority)

Runner chuẩn (`public_demo.runner.run_public_demo`) **không** nhận ghi đè prompt/seed/độ phân giải và chứng minh acceptance contract đóng băng:

```text
Prompt: a red fox in a snowy forest at golden hour, high detail
Seed: 42
Steps: 4
CFG: 1.0
Resolution: 512x512
split_block: 1
num_blocks: 12
```

Các PASS fact mong đợi:

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

## Lane B — showcase production (demo đa-ảnh bền vững)

`public_demo.showcase.run_public_showcase` tự thực hiện **một lần nạp model riêng** (không bao giờ dùng chung Lane A), áp dụng cùng placement T4 kép, giữ model nóng xuyên suốt bảng case xác định (`s01..s24`) và ghi evidence từng ảnh:

- output gắn seed `outputs/<case>-<res>x<res>-seed<seed>.png`;
- telemetry từng case tại `telemetry/<case>.jsonl`;
- đỉnh GPU allocated + reserved cho từng ảnh trên cả hai T4;
- routing facts rút từ telemetry (thứ tự block, participation, một lane instance, không CPU fallback);
- `showcase-summary.json`, `case-results.json`, `showcase-runtime.log` và manifest, tất cả tại:
  `artifacts/public-showcase/<run_id>/`.

> Phần trình bày chính theo từng ảnh của notebook hiển thị độc lập mọi PNG đã sinh và in prompt xác định chính xác ngay phía trên ảnh tương ứng, cùng case id, category, độ phân giải, seed, thời gian inference và SHA-256. Bảng prompt rút gọn vẫn được giữ làm audit view.

Các hằng số showcase (cố định theo contract):

```text
resolutions: 512 / 768 / 1024
steps: 4
CFG: 1.0
mandatory minimum cases: 12
max cases: 24
target inference window: 300 s
verdict key: PUBLIC_SHOWCASE_FINAL_VERDICT
```

## Kết luận final mong đợi

```text
PUBLIC_DEMO_FINAL_VERDICT=PASS
PUBLIC_SHOWCASE_FINAL_VERDICT=PASS
```

Một showcase đẹp không bao giờ được che một canonical qualification thất bại.

## Kỷ luật xuất bản

Để có bằng chứng công khai sạch:

- bắt đầu từ Kaggle kernel mới;
- chạy tất cả ô theo thứ tự đúng một lần;
- không nạp model thủ công trước;
- không xóa state cũ và chạy lại trong cùng một lần chứng minh;
- nếu một ô thất bại, giữ nguyên lần thất bại đó và chẩn đoán từ `first_failed_gate`, `error`, `summary.json`, `runtime.log`.

## Outputs

Các artifact runtime được ghi dưới `/kaggle/working` trong cây project/output đã clone. Kaggle model vẫn read-only dưới `/kaggle/input`.