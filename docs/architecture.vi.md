# Kiến trúc

> 🌐 Language / Ngôn ngữ: [English](architecture.md) | **Tiếng Việt**

## Mục tiêu

Repository này chạy Mage-Flow-Turbo bằng pipeline Mage/PyTorch upstream trên đúng hai GPU Tesla T4, đồng thời giữ một lần load model logic duy nhất và một luồng text-to-image duy nhất.

## Các lớp runtime

```text
Kaggle notebook
  |
  +-- public_demo/bootstrap.py
  |     +-- xác minh upstream Mage commit/tree đã pin
  |     +-- tải và xác minh wheel loguru đã pin
  |     +-- chuẩn bị bootstrap site biệt lập
  |
  +-- public_demo/runner.py
  |     +-- xác minh qualified runtime baseline đã đóng băng
  |     +-- freeze/reassert SDPA
  |     +-- resolve owner-qualified Kaggle model path
  |     +-- tạo RunContract canonical
  |
  +-- public_demo/live_execution.py
        +-- yêu cầu Tesla T4 x2
        +-- load model đúng một lần
        +-- áp dụng dual-device plan tường minh
        +-- chạy một T2I trajectory
        +-- reduce telemetry thành routing/precision/output evidence
```

Runtime đã qualification nằm trong `mage_t4x2/`. Lớp public demo điều phối các module này mà không thay thế authority artifact đã đóng băng trong `authority/`.

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

Với canonical run 4 steps, chuỗi transformer block `[0..11]` phải xuất hiện đúng một lần cho mỗi denoising invocation.

## Các module chính

- `mage_t4x2/device_plan.py`: placement tường minh theo component/block.
- `mage_t4x2/transformer_placement.py`: áp dụng và xác minh placement đầy đủ của transformer.
- `mage_t4x2/device_bridges.py` và `dual_device_forward.py`: boundary cross-device và single logical transformer path.
- `mage_t4x2/spread_loader.py`: model-loading path đã qualification.
- `mage_t4x2/sdpa_contract.py`: SDPA contract fail-closed.
- `mage_t4x2/dtype_audit.py`: BF16 materialization evidence.
- `mage_t4x2/evidence_reducers.py`: suy ra block order, transfer, GPU participation và single-instance từ telemetry.
- `mage_t4x2/runtime_baseline.py`: xác minh successor runtime baseline đã đóng băng.
- `scripts/gpu_session.py`: orchestration GPU session được public execution layer sử dụng.

## Các boundary fail-closed

Public path dừng lại thay vì âm thầm nới lỏng contract nếu source identity, model path, phần cứng T4x2, SDPA, placement, routing, BF16 materialization, chính sách CPU fallback hoặc output validation không khớp evidence dự kiến.

## Phạm vi precision

Claim được hỗ trợ là:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

Kiến trúc này không tuyên bố Tesla T4 có native BF16 Tensor Core acceleration.
