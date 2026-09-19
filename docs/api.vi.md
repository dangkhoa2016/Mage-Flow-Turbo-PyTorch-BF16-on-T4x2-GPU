# Python và CLI API

> 🌐 Language / Ngôn ngữ: [English](api.md) | **Tiếng Việt**

Dự án này không expose REST service. Public interface được hỗ trợ là Python public-demo runner cùng CLI wrapper nhỏ.

## Python API

Import:

```python
from public_demo.runner import run_public_demo
```

Signature:

```python
run_public_demo(
    project_root=None,
    output_root=None,
    run_id=None,
)
```

Public qualification runner cố ý chỉ hỗ trợ canonical. Không có override parameter nào cho prompt/seed/steps/CFG/resolution/block/split: runner luôn chạy đúng immutable canonical public contract.

### Hành vi quan trọng

- `project_root` phải chứa `mage_t4x2/` và `scripts/`.
- `PublicDemoContract` canonical trong `public_demo/contract.py` là nguồn chân lý duy nhất cho public demo inputs, gồm prompt, seed (`42`), steps (`4`), CFG (`1.0`), resolution (`512x512`), số block (`12`), `split_block` (`1`), attention SDPA, dtype BF16, model ID/revision và upstream commit/tree.
- Runner fail-closed với mọi public input không phải canonical (`NONCANONICAL_PUBLIC_INPUT`).
- Model path chỉ được resolve tới owner-qualified Kaggle attachment dưới `/kaggle/input/models/<owner>/<model>/pytorch/default/1`; basename-only path, bỏ owner namespace và remote fallback đều bị từ chối.
- Runner xác minh qualified baseline trước live execution.
- SDPA được xác minh trước model execution, reassert sau model construction và kiểm tra lại ngay trước inference.
- Live path yêu cầu đúng hai GPU Tesla T4.
- `model_load_count` được derive từ load lifecycle evidence của run và phải đúng bằng `1`.
- CPU fallback, `stable-diffusion.cpp` và `sd-cli` nằm ngoài public contract.

### Giá trị trả về

Hàm trả về dictionary. Các top-level field ổn định đang được notebook/CLI sử dụng gồm:

| Field | Ý nghĩa |
|---|---|
| `status` | `PASS`, `FAIL` hoặc trạng thái trung gian khi đang chạy |
| `run_id` | định danh evidence run |
| `first_failed_gate` | gate fail-closed đầu tiên, hoặc `None` khi PASS |
| `error` | nội dung lỗi, hoặc `None` khi PASS |
| `summary_path` | summary JSON được tạo |
| `verdict_lines` | các gate line theo đúng thứ tự |

Summary còn chứa hardware, routing, placement, precision, safety và output evidence đã reduce nếu live inference đi tới các stage đó.

## CLI API

```bash
python scripts/public_kaggle_demo.py   [--project-root PATH]   [--output-root PATH]
```

Exit status:

- `0`: runner không kết thúc với final `FAIL`.
- `1`: final status là `FAIL`.

CLI cố ý không expose switch cho prompt/seed/steps/CFG/resolution hay model path. Các giá trị đó thuộc canonical public run contract.

## Internal interface

Các module trong `mage_t4x2/` là building block của qualification/runtime. Đây là source public, nhưng repository không cam kết chúng là general-purpose library API có versioning ổn định.
