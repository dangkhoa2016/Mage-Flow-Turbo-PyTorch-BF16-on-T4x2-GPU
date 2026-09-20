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
    model_path=None,
    prompt=CANONICAL_PROMPT,
    seed=None,
    steps=None,
    width=None,
    height=None,
    cfg_scale=None,
    num_blocks=12,
    split_block=1,
    output_root=None,
    run_id=None,
)
```

### Hành vi quan trọng

- `project_root` phải chứa `mage_t4x2/` và `scripts/`.
- `prompt` phải đúng canonical prompt, nếu không runner fail-closed.
- Nếu không truyền `model_path`, runner chỉ resolve owner-qualified Kaggle path dưới `/kaggle/input/models/<owner>/<model>/pytorch/default/1`.
- Runner xác minh R2G baseline trước live execution.
- SDPA được xác minh trước model execution, reassert sau model construction và kiểm tra lại ngay trước inference.
- Live path yêu cầu đúng hai GPU Tesla T4.
- Model load count phải giữ đúng một.
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
python scripts/public_kaggle_demo.py   [--project-root PATH]   [--model-path PATH]   [--output-root PATH]
```

Exit status:

- `0`: runner không kết thúc với final `FAIL`.
- `1`: final status là `FAIL`.

CLI cố ý không expose switch cho prompt/seed/steps/CFG/resolution. Các giá trị đó thuộc canonical public run contract.

## Internal interface

Các module trong `mage_t4x2/` là building block của qualification/runtime. Đây là source public, nhưng repository không cam kết chúng là general-purpose library API có versioning ổn định.
