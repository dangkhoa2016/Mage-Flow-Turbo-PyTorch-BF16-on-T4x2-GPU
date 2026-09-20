# Cách sử dụng

> 🌐 Language / Ngôn ngữ: [English](usage.md) | **Tiếng Việt**

## Cách khuyến nghị: Kaggle notebook

Dùng notebook trong repository:

```text
notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb
```

Chuẩn bị Kaggle session mới với:

- Accelerator: **Tesla T4 x2**
- Internet: **ON**
- Model attach: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- Canonical text-to-image demo không cần dataset bổ sung

Model mount dự kiến:

```text
/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1
```

Dùng fresh kernel và **Run All đúng một lần**. Không rerun riêng heavy cell đã lỗi rồi dùng session trộn đó làm final publication evidence.

## CLI entry point

Public CLI là:

```bash
python scripts/public_kaggle_demo.py
```

Các argument tùy chọn:

```bash
python scripts/public_kaggle_demo.py \
  --project-root /path/to/repository \
  --model-path /path/to/model \
  --output-root /path/to/output
```

CLI gọi `public_demo.runner.run_public_demo()`, in toàn bộ gate verdict cùng run ID và summary path, và chỉ trả exit code `1` khi final status là `FAIL`.

## Canonical run contract

Public demo cố ý giới hạn phạm vi:

| Thiết lập | Giá trị |
|---|---|
| Prompt | `a red fox in a snowy forest at golden hour, high detail` |
| Seed | `42` |
| Steps | `4` |
| CFG | `1.0` |
| Resolution | `512x512` |
| Transformer blocks | `12` |
| `split_block` | `1` |
| Attention | `sdpa` |
| CPU fallback | tắt |

Public runner từ chối prompt khác. Mục tiêu của nó là qualification/public demonstration có khả năng tái lập, không phải playground prompt tổng quát.

## Output

Một run thành công ghi evidence directory dưới output root đã chọn. Result trả về gồm:

- `run_id`
- final `status`
- `first_failed_gate`
- `error`
- `summary_path`
- `verdict_lines`
- các đường dẫn output/evidence và live fact đã reduce

Hãy giữ nguyên evidence directory đầy đủ của cả failed run hoặc successful run khi review.

## Tài liệu liên quan

- [Kiến trúc](architecture.vi.md)
- [Python và CLI API](api.vi.md)
- [Kaggle](kaggle.vi.md)
- [Giới hạn](limitations.vi.md)
- [Provenance](provenance.vi.md)
