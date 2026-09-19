# Testing và Validation

> 🌐 Language / Ngôn ngữ: [English](testing.md) | **Tiếng Việt**

## Hai lớp validation

Repository này phân biệt CPU/static validation và real GPU acceptance.

### CPU/static validation

Các test an toàn CPU có thể xác minh:

- source/bootstrap contract;
- path và provenance rule;
- byte integrity của qualified manifest;
- routing/telemetry reducer bằng fixture;
- cấu trúc notebook;
- documentation contract;
- hành vi fail-closed không cần load model.

Các check này không được mô tả như bằng chứng rằng live dual-T4 inference đã thành công.

### Real GPU validation

Các claim về hai GPU thực sự tham gia, model placement, cross-device transfer, BF16 live materialization, không CPU fallback và generated image output cần real **Tesla T4 x2** session.

## Lệnh local hữu ích

Tối thiểu:

```bash
python -m compileall -q mage_t4x2 public_demo scripts
python -m pytest -q
```

Focused public-demo tests:

```bash
python -m pytest -q \
  tests/test_public_bootstrap.py \
  tests/test_public_contract.py \
  tests/test_public_demo_gates.py \
  tests/test_public_runner_static.py \
  tests/test_public_docs_contract.py
```

## Authority integrity

Public runner tự xác minh `authority/runtime-baseline.json` trước khi tiếp tục.

Để kiểm tra độc lập:

```python
from mage_t4x2.runtime_baseline import verify_runtime_baseline
print(verify_runtime_baseline())
```

Qualified state dự kiến là `status == "PASS"` và không có required file bị thiếu hoặc hash/size mismatch.

## Publication validation

Đối với final public Kaggle artifact:

- bắt đầu từ notebook/kernel mới;
- chạy toàn bộ cell đúng một lần;
- tránh cleanup/retry cell làm thay đổi failed state;
- giữ nguyên toàn bộ output;
- giữ evidence directory;
- ghi lại Git commit/tag notebook sử dụng.

Một run thành công về kỹ thuật và một pristine publication artifact là hai acceptance question có liên quan nhưng khác nhau.
