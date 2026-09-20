# Hướng dẫn đóng góp

> 🌐 Language / Ngôn ngữ: [English](CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã đóng góp cho dự án engineering Mage-Flow-Turbo PyTorch BF16 dual-T4 này.

## Nguyên tắc

Mọi thay đổi nên:

- tập trung và dễ review;
- có khả năng tái lập;
- nói rõ có ảnh hưởng closed R2G authority hay không;
- có thể kiểm tra an toàn mà không load model, trừ khi chính hành vi GPU là đối tượng thay đổi.

Không nới lỏng fail-closed gate chỉ để demo chạy PASS.

## Development validation

Với thay đổi CPU/static, tối thiểu chạy:

```bash
python -m compileall -q mage_t4x2 public_demo scripts
python -m pytest -q
```

Với thay đổi tập trung vào public demo:

```bash
python -m pytest -q   tests/test_public_bootstrap.py   tests/test_public_demo_gates.py   tests/test_public_runner_static.py   tests/test_public_docs_contract.py
```

CPU/static PASS không phải evidence rằng live dual-T4 inference đã thành công.

## Chính sách tài liệu song ngữ

Mỗi tài liệu Markdown của project nên có cặp English/Vietnamese trong cùng thư mục:

- `name.md`
- `name.vi.md`

Documentation commit nên thay đổi cả hai phía cùng lúc. Mỗi cặp phải có language switcher gần đầu tài liệu.

## Thay đổi nhạy cảm với authority

Các file được liệt kê trong `authority/r2g-runtime-baseline.json` là runtime authority file đã byte-qualified. Nếu thay đổi một file trong số đó, phải nói rõ byte-equivalence với closed R2G authority không còn được giữ và xác định có cần requalification hay không.

## Pull request

Hãy mô tả:

- thay đổi gì và vì sao;
- các lệnh validation thực sự đã chạy;
- thay đổi chỉ CPU/static hay có T4x2 evidence mới;
- tác động tới source/model/dependency provenance;
- tác động tới public notebook hoặc release workflow;
- tác động về tương thích hoặc bảo mật.

Không commit model weights, secret, Kaggle token, runtime cache, generated evidence archive hoặc credential cá nhân vào Git.
