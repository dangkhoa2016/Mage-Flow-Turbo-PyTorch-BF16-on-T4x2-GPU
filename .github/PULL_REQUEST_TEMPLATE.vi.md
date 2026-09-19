# Pull Request

> 🌐 Language / Ngôn ngữ: [English](PULL_REQUEST_TEMPLATE.md) | **Tiếng Việt**

## Tóm tắt

<!-- Thay đổi gì và vì sao? Giữ PR tập trung vào một mục tiêu có thể review rõ ràng. -->

## Phạm vi

- [ ] Qualified runtime / routing
- [ ] Public demo runner / bootstrap
- [ ] Kaggle notebook
- [ ] Test / CI
- [ ] Tài liệu / repository metadata
- [ ] Khác

## Validation đã chạy

- [ ] `python -m compileall -q mage_t4x2 public_demo scripts`
- [ ] các `pytest` test liên quan
- [ ] kiểm tra language pair khi thay đổi Markdown
- [ ] không thêm credential, model weights, runtime cache hoặc generated evidence archive

## Tác động tới authority

- [ ] Không thay đổi file được `authority/runtime-baseline.json` bảo vệ.
- [ ] Có thay đổi protected file và PR này mô tả rõ tác động authority/requalification.

## GPU claim

<!-- Nêu rõ PR chỉ CPU/static hay có real Tesla T4 x2 evidence mới. Không suy diễn GPU acceptance từ CPU CI. -->

## Tương thích / bảo mật

<!-- Ghi rõ tác động source provenance, dependency, model path, backward compatibility hoặc security. -->
