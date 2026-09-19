# Kaggle

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](kaggle.md)

## Môi trường đích

Public demo nhắm tới một Kaggle Notebook mới với:

- Tăng tốc: **Tesla T4 x2**
- Internet: **ON**
- Model đính kèm: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- Không cần dataset bổ sung cho demo text-to-image chuẩn
- Đỉnh GPU và showcase evidence ghi tại `artifacts/public-showcase/`

Đường dẫn model mong đợi:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

## Bootstrap nguồn

Notebook clone repository này vào `/kaggle/working` và phân giải `SOURCE_REF` đã commit (mặc định `v1.0.0`, một release tag hoặc SHA 40 ký tự chính xác) trước khi import các module project.

Runtime sau đó tải chính xác commit Mage upstream đã pin và xác minh cả hai:

- upstream commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`

Dependency bootstrap `loguru==0.7.3` cũng được xác minh bằng kích thước wheel và SHA-256 trước khi cài vào target bootstrap cô lập.

## Quy trình chạy khuyến nghị

1. Tạo Kaggle Notebook hoàn toàn mới.
2. Chọn **GPU T4 x2**.
3. Bật Internet.
4. Đính kèm Kaggle model cần thiết.
5. Import/mở `notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb`.
6. Bắt đầu với kernel mới.
7. Chạy **Run All đúng một lần**.
8. Giữ lại output notebook và thư mục evidence đã sinh.

Không tự tải lên publication ZIP hay identity JSON. Workflow công khai dùng Git source bootstrap.

## Lane A — kiểm định chuẩn

Một lane A thành công xác lập, trong các kiểm tra khác:

- đúng hai thiết bị Tesla T4;
- model load count bằng một;
- một quỹ đạo T2I;
- GPU0 và GPU1 tham gia;
- bốn transformer invocation cho profile bốn bước;
- bốn transfer biên 0->1 mong đợi;
- bốn return 1->0;
- một transfer VAE-input 0->1;
- BF16 materialization PASS;
- CPU fallback không quan sát thấy;
- output RGB 512x512 hợp lệ.

## Lane B — showcase production

`run_public_showcase` (từ `public_demo.showcase`) là một lane riêng với một lần nạp model của chính nó. Nó áp dụng cùng placement T4 kép và giữ model nóng xuyên suốt bảng case xác định ở `512 / 768 / 1024`. Evidence từng ảnh (timing, đỉnh GPU allocated/reserved, routing facts, SHA-256) được ghi tại `artifacts/public-showcase/<run_id>/` cùng `showcase-summary.json`, `case-results.json` và manifest. Key final fail-closed là `PUBLIC_SHOWCASE_FINAL_VERDICT`.

## Lần chạy thất bại

Nếu một ô nặng thất bại, hãy giữ nguyên lần thất bại đó. Để có provenance xuất bản nghiêm ngặt, hãy bắt đầu một phiên Kaggle **mới khác** thay vì sửa state, xóa thư mục runtime, hay chỉ chạy lại ô thất bại.

## Notebook và authority

Public notebook là bề mặt tái lập/xuất bản. Evidence authority qualified đã đóng băng vẫn là bản ghi định tuyến/runtime chính thức và không bị ghi đè bởi một public run mới.