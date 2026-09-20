# Kaggle

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

## Môi trường mục tiêu

Public demo nhắm tới Kaggle Notebook mới với:

- Accelerator: **Tesla T4 x2**
- Internet: **ON**
- Model attach: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- Canonical text-to-image demo không cần dataset bổ sung

Model path dự kiến:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

## Source bootstrap

Notebook clone repository này vào `/kaggle/working` và xác minh public source ref đã chọn trước khi import project module.

Sau đó runtime lấy đúng upstream Mage commit đã pin và xác minh cả:

- upstream commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`

Bootstrap dependency `loguru==0.7.3` cũng được xác minh bằng exact wheel size và SHA-256 trước khi cài vào bootstrap target biệt lập.

## Quy trình chạy khuyến nghị

1. Tạo Kaggle Notebook hoàn toàn mới.
2. Chọn **GPU T4 x2**.
3. Bật Internet.
4. Attach Kaggle model bắt buộc.
5. Import/mở `notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb`.
6. Bắt đầu bằng fresh kernel.
7. Chạy **Run All đúng một lần**.
8. Giữ nguyên notebook output và evidence directory được sinh ra.

Không cần upload publication ZIP hoặc identity JSON thủ công. Public workflow dùng Git source bootstrap.

## Acceptance signal

Một run thành công phải thiết lập, cùng các check khác:

- đúng hai Tesla T4;
- model load count bằng một;
- một T2I trajectory;
- cả GPU0 và GPU1 tham gia;
- bốn transformer invocation cho profile bốn bước;
- bốn transformer boundary transfer 0->1 dự kiến;
- bốn transformer return 1->0;
- một VAE-input transfer 0->1;
- BF16 materialization PASS;
- không quan sát CPU fallback;
- output RGB 512x512 hợp lệ.

## Failed run

Nếu heavy cell lỗi, hãy giữ nguyên failed attempt. Với strict publication provenance, tạo **Kaggle session mới** thay vì chỉnh state, xóa runtime directory hoặc rerun riêng failed cell.

## Notebook và authority

Public notebook là bề mặt reproducibility/publication. Closed R2G authority evidence vẫn là record qualification routing/runtime chính thức và không bị ghi lại bởi public run mới.
