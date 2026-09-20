# Giới hạn

> 🌐 Language / Ngôn ngữ: [English](limitations.md) | **Tiếng Việt**

## Phạm vi phần cứng

Public path đã qualification nhắm tới **đúng hai GPU NVIDIA Tesla T4**. GPU khác, một T4, CPU-only, P100, TPU hoặc topology accelerator khác không nằm trong public dual-T4 acceptance claim.

## Cách mô tả precision

Phát biểu được hỗ trợ là:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

Điều này không có nghĩa Tesla T4 cung cấp native BF16 Tensor Core acceleration.

## Public demo profile cố định

Public runner là bề mặt reproducibility/qualification, không phải image-generation API tổng quát. Canonical demo cố định:

- prompt;
- resolution 512x512;
- bốn denoising steps;
- CFG 1.0;
- seed 42;
- 12 transformer blocks;
- split block 1.

Thay đổi các giá trị này có thể hữu ích cho thử nghiệm, nhưng các run đó nằm ngoài exact public acceptance profile nếu chưa được qualification riêng.

## Phạm vi model

Repository nhắm tới Mage-Flow-Turbo model ID/revision đã pin trong authority evidence. Dự án không tuyên bố tương thích với mọi Mage model, Edit-Turbo, model revision tương lai hoặc diffusion architecture khác.

## Internet và external availability

Public Git bootstrap cần Internet để lấy repository/upstream/dependency content. Availability của GitHub, PyPI và Kaggle model attachment nằm ngoài quyền kiểm soát của repository này.

## Không có service SLA

Đây là repository engineering demo và reproducibility. Dự án không cung cấp:

- hosted REST service;
- multi-tenant isolation;
- cam kết availability/SLA;
- production autoscaling;
- authentication/authorization infrastructure;
- bảo đảm tương thích dài hạn cho internal module `mage_t4x2`.

## Authority và môi trường mới

CPU/static CI có thể phát hiện nhiều regression nhưng không thay thế real T4x2 qualification cho GPU claim. Tương tự, run thành công trên phần cứng khác không phải evidence trực tiếp cho qualified Kaggle T4x2 topology.

## License upstream và model

MIT license của repository này áp dụng cho code và tài liệu nguyên bản thuộc repository. Upstream Mage source và model artifact vẫn chịu license và điều khoản riêng của chúng.
