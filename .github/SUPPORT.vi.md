# Hỗ trợ

> 🌐 Language / Ngôn ngữ: [English](SUPPORT.md) | **Tiếng Việt**

Đây là dự án engineering/reproducibility mã nguồn mở và không cung cấp commercial support SLA.

## Nơi gửi yêu cầu

- Dùng bug-report form cho lỗi có thể tái hiện.
- Dùng feature-request form cho cải tiến có phạm vi rõ ràng.
- Dùng question form cho câu hỏi setup/usage.
- Dùng quy trình trong [SECURITY.vi.md](SECURITY.vi.md) cho lỗ hổng bảo mật.

## Thông tin nên cung cấp

Khi cần hỗ trợ kỹ thuật, hãy cung cấp:

- repository commit SHA hoặc release tag;
- phiên bản Python/PyTorch nếu liên quan;
- môi trường (Kaggle/local/khác);
- accelerator topology;
- Internet có bật hay không;
- attached model path;
- first failed gate;
- đoạn log tối thiểu đã loại bỏ secret.

Với GPU claim, hãy nêu rõ session có thực sự là Tesla T4 x2 hay không. Output CPU-only không thể chứng minh live dual-T4 routing behavior.
