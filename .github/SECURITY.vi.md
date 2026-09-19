# Chính sách bảo mật

> 🌐 Language / Ngôn ngữ: [English](SECURITY.md) | **Tiếng Việt**

## Báo cáo

Không mở public issue cho lỗ hổng nghi ngờ, credential bị lộ, xử lý artifact không an toàn, dependency compromise, command-injection path hoặc phát hiện nhạy cảm về bảo mật khác.

Dùng GitHub private vulnerability reporting khi repository đã bật. Nếu chưa có, gửi email tới `i.am@dangkhoa.dev` với:

- mô tả ngắn gọn vấn đề;
- component và revision bị ảnh hưởng;
- bước tái hiện hoặc proof of concept;
- tác động dự kiến;
- mitigation đề xuất nếu đã biết.

Không gửi credential production thật hoặc dữ liệu riêng tư của bên thứ ba.

## Ranh giới bảo mật riêng của dự án

Repository này không phải hosted internet service. Public execution surface chính là Kaggle notebook cùng local Python/CLI runner.

Các boundary quan trọng:

- model path và project source path được xác minh tường minh;
- CPU fallback bị cấm trong canonical GPU contract;
- upstream source và bootstrap wheel identity được pin/xác minh;
- generated evidence có thể chứa environment path và cần review trước khi chia sẻ công khai;
- notebook dùng Internet nên availability/supply-chain assumption của GitHub/PyPI có ý nghĩa.

## Phiên bản được hỗ trợ

Security fix nhắm tới nhánh `main` hiện tại và, sau stable release, stable release mới nhất còn được hỗ trợ khi phù hợp. Historical commit không được đảm bảo backport.
