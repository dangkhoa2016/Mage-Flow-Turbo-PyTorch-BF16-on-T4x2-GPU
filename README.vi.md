# Mage-Flow-Turbo PyTorch BF16 trên GPU T4×2

<p align="center">
  <a href="https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dangkhoa2016/Mage-Flow-Turbo-PyTorch-BF16-on-T4x2-GPU/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-upstream-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-Tesla%20T4%20%C3%972-76B900?logo=nvidia&logoColor=white">
  <img alt="Precision" src="https://img.shields.io/badge/Precision-BF16%20materialization-2563EB">
  <img alt="Attention" src="https://img.shields.io/badge/Attention-SDPA-7C3AED">
  <img alt="Authority" src="https://img.shields.io/badge/Runtime-QUALIFIED-0A7E07">
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-T4%20%C3%972-20BEFF?logo=kaggle&logoColor=white">
</p>

> 🌐 Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Dự án engineering có khả năng tái lập để chạy **Mage-Flow-Turbo** bằng pipeline **PyTorch upstream** trên **Kaggle Tesla T4 ×2**, đồng thời giữ **một lần load model logic duy nhất** và **một luồng text-to-image duy nhất đi qua cả hai GPU** (Lane A — kiểm định chuẩn), bên cạnh đó chạy thêm **một lane showcase production đa-ảnh bền vững tách biệt** (Lane B) với một lần load model riêng của chính nó.

Đây là dự án engineering độc lập xây dựng quanh model/runtime Microsoft Mage upstream. Dự án **không phải bản phát hành chính thức của Microsoft**.

---

## Tổng quan nhanh

| Hạng mục | Profile public đã qualification |
|---|---|
| Runtime | Upstream Mage / PyTorch |
| Accelerator | Đúng **2 × NVIDIA Tesla T4** |
| Precision | **BF16 dtype/materialization** |
| Attention | **SDPA** |
| Kiểu inference | Một logical T2I trajectory |
| Số lần load model | Đúng **1** |
| Transformer split | Block 0 trên `cuda:0`; block 1–11 trên `cuda:1` |
| VAE | `cuda:1` |
| Resolution demo | RGB `512×512` (Lane A) |
| Steps / CFG / Seed | `4` / `1.0` / `42` |
| Showcase lane | Đa-ảnh bền vững (12–24 ảnh, `512/768/1024`, seeds `1001+`) |
| Showcase verdict | `PUBLIC_SHOWCASE_FINAL_VERDICT` (fail-closed) |
| CPU fallback | Bị cấm bởi canonical contract |
| Quantization | Không |
| sd.cpp / sd-cli | Không sử dụng |
| Runtime authority | **qualified CLOSED_PASS** |
| License | MIT cho code/docs nguyên bản của repository |

Cách mô tả precision được hỗ trợ là:

> **BF16 dtype/materialization + functional upstream PyTorch execution on Tesla T4**

Dự án **không** tuyên bố Tesla T4 có native BF16 Tensor Core acceleration.

---

## Vì sao dự án này tồn tại

Load Mage-Flow-Turbo trên một accelerator lớn có thể khá trực tiếp, nhưng xây dựng một **single-trajectory dual-T4 execution path** có khả năng tái lập lại là một bài toán engineering khác.

Repository này tập trung rõ ràng vào bài toán đó:

- giữ **một logical model instance**;
- giữ **một prompt / latent / scheduler trajectory**;
- chia transformer qua cả hai T4 thay vì chạy hai job độc lập;
- làm cho cross-device transfer có thể quan sát và kiểm chứng;
- fail-closed nếu hardware, source identity, attention backend, routing, precision hoặc output sai contract;
- giữ một qualified runtime baseline có thể xác minh theo byte;
- cung cấp public Kaggle workflow không phụ thuộc ZIP nội bộ hoặc hidden setup.

Mục tiêu không chỉ là sinh được một ảnh. Mục tiêu là làm execution path **có thể kiểm tra, tái lập và review**.

---

## Repository này chứng minh điều gì

- Chạy bằng PyTorch upstream, không dùng `stable-diffusion.cpp` / `sd-cli`.
- BF16 dtype/materialization cho text encoder, transformer và VAE.
- Chỉ một lần load model logic cho mỗi lane (tổng hai lần load độc lập: một cho Lane A, một cho Lane B — không bao giờ dùng chung).
- Chỉ một logical T2I trajectory phân bổ qua hai GPU T4 trong Lane A chuẩn.
- Một lane showcase production bền vững thứ hai (Lane B) giữ model nóng xuyên suốt workload xác định đa-prompt, đa-seed, đa-độ-phân-giải (`512 / 768 / 1024`) với evidence timing, đỉnh GPU allocated/reserved, routing và SHA-256 output cho từng ảnh, cùng gallery contact sheet có nhãn.
- Transformer block 0 trên `cuda:0`; block 1–11 và output head trên `cuda:1`.
- Transformer boundary transfer `cuda:0 -> cuda:1` tường minh.
- Transformer result return `cuda:1 -> cuda:0` tường minh.
- VAE input được chuyển từ `cuda:0` sang `cuda:1`.
- Attention dùng SDPA, T4 path không phụ thuộc `flash_attn`.
- Provenance, placement, routing, precision và output check theo fail-closed.
- Public notebook bootstrap source từ Git thay vì internal publication archive.
- Lane showcase fail-closed: một gallery đẹp không bao giờ che được một canonical qualification thất bại.
- CPU/static CI xác minh contract nhưng không giả vờ thay thế real T4×2 evidence.

---

## Luồng thực thi end-to-end

```text
Fresh Kaggle Notebook
        |
        |  T4 ×2 + Internet ON
        v
Clone repository này
        |
        +--> xác minh public source ref
        |
        +--> xác minh frozen qualified runtime baseline
        |
        +--> lấy đúng upstream Mage commit/tree đã pin
        |
        +--> xác minh bootstrap wheel đã pin
        |
        v
Resolve owner-qualified Kaggle model mount
        |
        v
Freeze SDPA trước model construction
        |
        v
Load MỘT model instance (Lane A)
        |
        v
Áp dụng explicit dual-device placement
        |
        +--> text encoder ---------------- cuda:0
        +--> transformer block 0 --------- cuda:0
        +--> transformer blocks 1..11 ---- cuda:1
        +--> norm_out / proj_out --------- cuda:1
        +--> VAE ------------------------- cuda:1
        |
        v
Reassert SDPA
        |
        v
Chạy một T2I trajectory 4 steps
        |
        v
Reduce telemetry + dtype + output evidence
        |
        v
Lane A PASS / FAIL và giữ nguyên evidence
        |
        v
LANE B  (production showcase)
        |
        v
Resolve owner-qualified showcase model path
        |
        v
Load THÊM MỘT model instance (độc lập với Lane A)
        |
        v
Áp dụng explicit dual-device placement (model nóng)
        |
        v
Chạy lịch đa-ảnh xác định (512/768/1024)
        |
        v
Telemetry từng ảnh + đỉnh GPU + SHA-256 + gallery
        |
        v
Lane B PASS / FAIL và giữ nguyên evidence
```

Lane showcase chạy **sau** và **độc lập** với Lane A chuẩn; nó không bao giờ ghi đè evidence Lane A.

---

## Runtime topology

```text
Prompt
  |
Text encoder ------------------------ cuda:0
  |
Transformer block 0 ---------------- cuda:0
  |
activation transfer 0 -> 1
  |
Transformer blocks 1..11 ----------- cuda:1
norm_out / proj_out ----------------- cuda:1
  |
transformer result 1 -> 0
  |
latent / scheduler path ------------- cuda:0
  |
VAE input 0 -> 1
  |
VAE --------------------------------- cuda:1
  |
512×512 RGB image
```

Với canonical demo 4 steps (Lane A), chuỗi transformer `[0..11]` phải xuất hiện **bốn lần**, mỗi lần tương ứng một denoising invocation. Lane B dùng lại đúng placement T4 kép trong khi model giữ nóng xuyên suốt lịch showcase.

---

## Chạy nhanh trên Kaggle

Public notebook được thiết kế để người dùng **không cần** internal ZIP, identity JSON hay private source bundle.

### 1. Tạo Kaggle Notebook mới

Cấu hình:

- Accelerator: **GPU T4 ×2**
- Internet: **ON**
- Bắt đầu bằng fresh kernel.

### 2. Attach Kaggle Model bắt buộc

Attach:

```text
dangkhoa2016/mage-flow-community-mage-flow-turbo
```

Đường dẫn read-only dự kiến:

```text
/kaggle/input/models/dangkhoa2016/
mage-flow-community-mage-flow-turbo/
pytorch/default/1
```

Canonical text-to-image demo không cần dataset bổ sung.

### 3. Mở public notebook

```text
notebooks/kaggle-production-demo-mage-flow-turbo-bf16-t4x2.ipynb
```

Notebook clone repository này vào `/kaggle/working`, xác minh source identity, bootstrap đúng upstream Mage source và dependency đã pin, resolve Kaggle model mount, rồi chạy hai lane: kiểm định chuẩn **Lane A** và showcase production bền vững **Lane B**.

Với publication-quality run, dùng **Run All đúng một lần**.

Xem [Hướng dẫn Kaggle public demo](docs/kaggle-public-demo.vi.md) và [Ghi chú thiết lập Kaggle](docs/kaggle.vi.md).

---

## Python và CLI entrypoint

### Python

```python
from public_demo.runner import run_public_demo

result = run_public_demo(
    project_root="/path/to/repository",
)

print(result["status"])
print(result["summary_path"])
```

Canonical runner (`public_demo/runner.py`) cố ý không nhận bất kỳ override nào cho prompt/profile. Prompt, seed, steps, CFG, resolution, topology block/split, attention SDPA, dtype BF16, model identity và owner-qualified model path đều cố định bởi canonical public contract trong `public_demo/contract.py`; input không canonical sẽ fail-closed.

Showcase runner (`public_demo/showcase.py`) là đối tác Lane B: cũng không nhận override nào, tự thực hiện một lần load model riêng và chạy bảng showcase xác định (`s01..s24`, `512/768/1024`, seeds `1001..1024`) với evidence fail-closed từng ảnh và manifest.

```python
from public_demo.showcase import run_public_showcase

showcase = run_public_showcase(project_root="/path/to/repository")
print(showcase["status"])       # PASS / FAIL
print(showcase["summary_path"]) # artifacts/public-showcase/<run_id>/showcase-summary.json
```

### CLI

```bash
python scripts/public_kaggle_demo.py \
  --project-root /path/to/repository \
  --output-root /path/to/output
```

CLI in ordered gate verdict, run ID, final status và summary path.

Xem [Python và CLI API](docs/api.vi.md) và [Cách sử dụng](docs/usage.vi.md).

---

## Lane A — Cấu hình demo canonical

| Thiết lập | Giá trị |
|---|---|
| Model | `dangkhoa2016/mage-flow-community-mage-flow-turbo` |
| Resolution | `512×512` |
| Steps | `4` |
| CFG | `1.0` |
| Seed | `42` |
| Attention | `sdpa` |
| Precision | BF16 dtype/materialization |
| GPU | Tesla T4 ×2 |
| `split_block` | `1` |
| Transformer blocks | `12` |
| CPU fallback | tắt |
| Model load count | `1` |

## Lane B — Cấu hình showcase

Showcase là một lane riêng với một lần load model của chính nó và workload xác định, do repository kiểm soát:

| Thiết lập | Giá trị |
|---|---|
| Cases | `s01..s24` (12 mandatory + 12 extension) |
| Mandatory minimum | `12` |
| Maximum | `24` |
| Resolutions | `512`, `768`, `1024` |
| Steps / CFG | `4` / `1.0` |
| Seeds | `1001..1024` |
| Target inference window | `300 s` |
| Outputs | `outputs/<case>-<res>x<res>-seed<seed>.png`, SHA-256 từng ảnh |
| Telemetry | `telemetry/<case>.jsonl` |
| Summary | `showcase-summary.json`, `case-results.json`, `showcase-runtime.log`, manifest tại `artifacts/public-showcase/<run_id>/` |
| Final key | `PUBLIC_SHOWCASE_FINAL_VERDICT` |

---

## Lane A — Acceptance matrix

Closed qualified authority đã thiết lập các fact cốt lõi sau:

| Acceptance item | Qualified status |
|---|---|
| qualified runtime authority | `CLOSED_PASS` |
| T4 ×2 preflight | PASS |
| Single model load | `MODEL_LOAD_COUNT=1` |
| Single T2I trajectory | PROVEN |
| GPU0 participation | PROVEN |
| GPU1 participation | PROVEN |
| Transformer invocations | 4 / 4 |
| Transformer boundary transfer 0 → 1 | 4 / 4 |
| Transformer return 1 → 0 | 4 / 4 |
| VAE input transfer 0 → 1 | 1 / 1 |
| BF16 materialization | PASS |
| CPU fallback | NOT_OBSERVED |
| Output validation | PASS |
| Output profile | RGB 512×512 |

Public notebook là bề mặt reproducible demonstration; nó **không** ghi lại frozen authority evidence.

### Lane B — tóm tắt acceptance

Lane showcase phải thỏa mãn:

```text
SHOWCASE_MODEL_LOAD_COUNT=1
ROUTING=PASS           (block order, participation, một lane instance, không CPU fallback)
MIN_CASES=PASS         (>= 12 case đã chạy)
OUTPUT_VALIDATION=PASS (mọi ảnh hợp lệ, adapter detach sau mỗi case)
GALLERY=PASS
DIGEST_VERIFY=PASS     (mọi SHA-256 ghi lại khớp artifact)
MANIFEST_BUILD=PASS
PUBLIC_SHOWCASE_FINAL_VERDICT=PASS | FAIL | NOT_RUN
```

---

## Chuỗi reproducibility

Dự án pin và xác minh các identity quan trọng cho qualified path:

- Model ID: `dangkhoa2016/mage-flow-community-mage-flow-turbo`
- Model revision: `65bb3500f0da9df6a41ec6383716fc02cf014773`
- Upstream Mage repository: `https://github.com/microsoft/Mage`
- Upstream Mage commit: `76bec2bb3818863f470de7e867c2dc7f1d0bfd83`
- Upstream `mage_flow` tree: `946b91bcb2cac75e6cfe8399f0f7f330a2280adf`
- qualified runtime baseline manifest SHA-256: `b829fdea0b568ff80ff74d1c58344d7f289d781d42029f2b42a34a8d9c266466`
- Bootstrap dependency: `loguru==0.7.3`
- Bootstrap wheel SHA-256: `31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c`

Runtime baseline lưu byte size + SHA-256 của qualified runtime files. Public runner xác minh baseline này trước live inference.

Xem [Provenance và authority](docs/provenance.vi.md).

---

## Safety và fail-closed behavior

Public path dừng lại thay vì âm thầm nới lỏng contract nếu, trong số các trường hợp khác:

- project checkout cũ hoặc không đúng source ref đã chọn;
- upstream Mage commit/tree identity khác pin;
- bootstrap wheel khác size hoặc SHA-256;
- qualified runtime baseline không khớp;
- không có đúng hai Tesla T4;
- thiếu owner-qualified Kaggle model attachment;
- SDPA không hiệu lực sau model construction hoặc ngay trước inference;
- model load count khác một;
- block order hoặc transfer cardinality sai contract;
- một GPU không tham gia như dự kiến;
- quan sát thấy CPU fallback;
- BF16 live materialization không đúng qualified profile;
- generated output không phải ảnh RGB 512×512 hợp lệ;
- showcase chạy ít hơn mandatory minimum, bất kỳ ảnh showcase nào fail validation, routing facts không đủ xanh, digest output ghi lại sai khớp artifact, hoặc showcase manifest không thể build.

Thiết kế này cố ý ưu tiên FAIL rõ ràng hơn một PASS gây hiểu nhầm.

---

## Cấu trúc repository

```text
.
├── mage_t4x2/        qualified runtime, routing, placement và evidence modules
├── public_demo/      public Git/PyPI bootstrap, reproducible demo runner và showcase lane
├── scripts/          CLI, session orchestration và authority drivers
├── authority/        frozen runtime baselines và provenance manifests
├── notebooks/        bilingual public Kaggle production demo
├── docs/             architecture, usage, API, Kaggle, limits và provenance
├── tests/            CPU/static contract và public workflow tests
├── .github/          CI, issue forms, security, support và contribution policy
└── LICENSE           MIT license cho original work của repository
```

---

## Tài liệu

| Chủ đề | Tiếng Việt |
|---|---|
| Kiến trúc | [docs/architecture.vi.md](docs/architecture.vi.md) |
| Cách sử dụng | [docs/usage.vi.md](docs/usage.vi.md) |
| Python & CLI API | [docs/api.vi.md](docs/api.vi.md) |
| Thiết lập Kaggle | [docs/kaggle.vi.md](docs/kaggle.vi.md) |
| Kaggle public demo | [docs/kaggle-public-demo.vi.md](docs/kaggle-public-demo.vi.md) |
| Giới hạn | [docs/limitations.vi.md](docs/limitations.vi.md) |
| Provenance / authority | [docs/provenance.vi.md](docs/provenance.vi.md) |
| Testing / validation | [docs/testing.vi.md](docs/testing.vi.md) |
| Đóng góp | [.github/CONTRIBUTING.vi.md](.github/CONTRIBUTING.vi.md) |
| Bảo mật | [.github/SECURITY.vi.md](.github/SECURITY.vi.md) |
| Hỗ trợ | [.github/SUPPORT.vi.md](.github/SUPPORT.vi.md) |

Mỗi project Markdown guide đều có bản English tương ứng.

---

## Phạm vi và non-goals

Repository này cố ý giữ phạm vi hẹp.

Dự án **không** tuyên bố:

- hỗ trợ mọi GPU topology;
- đã qualification trên P100, TPU, CPU-only hoặc GPU không liên quan;
- tương thích tổng quát với mọi Mage model/revision;
- cung cấp hosted REST service hoặc production SLA;
- có multi-user serving, autoscaling hoặc authentication;
- Tesla T4 có native BF16 Tensor Core acceleration;
- CPU/static CI chứng minh real dual-T4 execution.

Xem [Giới hạn](docs/limitations.vi.md) để biết boundary chính xác.

---

## Testing

CPU/static validation:

```bash
python -m compileall -q mage_t4x2 public_demo scripts
python -m pytest -q
```

CI xác minh static/public contract trên các Python version được hỗ trợ, nhưng real GPU claim vẫn gắn với real **Tesla T4 ×2** evidence.

Xem [Testing và validation](docs/testing.vi.md).

---

## Đóng góp, bảo mật và hỗ trợ

Contribution được chào đón khi vẫn giữ provenance rõ ràng và nói rõ thay đổi có ảnh hưởng qualified runtime bytes hay không.

- [Hướng dẫn đóng góp](.github/CONTRIBUTING.vi.md)
- [Quy tắc ứng xử](.github/CODE_OF_CONDUCT.vi.md)
- [Chính sách bảo mật](.github/SECURITY.vi.md)
- [Chính sách hỗ trợ](.github/SUPPORT.vi.md)

Security-sensitive finding nên dùng private reporting path trong security policy thay vì public issue.

---

## License và upstream code

Code và tài liệu nguyên bản của repository này được phát hành theo [MIT License](LICENSE).

Copyright (c) 2026 **Đăng Khoa <i.am@dangkhoa.dev>**.

Upstream Mage source được lấy từ public Git commit đã pin thay vì copy vào repository này. Upstream source và model artifact vẫn chịu license và điều khoản riêng; hãy kiểm tra trước khi redistribution hoặc production use.
