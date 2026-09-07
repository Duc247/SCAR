# Original User Request

## 2026-09-07T08:54:05Z

Thực hiện rà soát toàn diện logic mã nguồn, kiểm chứng toán học, khắc phục triệt để các lỗi phát hiện và bổ sung test case cho repository SCAR (Multimodal MyoPS-380 segmentation: CINE/LGE/T2w với CMSPA-Net).

Working directory: d:\NCKH\SCAR
Integrity mode: development

## Verification Resources
- Bộ unit tests hiện hữu: `python -m pytest` tại thư mục `tests/` (hiện có 56 bài test đã pass).
- Công cụ kiểm tra chu trình forward/backward/AdamW: `python tools/sanity_check.py --profile testing`.
- Công cụ kiểm tra parity toán học với repo gốc: `python tools/verify_parity.py`.
- Tài liệu kiến trúc và đặc tả toán học chi tiết: `README.md`.

## Requirements

### R1. Rà soát Data Contract & Pipeline Tiền xử lý
- Kiểm tra tính đồng bộ 3 modality (bSSFP/CINE, LGE, T2w) trên cả 2D slice (`.npz`) và 3D volume (`.h5`).
- Xác minh thuật toán hoán vị nhãn canonical `[0, 1, 2, 3]` vs legacy `[0, 1, 3, 2]`, ngăn chặn việc mapping sai nhãn lớp 2 (phù nề) và lớp 3 (sẹo).
- Đảm bảo việc chia split bệnh nhân (manifests) diễn ra ở mức ca bệnh (patient-level), tuyệt đối không rò rỉ (data leakage) giữa train, validation và test.

### R2. Thẩm định Kiến trúc CMSPA-Net & Các Khối Attention
- Kiểm tra module SSPANet: Xác thực công thức Strip Pooling RMS `sqrt(mean(x²)+eps)` với Conv 1×3 / 3×1 và Channel Attention Conv 7×7 khớp với mô tả Hình 2 trong bài báo.
- Kiểm tra module CMSPA Bottleneck: Xác thực Anatomy gate từ CINE và Pathology gate từ độ lệch chuẩn std theo kênh C của LGE + T2w.
- Kiểm tra 3 tầng Skip Fusion và Decoder 4 tầng, đảm bảo gradient flow thông suốt qua toàn bộ 3 nhánh encoder và bottleneck mà không có layer nào bị ngắt kết nối.

### R3. Thẩm định Hàm Loss, Metrics & Tính ổn định số
- Kiểm tra `SegmentationLoss` và `DiceLoss`, đặc biệt trong chế độ Mixed Precision (AMP FP16): đảm bảo các phép tính tổng lũy kế, căn bậc hai và mẫu số Dice không gây tràn số, chia cho 0, hoặc phát sinh NaN/Inf.
- Kiểm tra ma trận nhầm lẫn (`ConfusionMeter`) và khoảng cách bề mặt (`SurfaceDistance` / HD95, ASD): đảm bảo xử lý an toàn giá trị `null` khi thiếu thông tin voxel spacing vật lý thay vì tự động giả định 1.0mm.

### R4. Thẩm định Trainer & Vòng lặp Huấn luyện
- Kiểm tra logic vòng lặp train/val, tích lũy gradient (`accum_steps`), gradient clipping (`clip_grad = 1.0`), cơ chế lưu checkpoint (`best.pt`, `latest.pt`) và tính năng phục hồi huấn luyện (`resume`).
- Xác thực tính lặp lại (reproducibility) thông qua seed ngẫu nhiên và cấu hình thiết bị (`cpu` / `cuda`).

### R5. Khắc phục Lỗi & Bổ sung Test Case
- Với mọi lỗi logic, sai lệch toán học hoặc nguy cơ tràn số phát hiện được, thực hiện sửa đổi trực tiếp vào mã nguồn theo đúng chuẩn kiến trúc.
- Bổ sung unit tests tương ứng vào thư mục `tests/` để bảo đảm các lỗi này không bị tái phát (regression testing).

## Acceptance Criteria

### Tính đúng đắn toán học & độ ổn định
- [ ] Tất cả công thức trong `SSPANet`, `CMSPA`, `SegmentationLoss`, `DiceLoss`, `ConfusionMeter`, `SurfaceDistance` khớp chính xác với đặc tả thiết kế trong `README.md`.
- [ ] Không có lỗi ngắt kết nối đồ thị gradient (disconnected graph) hay lỗi NaN/Inf dưới cả CPU và CUDA (AMP FP16).
- [ ] `python -m pytest` đạt 100% pass trên toàn bộ test suite hiện có và các test case mới được bổ sung.
- [ ] `python tools/sanity_check.py --profile testing` và `--profile production` hoàn thành thành công mà không có lỗi.
- [ ] Báo cáo rà soát chi tiết được cập nhật đầy đủ, ghi nhận rõ nguyên nhân và phương án xử lý cho từng vấn đề được tìm thấy.
