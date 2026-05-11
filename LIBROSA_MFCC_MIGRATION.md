# Tổng hợp thay đổi sau khi bỏ `librosa` và `MFCC`

## 1. Mục tiêu

Chuyển hệ thống tìm kiếm âm thanh sang hướng **ít phụ thuộc thư viện hơn**, đồng thời vẫn giữ được pipeline trích xuất đặc trưng, truy vấn và xếp hạng kết quả.

## 2. Những gì đã bỏ

- Bỏ hoàn toàn `librosa`.
- Bỏ cách trích đặc trưng dựa trên `MFCC`.
- Bỏ `matplotlib` trong phần hiển thị kết quả.

## 3. Những gì đã thay thế

### 3.1. Tiền xử lý âm thanh

- Dùng `soundfile` để đọc file âm thanh.
- Dùng `numpy` và `scipy.signal.resample_poly` để:
  - chuyển về mono,
  - resample về 16 kHz,
  - chuẩn hóa độ dài 5 giây.

### 3.2. Trích đặc trưng

Thay `MFCC` bằng **ma trận phổ 24 băng tần theo thời gian**:

- Tính STFT bằng `scipy.signal.stft`.
- Chia phổ thành 24 dải tần.
- Lấy log-energy cho từng dải để tạo:
  - `feature_vector` 24 chiều cho tìm kiếm vector,
  - `spectral_matrix` theo frame cho DTW.

Ngoài ra, hệ thống vẫn giữ và tính các đặc trưng phổ quan trọng theo tinh thần bài giảng:

- `spectral centroid`: trung tâm năng lượng của phổ, phản ánh âm sắc sáng/tối.
- `spectral bandwidth`: độ phân tán của năng lượng quanh centroid, phản ánh phổ rộng hay hẹp.
- `spectral distribution`: phân bố năng lượng theo các dải tần, được biểu diễn trực tiếp qua 24 băng tần log-energy.

### 3.3. Đặc trưng miền thời gian

Vẫn giữ các đặc trưng:

- `energy`
- `zcr`
- `silence ratio`
- `frame statistics`

### 3.4. Đặc trưng miền tần số

Vẫn giữ và tính các đặc trưng phổ quan trọng theo tinh thần bài giảng:

- `spectral centroid`: trung tâm năng lượng của phổ, phản ánh âm sắc sáng/tối.
- `spectral bandwidth`: độ phân tán của năng lượng quanh centroid, phản ánh phổ rộng hay hẹp.
- `spectral distribution`: phân bố năng lượng theo các dải tần, được biểu diễn trực tiếp qua 24 băng tần log-energy.

### 3.5. So sánh / xếp hạng

- Giữ `cosine` và `euclidean` cho bước tìm ứng viên nhanh.
- Giữ `DTW` tự cài đặt để re-rank theo chuỗi frame.
- `DTW` không còn phụ thuộc `librosa`.

### 3.6. Giao diện

- Bỏ `matplotlib`.
- Dùng SVG inline để vẽ waveform và spectrogram.

## 4. Các file đã ảnh hưởng chính

- `src/feature_extraction.py`
  - Thay preprocessing và feature extraction.
  - Thay MFCC bằng 24-band spectral features.
  - Tự cài DTW.
  - Thêm frame-level silence threshold.

- `src/retrieval.py`
  - Dùng feature mới cho indexing và search.
  - Giữ vector search + DTW re-ranking.

- `src/database.py`
  - Lưu vector 24 chiều và spectral matrix JSON.
  - Giữ tên cột cũ `mfcc_matrix` để tương thích dữ liệu cũ.

- `app.py`
  - Bỏ `matplotlib`.
  - Hiển thị waveform / spectrogram bằng SVG.
  - Thêm xử lý lỗi thân thiện cho file upload không hỗ trợ.

- `requirements.txt`
  - Gỡ `librosa` và `matplotlib`.
  - Giữ các thư viện tối thiểu cần cho pipeline mới.

## 5. Tác động thực tế

- Ít phụ thuộc thư viện hơn, dễ chạy hơn trên máy cục bộ.
- Pipeline vẫn có 2 tầng:
  - tìm nhanh bằng vector 24 chiều,
  - re-rank bằng DTW.
- Chất lượng xếp hạng top-5 vẫn ổn trên dataset hiện tại.
- DTW vẫn là bước chậm nhất, nên cần tune `dtw_candidate_pool` nếu ưu tiên latency.

## 6. Ghi chú quan trọng

- `MFCC` không còn là đặc trưng chính, nhưng hệ thống vẫn làm việc theo đúng ý tưởng so khớp âm thanh theo đặc trưng số.
- Nếu muốn bám sát bài giảng hơn, có thể trình bày các đặc trưng miền thời gian và miền tần số là phần chính, còn DTW là phần mở rộng thực nghiệm.
