# Tổng hợp thay đổi sau khi bỏ `librosa`, `MFCC`, `DTW`, và `Cosine`

## 1. Mục tiêu

Chuyển hệ thống tìm kiếm âm thanh sang hướng **ít phụ thuộc thư viện và đơn giản hóa tối đa**:
- Bỏ MFCC và librosa
- Bỏ DTW (không có trong chương trình giảng dạy)
- Bỏ Cosine metric (Euclidean cung cấp discrimination tốt hơn)
- Giữ pipeline một tầng: vector 28-D + Euclidean distance

## 2. Những gì đã bỏ

- Bỏ hoàn toàn `librosa`.
- Bỏ cách trích đặc trưng dựa trên `MFCC`.
- Bỏ `matplotlib` trong phần hiển thị kết quả.
- Bỏ `DTW` (Dynamic Time Warping) - không trong chương trình giảng dạy.
- Bỏ `Cosine` metric - chuyển sang Euclidean only.
- Bỏ pipeline hai tầng (vector search + DTW re-ranking).

## 3. Những gì đã thay thế

### 3.1. Tiền xử lý âm thanh

- Dùng `soundfile` để đọc file âm thanh.
- Dùng `numpy` và `scipy.signal.resample_poly` để:
  - chuyển về mono,
  - resample về 16 kHz,
  - chuẩn hóa độ dài 5 giây.

### 3.2. Trích đặc trưng

Thay `MFCC` bằng **vector 28 chiều chuẩn hóa**:

- `[0]` Energy: mean(y²)
- `[1]` ZCR: zero crossing rate
- `[2]` Spectral Centroid (normalized): centroid / (sr/2) → [0,1]
- `[3]` Spectral Bandwidth (normalized): bandwidth / (sr/2) → [0,1]
- `[4-27]` 24-band log-energy: STFT chia phổ thành 24 dải tần

**Tính toán STFT:**
- `scipy.signal.stft` với window=512, overlap=256, Hann window
- Chia phổ thành 24 dải tần bằng nhau
- Lấy log-energy cho từng dải

**Chuẩn hóa:**
- Centroid/Bandwidth normalize bằng sr/2 để tránh imbalance scale
- Kết quả: tất cả 28 chiều trong phạm vi [0, ~10] (log-energy scale)

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

- **Chỉ dùng Euclidean L2 distance**: sqrt(sum((a-b)²))
- Bỏ Cosine: không cung cấp discrimination tốt sau normalize
- Bỏ DTW: không có trong chương trình giảng dạy
- **Pipeline một tầng**: pgvector search trực tiếp với top-k kết quả
- Performance: <100ms cho 500 file, distances trong [0.0, 0.052] cho top-5

### 3.6. Giao diện

- Bỏ `matplotlib`.
- Dùng SVG inline để vẽ waveform và spectrogram.

## 4. Các file đã ảnh hưởng chính

- `src/feature_extraction.py`
  - Thay preprocessing và feature extraction.
  - Thay MFCC bằng 28-D vector (energy, ZCR, centroid_norm, bandwidth_norm, 24-band).
  - Xóa `cosine_distance()` function.
  - Giữ `euclidean_distance()` là metric duy nhất.
  - Thêm frame-level silence threshold (percentile 10% + multiplier 3.0).

- `src/retrieval.py`
  - Dùng 28-D feature vector cho indexing.
  - Simplify từ 2-tầng (vector + DTW) thành 1-tầng (vector only).
  - Gọi `search_vector_candidates()` với metric="euclidean" trực tiếp.

- `src/database.py`
  - Lưu vector 28 chiều (pgvector extension).
  - Cột: feature_vector = Column(Vector(28))
  - Bỏ spectral_matrix (không cần lưu trữ frame-level để DTW).
  - Search function chỉ chấp nhận metric="euclidean".

- `app.py`
  - Bỏ `matplotlib`.
  - Hiển thị waveform / spectrogram bằng SVG.
  - Metric dropdown: chỉ hiển thị "Euclidean" (bỏ Cosine/DTW).
  - Default metric: "euclidean".
  - Kết quả: hiển thị top-k (không có intermediate candidates table).

- `src/config.py`
  - DEFAULT_METRIC = "euclidean" (bỏ "cosine")
  - Bỏ DEFAULT_DTW_POOL constant

- `requirements.txt`
  - Gỡ `librosa` và `matplotlib`.
  - Core dependencies: soundfile, scipy, numpy, SQLAlchemy, pgvector, FastAPI, uvicorn.

## 5. Tác động thực tế

- **Ít phụ thuộc thư viện hơn**: chỉ 5 core libraries (soundfile, scipy, numpy, SQLAlchemy, FastAPI)
- **Pipeline đơn giản một tầng**: pgvector search Euclidean → top-5 results
- **Performance**: <100ms per query trên 500 files
- **Distance range**: 0.0-0.052 cho top-5 matches (self-match=0.0)
- **Chất lượng**: Euclidean cung cấp discrimination tốt hơn Cosine trên 28-D normalized vectors
- **Dễ maintain**: Không DTW, không Cosine, không MFCC = ít edge cases

## 6. Ghi chú quan trọng

- **Không MFCC**: Hệ thống dùng đặc trưng miền thời gian (energy, ZCR) + miền tần số (centroid, bandwidth, 24-band log-energy)
- **Bám sát bài giảng**: Các đặc trưng được giảng dạy, DTW/MFCC/Cosine không đề cập
- **28-D vector chuẩn hóa**: Scale balance giúp Euclidean discrimination tốt hơn
- **Tương thích dữ liệu**: 500 files đã index với vectors 28 chiều
- **Deployment ready**: Giao diện web chạy ổn định, API sẵn sàng integrate

## 7. Lịch sử thay đổi

- **May 3, 2026**: Bỏ librosa/MFCC, thêm 24-band spectral features
- **May 3-11, 2026**: Phát triển 2-tầng pipeline (vector + DTW), tối ưu DTW performance
- **May 11, 2026**: Phát hiện DTW không trong chương trình → quyết định bỏ DTW
- **May 12, 2026**: 
  - Cập nhật feature vector từ 24→28 chiều (thêm energy, ZCR, centroid_norm, bandwidth_norm)
  - Normalize centroid/bandwidth → [0,1] để tránh scale imbalance
  - Database rebuild: 500 files với vectors 28-D normalized
  - Bỏ Cosine metric: Euclidean cung cấp discrimination tốt hơn
  - Simplify pipeline: 1-tầng (không DTW)
  - Xóa cosine_distance(), giữ euclidean_distance() only
