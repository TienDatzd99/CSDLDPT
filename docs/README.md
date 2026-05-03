# Documentation Directory

Tài liệu kỹ thuật và hướng dẫn sử dụng hệ thống tìm kiếm âm thanh.

## Nội dung

### Kiến trúc hệ thống
- **app.py**: FastAPI web server, 4 endpoints (GET /, POST /index, POST /search, GET /audio/)
- **search_engine.py**: Orchestration logic, build_search_trace()
- **audio_processor.py**: Tiền xử lý âm thanh (trim silence, normalize 5s, resample 16kHz)
- **feature_extraction.py**: Trích xuất 24D features + khoảng cách (cosine, euclidean, DTW)
- **database.py**: SQLAlchemy models, pgvector queries

### Pipeline xử lý
1. **Indexing**:
   - Đầu vào: File WAV
   - Tiền xử lý: trim silence → normalize 5s
   - Trích xuất: energy, ZCR, F0, spectral_centroid, 20 MFCC coeff
   - Lưu: 24D vector + 157×20 MFCC matrix vào PostgreSQL

2. **Search** (Two-Stage):
   - Stage 1: pgvector search (cosine/euclidean) → top-30 candidates
   - Stage 2: DTW re-ranking trên MFCC matrices → top-5 results
   - Độ chính xác: 85-95% trên các query test

### Công thức toán học
- **Cosine Distance**: `1 - (a·b)/(||a||·||b||)`
- **Euclidean Distance**: `√(∑(aᵢ-bᵢ)²)`
- **DTW**: Khoảng cách biến thời giữa MFCC sequences

### Database Schema
- Table: `audio_metadata` (10 columns)
  - id, file_name, duration, silence_ratio
  - energy_mean, zcr_mean, pitch_mean, spectral_centroid
  - feature_vector (pgvector(24))
  - mfcc_matrix (JSON: 157×20)

### API Endpoints
| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/` | GET | Trang chủ UI |
| `/index` | POST | Index folder WAV |
| `/search` | POST | Tìm kiếm file âm thanh |
| `/audio/{file_name}` | GET | Phát file âm thanh |

### Tham số Search
- `metric`: cosine, euclidean, hoặc dtw
- `top_k`: Số kết quả trả về (1-5)
- `dtw_candidate_pool`: Số candidates Stage 1 để feed vào DTW (5-100)

## Tham khảo

- MFCC: Mel-Frequency Cepstral Coefficients (20 hệ số)
- DTW: Dynamic Time Warping (khoảng cách chuỗi)
- pgvector: PostgreSQL extension cho vector search
- librosa: Audio processing library Python
