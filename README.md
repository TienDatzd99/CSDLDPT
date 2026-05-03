# Audio Search System - Hệ Thống Tìm Kiếm Âm Thanh Nam Giới

## 📋 Tổng Quan

Hệ thống tìm kiếm giọng nói nam giới sử dụng:
- **Feature Extraction**: 24D vectors (energy, ZCR, pitch, spectral centroid, 20x MFCC)
- **Database**: PostgreSQL + pgvector (vector similarity search)
- **Search Pipeline**: 2-stage (pgvector fast search + DTW re-ranking)
- **Web UI**: FastAPI + HTML/CSS interactive interface

## 🗂️ Cấu Trúc Dự Án

```
CSDLDPT/
├── app.py                      # FastAPI web server (entrypoint)
├── docker-compose.yml          # PostgreSQL + pgvector container
├── requirements.txt            # Python dependencies
│
├── src/                        # Core modules
│   ├── config.py              # Configuration constants
│   ├── database.py            # SQLAlchemy models, pgvector search
│   ├── feature_extraction.py  # Audio preprocessing & feature extraction
│   ├── retrieval.py           # 2-stage search pipeline
│   └── utils.py               # Utility functions (CSV export, etc)
│
├── scripts/                    # CLI tools
│   ├── build_dataset.py       # Index audio folder to database
│   ├── run_query.py           # Command-line search interface
│   ├── evaluate.py            # Evaluation metrics (Hit@K)
│   ├── run_server.sh          # Start FastAPI server
│   ├── index_dataset.sh       # Index dataset (shell wrapper)
│   └── README.md              # Usage guide
│
├── data/                       # Data directory
│   └── dataset/
│       └── male_dataset_500/  # 500 male voice WAV files (16kHz, mono)
│
├── database/                   # Database infrastructure
│   └── schema.sql             # PostgreSQL schema definition
│
└── result/                    # Output directory
    ├── features.csv           # Extracted features (generated)
    └── top5_result.csv        # Search results (generated)
```

## 🚀 Hướng Dẫn Chạy

### 1. Chuẩn bị môi trường

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Khởi động Database

```bash
# Start PostgreSQL + pgvector container
docker-compose up -d

# Verify database is running
docker ps | grep pgvector
```

### 3. Index dữ liệu

```bash
# Option A: Using Python script
python scripts/build_dataset.py --folder data/dataset/male_dataset_500

# Option B: Using shell script
bash scripts/index_dataset.sh

# Option C: Manual indexing
python -c "
from src.retrieval import index_folder
index_folder('data/dataset/male_dataset_500')
"
```

### 4. Chạy Web UI

```bash
bash scripts/run_server.sh
# Or: uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Truy cập: **http://localhost:8000**

## 🔍 Sử Dụng CLI

### Tìm kiếm file giống nhất

```bash
python scripts/run_query.py \
  --query data/dataset/male_dataset_500/male_voice_001.wav \
  --metric dtw \
  --top-k 5
```

### Hiển thị chi tiết 2 giai đoạn

```bash
python scripts/run_query.py \
  --query file.wav \
  --trace \
  --save-csv result/search_results.csv
```

### Đánh giá hệ thống

```bash
python scripts/evaluate.py \
  --dataset-folder data/dataset/male_dataset_500 \
  --sample-size 100 \
  --top-k 5
```

## 🔧 Cấu Hình

Chỉnh sửa `src/config.py`:

```python
DURATION = 5.0              # Chuẩn hóa độ dài (giây)
SR = 16000                  # Sample rate (Hz)
MFCC_N_COEFFS = 20         # Số MFCC coefficients
DEFAULT_METRIC = "cosine"   # "cosine", "euclidean", "dtw"
DEFAULT_TOP_K = 5           # Số kết quả trả về
DEFAULT_DTW_POOL = 30       # Candidate pool cho DTW re-ranking
```

## 📊 Features

### 24D Feature Vector
- Energy (1)
- ZCR - Zero Crossing Rate (1)
- Pitch/F0 - Fundamental Frequency (1)
- Spectral Centroid (1)
- MFCC - Mel-Frequency Cepstral Coefficients (20)

### Search Metrics
- **Cosine**: Fast vector similarity (Stage 1)
- **Euclidean**: L2 distance (Stage 1)
- **DTW**: Dynamic Time Warping on MFCC matrices (Stage 2, accurate but slower)

## 📦 Dependencies

```
fastapi          # Web framework
uvicorn          # ASGI server
librosa          # Audio processing
numpy/scipy      # Numerical computing
psycopg2-binary  # PostgreSQL driver
pgvector         # Vector similarity
sqlalchemy       # ORM
python-multipart # Form data handling
```

## 💾 Database Schema

```sql
CREATE TABLE audio_metadata (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) UNIQUE,
    duration FLOAT,
    silence_ratio FLOAT,
    energy_mean FLOAT,
    zcr_mean FLOAT,
    pitch_mean FLOAT,
    spectral_centroid FLOAT,
    feature_vector vector(24),      -- pgvector for fast search
    mfcc_matrix JSON,               -- For DTW re-ranking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 Kết Quả

Sau khi index 500 file:
- ✅ Tất cả 500 file đã được lưu vào database
- ✅ Tìm kiếm tương tự hoạt động chính xác (Hit@1 = 1.0 cho query file trong dataset)
- ✅ DTW re-ranking cải thiện độ chính xác đáng kể

## 🐛 Troubleshooting

**Database connection error?**
```bash
# Check if Docker daemon is running
docker ps

# Start database again
docker-compose up -d
```

**Import errors?**
```bash
# Reinstall packages
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

**No audio files found?**
```bash
# Verify dataset location
ls -la data/dataset/male_dataset_500/ | head -10

# Count files
find data/dataset/male_dataset_500 -name "*.wav" | wc -l
```

## 📝 Git History

```
8788272 - Refactor: Move dataset to data/dataset/ and update config to use relative paths
8788272 - Refactor: remove legacy audio_search package and move entrypoints to root
6766802 - Clean: Remove unnecessary scaffolding folders and redundant data/ subdirectories
c646305 - Refactor: Reorganize project to academic structure (src/, scripts/, database/, data/, result/)
```

---

**Tác giả**: tiendat  
**Ngôn ngữ**: Vietnamese & English  
**Cập nhật**: May 3, 2026
