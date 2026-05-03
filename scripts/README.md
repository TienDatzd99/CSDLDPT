# Scripts Directory

Các script tiện ích để chạy ứng dụng và xử lý dữ liệu.

## Danh sách script

### `run_server.sh`
Khởi động FastAPI development server
- Cổng: 8000
- URL: http://localhost:8000
- Tự động reload khi file thay đổi

**Sử dụng:**
```bash
bash scripts/run_server.sh
```

### `index_dataset.sh`
Index tất cả file WAV từ dataset folder vào PostgreSQL
- Quét thư mục `male_dataset_500` (500 file)
- Trích xuất 24D feature vectors
- Lưu vào bảng `audio_metadata` với pgvector

**Sử dụng:**
```bash
bash scripts/index_dataset.sh
# Hoặc chỉ định folder khác:
bash scripts/index_dataset.sh /path/to/dataset
```

## Yêu cầu trước

1. Virtual environment activated: `source .venv/bin/activate`
2. PostgreSQL + pgvector chạy: `docker-compose up`
3. Dependencies cài đặt: `pip install -r requirements.txt`

## Quy trình khởi động

```bash
# 1. Khởi động database
docker-compose up -d

# 2. Index dataset (lần đầu tiên hoặc khi có file mới)
bash scripts/index_dataset.sh

# 3. Chạy server
bash scripts/run_server.sh
```

## Lưu ý

- Tất cả script phải chạy từ thư mục gốc project
- Scripts tự động activate virtual environment nếu tìm được
- Kiểm tra logs nếu có lỗi connection database
