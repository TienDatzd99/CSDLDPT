Cấu trúc thư mục đề xuất

- data/
  - raw/           # chứa WAV gốc (bỏ qua git)
  - processed/     # cache các file đã tiền xử lý

- audio_search/    # mã nguồn chính (FastAPI app)
  - app.py
  - search_engine.py
  - audio_processor.py
  - database.py
  - ...

- scripts/
  - index_dataset.py   # script index dataset vào DB
  - run_server.sh      # script khởi động server và DB

- notebooks/         # Jupyter notebooks demo
- tests/             # test unit/simple
- docs/              # tài liệu

Ghi chú:
- Thêm `data/raw` vào .gitignore để tránh đẩy dữ liệu lớn.
- Dùng `scripts/index_dataset.py` để nạp dữ liệu vào DB.
- Dùng `scripts/run_server.sh` để khởi động environment phát triển.
