# Notebooks Directory

Jupyter notebooks cho phân tích dữ liệu, thử nghiệm tính năng, và demo.

## Danh sách Notebook

### `01_feature_extraction_demo.ipynb`
Demo trích xuất 24D features từ file WAV
- Load audio file
- Visualize waveform và spectrogram
- Tính energy, ZCR, F0 fundamental frequency
- Vẽ MFCC heatmap (20 hệ số)
- So sánh features của 2 file

### `02_similarity_metrics_comparison.ipynb`
So sánh 3 metric khoảng cách
- Load query và 10 database files
- Tính cosine, euclidean, DTW distance
- Vẽ similarity matrix
- Ranking so sánh

### `03_search_pipeline_analysis.ipynb`
Phân tích two-stage search pipeline
- Query file → Stage 1: pgvector search
- Visualize top-30 candidates
- Stage 2: DTW re-ranking
- So sánh kết quả: pgvector vs DTW

### `04_dataset_statistics.ipynb`
Thống kê tập dataset 500 file
- Duration distribution
- Silence ratio histogram
- Feature statistics (energy, ZCR, F0, etc.)
- MFCC patterns

## Chạy Notebook

```bash
# Khởi động Jupyter
jupyter notebook

# Hoặc Jupyter Lab (nếu cài)
jupyter lab
```

## Template Notebook

### Cell 1: Import và Setup
```python
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from audio_search.audio_processor import preprocess_audio
from audio_search.feature_extraction import extract_features
from audio_search.database import search_vector_candidates
```

### Cell 2: Load Audio
```python
file_path = '../male_dataset_500/sample_001.wav'
y, sr, silence_ratio = preprocess_audio(file_path)
print(f"Duration: {len(y)/sr:.2f}s, SR: {sr}, Silence: {silence_ratio:.2%}")
```

### Cell 3: Extract & Visualize
```python
energy, zcr, f0, centroid, feature_vec, mfcc_mat = extract_features(y, sr)
print(f"Feature vector (24D): {feature_vec}")
plt.imshow(mfcc_mat.T, aspect='auto', origin='lower')
plt.colorbar()
plt.title("MFCC Matrix")
plt.show()
```

## Yêu cầu

- jupyter >= 7.0
- matplotlib >= 3.5
- librosa >= 0.10
- numpy >= 1.24
- scipy >= 1.10

## Lưu ý

- Không commit `.ipynb_checkpoints/` folder (thêm vào .gitignore)
- Xóa cell outputs trước commit để giảm file size
- Sử dụng markdown cells để document code

## Best Practices

1. Mỗi notebook = 1 topic/feature
2. Thêm title markdown cell ở đầu
3. Chia thành sections nhỏ (10-15 cells)
4. Thêm description cho complex visualization
5. Sử dụng `%matplotlib inline` cho interactive plots
