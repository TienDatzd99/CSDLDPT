"""
Configuration và hằng số cho hệ thống tìm kiếm giọng nói nam giới.
"""
import os

# Audio preprocessing settings
DURATION = 5.0  # Chuẩn hóa độ dài (giây)
SR = 16000      # Sample rate (Hz)

# Feature extraction settings
MFCC_N_COEFFS = 20  # Số hệ số MFCC

# Database settings
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/audio_db"
)

# Search settings
DEFAULT_METRIC = "cosine"  # "cosine", "euclidean", "dtw"
DEFAULT_TOP_K = 5
DEFAULT_DTW_POOL = 30

# Dataset folder (relative to project root)
from pathlib import Path
_project_root = Path(__file__).parent.parent
AUDIO_DATASET_FOLDER = os.getenv(
    "AUDIO_DATASET_FOLDER",
    str(_project_root / "data" / "dataset" / "male_dataset_500")
)
