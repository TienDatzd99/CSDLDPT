"""
Configuration and constants for the audio search system.
"""
import os

# Audio preprocessing settings
DURATION = 5.0  # Chuẩn hóa độ dài (giây)
SR = 16000      # Sample rate (Hz)

# Feature extraction settings
SPECTRAL_BAND_COUNT = 24  # Số băng tần phổ dùng cho vector/DTW
STFT_N_PER_SEG = 512
STFT_OVERLAP = 256
FRAME_LENGTH_SEC = 0.025
FRAME_HOP_SEC = 0.010
FRAME_SILENCE_RATIO = 0.10
FRAME_SILENCE_PERCENTILE = 10.0
FRAME_SILENCE_MULTIPLIER = 3.0  # multiplier applied to low-percentile frame energy to detect silence

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
