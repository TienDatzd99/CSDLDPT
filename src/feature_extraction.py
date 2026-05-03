"""
Trích xuất đặc trưng âm thanh (feature extraction).
- Tiền xử lý âm thanh
- Tính toán 24D features (energy, ZCR, pitch, spectral_centroid, 20x MFCC)
- Trích thêm spectral bandwidth để tăng khả năng phân biệt màu giọng
"""
import librosa
import numpy as np
from src.config import DURATION, SR, MFCC_N_COEFFS


def preprocess_audio(file_path):
    """
    Tiền xử lý file âm thanh:
    1. Đọc file với sample rate 16 kHz, mono
    2. Loại bỏ khoảng yên lặng
    3. Chuẩn hóa độ dài thành 5 giây
    
    Args:
        file_path (str): Đường dẫn tới file WAV
        
    Returns:
        tuple: (y, sr, silence_ratio)
            - y: Mảng âm thanh đã xử lý
            - sr: Sample rate (16000 Hz)
            - silence_ratio: Tỉ lệ yên lặng (0-1)
    """
    # 1. Đọc file âm thanh
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    
    # 2. Loại bỏ khoảng lặng
    y_trimmed, index = librosa.effects.trim(y, top_db=20)
    silence_ratio = 1 - (len(y_trimmed) / len(y)) if len(y) > 0 else 0.0
    
    # 3. Chuẩn hóa độ dài 5 giây
    target_length = int(DURATION * SR)
    if len(y_trimmed) > target_length:
        y_final = y_trimmed[:target_length]
    else:
        padding = target_length - len(y_trimmed)
        y_final = np.pad(y_trimmed, (0, padding), 'constant')
    
    return y_final, sr, silence_ratio


def extract_features(y, sr):
    """
    Trích xuất 24 đặc trưng từ âm thanh + 1 đặc trưng bổ sung:
    - Energy (1)
    - ZCR (1)
    - Pitch/F0 (1)
    - Spectral Centroid (1)
    - Spectral Bandwidth (1)
    - MFCC (20)
    
    Args:
        y: Mảng âm thanh (đã tiền xử lý)
        sr: Sample rate
        
    Returns:
        tuple: (energy, zcr, f0_mean, spectral_centroid, spectral_bandwidth, feature_vector, mfcc_matrix)
            - feature_vector: List 24 chiều cho pgvector
            - mfcc_matrix: np.array (n_frames, n_mfcc) cho DTW
    """
    # ===== ENERGY =====
    energy = float(np.mean(y ** 2))
    
    # ===== ZCR (Zero Crossing Rate) =====
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # ===== PITCH (F0) - Fundamental Frequency =====
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    f0_mean = float(np.nanmean(f0)) if np.any(voiced_flag) and not np.isnan(np.nanmean(f0)) else 0.0
    
    # ===== SPECTRAL CENTROID =====
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # ===== SPECTRAL BANDWIDTH =====
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    
    # ===== MFCC (Mel-Frequency Cepstral Coefficients) =====
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_COEFFS)
    mfcc_mean = np.mean(mfcc, axis=1)  # Trung bình theo thời gian: (20,)
    mfcc_matrix = mfcc.T  # Chuyển thành (n_frames, 20) cho DTW
    
    # ===== FEATURE VECTOR 24D =====
    # [energy, zcr, f0, centroid, 20x MFCC]
    feature_vector = np.concatenate(([energy, zcr, f0_mean, spectral_centroid], mfcc_mean))
    
    return energy, zcr, f0_mean, spectral_centroid, spectral_bandwidth, feature_vector.tolist(), mfcc_matrix


def cosine_distance(vec_a, vec_b):
    """
    Tính khoảng cách Cosine giữa 2 vector.
    
    distance = 1 - cosine_similarity
    
    Args:
        vec_a, vec_b: Hai vector (list hoặc np.array)
        
    Returns:
        float: Cosine distance (0-1)
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    cosine_sim = float(np.dot(a, b) / denom)
    return 1.0 - cosine_sim


def euclidean_distance(vec_a, vec_b):
    """
    Tính khoảng cách Euclidean (L2) giữa 2 vector.
    
    distance = sqrt(sum((a - b)^2))
    
    Args:
        vec_a, vec_b: Hai vector
        
    Returns:
        float: Euclidean distance
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def calculate_dtw_distance(mfcc_matrix_query, mfcc_matrix_db):
    """
    Tính Dynamic Time Warping (DTW) distance giữa 2 MFCC matrices.
    
    Dùng để so sánh chi tiết theo thời gian (re-ranking).
    
    Args:
        mfcc_matrix_query: MFCC matrix của query (n_frames, 20)
        mfcc_matrix_db: MFCC matrix từ database (n_frames, 20)
        
    Returns:
        float: DTW distance (thấp = giống, cao = khác)
    """
    query = np.asarray(mfcc_matrix_query, dtype=np.float32).T
    target = np.asarray(mfcc_matrix_db, dtype=np.float32).T
    D, _ = librosa.sequence.dtw(X=query, Y=target, metric='euclidean')
    return float(D[-1, -1])
