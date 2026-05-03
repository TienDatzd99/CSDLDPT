"""
Feature Extraction Module - Trích xuất 24 đặc trưng âm thanh
- 4 đặc trưng cơ bản: energy, zcr, pitch, spectral_centroid
- 20 hệ số MFCC
"""

import librosa
import numpy as np


def extract_features(y, sr):
    """
    Trích xuất 24 đặc trưng từ tín hiệu âm thanh
    
    Đầu vào:
        y: tín hiệu âm thanh (ndarray)
        sr: sample rate (int)
    
    Đầu ra:
        energy: năng lượng trung bình
        zcr: tần số cắt không trung bình
        f0_mean: tần số cơ bản trung bình (Hz)
        spectral_centroid: tâm trọng quang phổ (Hz)
        feature_vector: vector 24 chiều [energy, zcr, f0, centroid, 20x MFCC]
        mfcc_matrix: ma trận MFCC (157, 20) dùng cho DTW
    """
    
    # ===== 4 ĐẶC TRƯNG CỞ BẢN =====
    
    # 1. Energy: Năng lượng trung bình
    energy = float(np.mean(y ** 2))
    
    # 2. ZCR: Zero Crossing Rate - Tần số cắt không
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # 3. F0/Pitch: Tần số cơ bản (sử dụng PYIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    f0_mean = float(np.nanmean(f0)) if np.any(voiced_flag) and not np.isnan(np.nanmean(f0)) else 0.0
    
    # 4. Spectral Centroid: Tâm trọng quang phổ
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    
    # ===== 20 HỆ SỐ MFCC =====
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)  # Trung bình theo thời gian (1D: 20 hệ số)
    mfcc_matrix = mfcc.T  # Giữ nguyên ma trận (157, 20) dùng cho DTW
    
    # ===== VECTOR 24 CHIỀU =====
    # [energy, zcr, f0, centroid, 20x MFCC]
    feature_vector = np.concatenate(([energy, zcr, f0_mean, spectral_centroid], mfcc_mean))
    
    return energy, zcr, f0_mean, spectral_centroid, feature_vector.tolist(), mfcc_matrix


# ===== CÁC HÀM TÍNH KHOẢNG CÁCH =====

def cosine_distance(vec_a, vec_b):
    """
    Tính khoảng cách Cosine giữa 2 vector
    
    Công thức: distance = 1 - cosine_similarity
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
    Tính khoảng cách Euclidean (L2) giữa 2 vector
    
    Công thức: distance = sqrt(sum((a - b)^2))
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def calculate_dtw_distance(mfcc_matrix_query, mfcc_matrix_db):
    """
    Tính khoảng cách Dynamic Time Warping (DTW) giữa 2 ma trận MFCC
    
    DTW cho phép sắp xếp lại frame để tìm alignment tốt nhất
    Dùng cho Stage 2 re-ranking (chính xác cao)
    """
    # Chuyển đổi từ (n_frames, 20) thành (20, n_frames) cho librosa
    query = np.asarray(mfcc_matrix_query, dtype=np.float32).T
    target = np.asarray(mfcc_matrix_db, dtype=np.float32).T
    
    # Tính DTW distance
    D, _ = librosa.sequence.dtw(X=query, Y=target, metric='euclidean')
    
    # Trả về normalized DTW distance (last cell)
    return float(D[-1, -1])
