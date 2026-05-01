import librosa
import numpy as np

DURATION = 5.0 # 5 giây
SR = 16000 # 16 kHz

def preprocess_audio(file_path):
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
    # Thời gian
    energy = float(np.mean(y ** 2))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # Tần số
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    f0_mean = float(np.nanmean(f0)) if np.any(voiced_flag) and not np.isnan(np.nanmean(f0)) else 0.0
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_matrix = mfcc.T # dùng cho DTW
    
    # Feature vector 24 chiều: [energy, zcr, f0, centroid, 20x mfcc]
    feature_vector = np.concatenate(([energy, zcr, f0_mean, spectral_centroid], mfcc_mean))
    
    return energy, zcr, f0_mean, spectral_centroid, feature_vector.tolist(), mfcc_matrix


def cosine_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    cosine_sim = float(np.dot(a, b) / denom)
    return 1.0 - cosine_sim


def euclidean_distance(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def calculate_dtw_distance(mfcc_matrix_query, mfcc_matrix_db):
    # librosa.sequence.dtw expects shape (n_features, n_frames)
    query = np.asarray(mfcc_matrix_query, dtype=np.float32).T
    target = np.asarray(mfcc_matrix_db, dtype=np.float32).T
    D, _ = librosa.sequence.dtw(X=query, Y=target, metric='euclidean')
    return float(D[-1, -1])
