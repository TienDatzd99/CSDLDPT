"""
Audio Preprocessing Module - Tiền xử lý âm thanh
- Đọc file WAV
- Loại bỏ khoảng yên lặng
- Chuẩn hóa độ dài 5 giây
- Resample về 16 kHz, mono
"""

import librosa
import numpy as np

DURATION = 5.0  # 5 giây
SR = 16000      # 16 kHz


def preprocess_audio(file_path):
    """
    Tiền xử lý file âm thanh
    
    Các bước:
        1. Đọc file WAV (16 kHz, mono)
        2. Loại bỏ khoảng yên lặng ở đầu/cuối
        3. Chuẩn hóa độ dài = 5 giây (pad hoặc cắt)
    
    Đầu vào:
        file_path: đường dẫn tới file WAV
    
    Đầu ra:
        y: tín hiệu âm thanh (ndarray)
        sr: sample rate (int, = 16000)
        silence_ratio: tỉ lệ yên lặng bị loại bỏ (float)
    """
    
    # 1. Đọc file âm thanh
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    
    # 2. Loại bỏ khoảng yên lặng
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
