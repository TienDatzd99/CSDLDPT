"""
Audio feature extraction with minimal dependencies.

This module avoids librosa and uses only numpy, scipy, and soundfile.
"""
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly, stft
from src.config import (
    DURATION,
    SR,
    SPECTRAL_BAND_COUNT,
    STFT_N_PER_SEG,
    STFT_OVERLAP,
    FRAME_LENGTH_SEC,
    FRAME_HOP_SEC,
    FRAME_SILENCE_RATIO,
    FRAME_SILENCE_PERCENTILE,
    FRAME_SILENCE_MULTIPLIER,
)


def _to_mono(y):
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    return np.mean(y, axis=1).astype(np.float32, copy=False)


def _resample_audio(y, original_sr, target_sr):
    if original_sr == target_sr:
        return y
    gcd = np.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    return resample_poly(y, up, down).astype(np.float32, copy=False)


def _trim_silence(y, sr):
    if y.size == 0:
        return y

    frame_length = max(1, int(FRAME_LENGTH_SEC * sr))
    hop_length = max(1, int(FRAME_HOP_SEC * sr))
    frames = _frame_signal(y, frame_length, hop_length)
    if frames.size == 0:
        return y[:0]

    frame_energy = np.mean(frames ** 2, axis=1)
    max_energy = float(np.max(frame_energy))
    if max_energy <= 0.0:
        return y[:0]

    noise_floor = float(np.percentile(frame_energy, FRAME_SILENCE_PERCENTILE))
    threshold = float(noise_floor * FRAME_SILENCE_MULTIPLIER)

    # Cap by a fraction of peak frame energy to avoid over-trimming on high dynamic clips.
    threshold_cap = float(max_energy * FRAME_SILENCE_RATIO)
    if threshold_cap > 0.0:
        threshold = min(threshold, threshold_cap)

    active_frames = np.where(frame_energy > threshold)[0]
    if active_frames.size == 0:
        return y[:0]

    first_frame = int(active_frames[0])
    last_frame = int(active_frames[-1])
    start = first_frame * hop_length
    end = min(y.size, last_frame * hop_length + frame_length)
    return y[start:end]


def _estimate_pitch_autocorrelation(y, sr, fmin=50.0, fmax=300.0):
    if y.size == 0:
        return 0.0

    centered = y - np.mean(y)
    energy = float(np.mean(centered ** 2))
    if energy <= 1e-8:
        return 0.0

    min_lag = max(1, int(sr / fmax))
    max_lag = max(min_lag + 1, int(sr / fmin))
    if centered.size <= max_lag:
        return 0.0

    fft_size = 1 << int(np.ceil(np.log2(centered.size * 2 - 1)))
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size)[: max_lag + 1]

    if autocorr[0] <= 0.0:
        return 0.0

    search = autocorr[min_lag:max_lag + 1]
    if search.size == 0:
        return 0.0

    lag = int(np.argmax(search)) + min_lag
    peak_ratio = float(autocorr[lag] / autocorr[0])
    if peak_ratio < 0.2:
        return 0.0

    return float(sr / lag)


def _spectral_profile(y, sr):
    if y.size == 0:
        return 0.0, 0.0, 0.0, 0.0, np.zeros((0, SPECTRAL_BAND_COUNT), dtype=np.float32)

    nperseg = min(STFT_N_PER_SEG, y.size)
    noverlap = min(STFT_OVERLAP, max(0, nperseg - 1))
    freqs, _, zxx = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(zxx).T.astype(np.float32, copy=False)
    power = magnitude ** 2

    band_edges = np.linspace(0.0, sr / 2.0, SPECTRAL_BAND_COUNT + 1)
    band_matrix = np.zeros((power.shape[0], SPECTRAL_BAND_COUNT), dtype=np.float32)

    for band_idx in range(SPECTRAL_BAND_COUNT):
        left = band_edges[band_idx]
        right = band_edges[band_idx + 1]
        if band_idx == SPECTRAL_BAND_COUNT - 1:
            mask = (freqs >= left) & (freqs <= right)
        else:
            mask = (freqs >= left) & (freqs < right)
        if np.any(mask):
            band_matrix[:, band_idx] = np.log1p(np.sum(power[:, mask], axis=1))

    full_spectrum = np.abs(np.fft.rfft(y)) ** 2
    spectrum_freqs = np.fft.rfftfreq(y.size, d=1.0 / sr)
    total_power = float(np.sum(full_spectrum))

    if total_power > 0.0:
        spectral_centroid = float(np.sum(spectrum_freqs * full_spectrum) / total_power)
        spectral_bandwidth = float(
            np.sqrt(np.sum(((spectrum_freqs - spectral_centroid) ** 2) * full_spectrum) / total_power)
        )
        cumulative = np.cumsum(full_spectrum)
        rolloff_threshold = 0.85 * total_power
        rolloff_idx = int(np.searchsorted(cumulative, rolloff_threshold, side="left"))
        spectral_rolloff = float(spectrum_freqs[min(rolloff_idx, spectrum_freqs.size - 1)])
        flatness = float(
            np.exp(np.mean(np.log(full_spectrum + 1e-12))) / (np.mean(full_spectrum) + 1e-12)
        )
    else:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        spectral_rolloff = 0.0
        flatness = 0.0

    return spectral_centroid, spectral_bandwidth, spectral_rolloff, flatness, band_matrix


def _frame_signal(y, frame_length, hop_length):
    if y.size < frame_length:
        return np.zeros((0, frame_length), dtype=np.float32)

    frame_count = 1 + (y.size - frame_length) // hop_length
    frames = np.empty((frame_count, frame_length), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop_length
        frames[idx] = y[start:start + frame_length]
    return frames


def analyze_audio_frames(y, sr):
    """Phân tích theo frame để lấy energy, ZCR và silent ratio."""
    frame_length = max(1, int(FRAME_LENGTH_SEC * sr))
    hop_length = max(1, int(FRAME_HOP_SEC * sr))
    frames = _frame_signal(y, frame_length, hop_length)

    if frames.size == 0:
        return {
            "frame_count": 0,
            "frame_length_sec": FRAME_LENGTH_SEC,
            "hop_length_sec": FRAME_HOP_SEC,
            "frame_energy_mean": 0.0,
            "frame_energy_std": 0.0,
            "frame_zcr_mean": 0.0,
            "frame_zcr_std": 0.0,
            "frame_silent_ratio": 0.0,
            "silent_energy_threshold": 0.0,
            "noise_floor": 0.0,
        }

    frame_energy = np.mean(frames ** 2, axis=1)
    signs = np.sign(frames)
    signs[signs == 0.0] = 1.0
    frame_zcr = np.count_nonzero(np.diff(signs, axis=1), axis=1) / frame_length

    # Estimate a noise floor using configured percentile (robust to transients)
    noise_floor = float(np.percentile(frame_energy, FRAME_SILENCE_PERCENTILE))
    # Silent threshold = noise_floor * multiplier (user-configurable)
    silent_threshold = float(noise_floor * FRAME_SILENCE_MULTIPLIER)

    # Cap threshold to keep behavior stable on high dynamic range clips.
    threshold_cap = float(np.max(frame_energy) * FRAME_SILENCE_RATIO)
    if threshold_cap > 0.0:
        silent_threshold = min(silent_threshold, threshold_cap)

    silent_ratio = float(np.mean(frame_energy <= silent_threshold))

    return {
        "frame_count": int(frames.shape[0]),
        "frame_length_sec": FRAME_LENGTH_SEC,
        "hop_length_sec": FRAME_HOP_SEC,
        "frame_energy_mean": float(np.mean(frame_energy)),
        "frame_energy_std": float(np.std(frame_energy)),
        "frame_zcr_mean": float(np.mean(frame_zcr)),
        "frame_zcr_std": float(np.std(frame_zcr)),
        "frame_silent_ratio": silent_ratio,
        "silent_energy_threshold": silent_threshold,
        "noise_floor": noise_floor,
    }


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
    audio, sr = sf.read(file_path, always_2d=False)
    y = _to_mono(np.asarray(audio))
    y = _resample_audio(y, sr, SR)
    sr = SR

    y_trimmed = _trim_silence(y, sr)
    silence_ratio = 1 - (len(y_trimmed) / len(y)) if len(y) > 0 else 0.0

    target_length = int(DURATION * SR)
    if len(y_trimmed) == 0:
        y_final = np.zeros(target_length, dtype=np.float32)
    elif len(y_trimmed) > target_length:
        y_final = y_trimmed[:target_length]
    else:
        padding = target_length - len(y_trimmed)
        y_final = np.pad(y_trimmed, (0, padding), "constant").astype(np.float32, copy=False)
    
    return y_final, sr, silence_ratio


def extract_features(y, sr):
    """
    Trích xuất đặc trưng từ âm thanh:
    - Energy (1) - raw mean squared
    - ZCR (1) - zero crossing rate
    - Spectral centroid (1) - normalized to [0, 1]
    - Spectral bandwidth (1) - normalized to [0, 1]
    - 24 band log-energy values (24) - log scale
    = 28-D normalized vector cho Euclidean/Cosine search
    
    Args:
        y: Mảng âm thanh (đã tiền xử lý)
        sr: Sample rate
        
    Returns:
        tuple: (energy, zcr, spectral_centroid, spectral_bandwidth, feature_vector)
            - feature_vector: List 28 chiều [energy, zcr, centroid_norm, bandwidth_norm, + 24-band] cho pgvector
    """
    energy = float(np.mean(y ** 2))

    signs = np.sign(y)
    signs[signs == 0.0] = 1.0
    zcr = float(np.count_nonzero(np.diff(signs)) / max(len(y) - 1, 1))

    spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, spectral_matrix = _spectral_profile(y, sr)

    # Compute 24-band mean log-energy
    if spectral_matrix.size == 0:
        band_means = np.zeros(SPECTRAL_BAND_COUNT, dtype=np.float32)
    else:
        band_means = np.mean(spectral_matrix, axis=0).astype(np.float32, copy=False)

    # Normalize centroid and bandwidth to [0, 1] for better scale balance
    # Centroid: [0, sr/2] -> [0, 1]
    centroid_normalized = spectral_centroid / (sr / 2.0)
    # Bandwidth: typically [0, sr/2] -> [0, 1]
    bandwidth_normalized = spectral_bandwidth / (sr / 2.0)

    # Combine into 28-D: [energy, zcr, centroid_norm, bandwidth_norm, + 24-band]
    feature_vector = np.concatenate([
        [energy, zcr, centroid_normalized, bandwidth_normalized],
        band_means
    ]).astype(np.float32, copy=False)

    return (
        energy,
        zcr,
        spectral_centroid,
        spectral_bandwidth,
        feature_vector.tolist(),
    )


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

