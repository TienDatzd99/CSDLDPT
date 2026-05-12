"""
Search and ranking logic using Euclidean/Cosine distance (vector search).
"""
import os
from pathlib import Path
from src.feature_extraction import (
    preprocess_audio, extract_features, analyze_audio_frames
)
from src.database import search_vector_candidates, upsert_audio_metadata, SessionLocal, AudioMetadata
from src.config import DURATION, DEFAULT_METRIC, DEFAULT_TOP_K, AUDIO_DATASET_FOLDER


def distance_to_similarity(distance, metric):
    """
    Chuyển đổi distance thành similarity (0-1, cao = giống).
    
    Args:
        distance: Khoảng cách
        metric: "cosine" hoặc "euclidean"
        
    Returns:
        float: Similarity (0-1)
    """
    if metric == "cosine":
        # cosine_distance = 1 - cosine_similarity
        return max(0.0, min(1.0, 1.0 - float(distance)))
    # Euclidean: normalize
    return 1.0 / (1.0 + float(distance))


def index_audio_file(file_path):
    """
    Index 1 file âm thanh: trích xuất features và lưu vào DB.
    
    Args:
        file_path (str): Đường dẫn tới file WAV
    """
    y, sr, silence_ratio = preprocess_audio(file_path)
    energy, zcr, spectral_centroid, spectral_bandwidth, feature_vector = extract_features(y, sr)

    file_name = os.path.basename(file_path)
    upsert_audio_metadata(
        file_name=file_name,
        duration=DURATION,
        silence_ratio=silence_ratio,
        energy_mean=energy,
        zcr_mean=zcr,
        pitch_mean=0.0,  # Deprecated: not used
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        feature_vector=feature_vector,
    )


def index_folder(folder_path, pattern="*.wav"):
    """
    Index tất cả file WAV trong thư mục.
    
    Args:
        folder_path (str): Đường dẫn thư mục
        pattern (str): Pattern tìm file (mặc định "*.wav")
    """
    folder = Path(folder_path)
    files = sorted(folder.glob(pattern))
    for file_path in files:
        try:
            index_audio_file(str(file_path))
            print(f"✓ Indexed: {file_path.name}")
        except Exception as e:
            print(f"✗ Error indexing {file_path.name}: {e}")


def build_search_trace(query_file_path, metric=DEFAULT_METRIC, top_k=DEFAULT_TOP_K):
    """
    Tìm kiếm sử dụng Euclidean/Cosine distance.
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        metric (str): "cosine" hoặc "euclidean"
        top_k (int): Số kết quả
        
    Returns:
        dict: Trace object chứa query_summary và final_results
    """
    # Trích xuất query features
    y, sr, silence_ratio = preprocess_audio(query_file_path)
    energy, zcr, spectral_centroid, spectral_bandwidth, query_vector = extract_features(y, sr)
    frame_stats = analyze_audio_frames(y, sr)

    trace = {
        "query_summary": {
            "file_name": os.path.basename(query_file_path),
            "sample_rate": sr,
            "duration_sec": DURATION,
            "samples": len(y),
            "silence_ratio": silence_ratio,
            "energy": energy,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "feature_vector_dim": len(query_vector),
            **frame_stats,
        },
        "final_results": [],
    }

    # Vector search (Euclidean/Cosine)
    search_metric = metric if metric in {"cosine", "euclidean"} else "cosine"
    results = search_vector_candidates(query_vector, top_k=top_k, metric=search_metric)
    
    trace["final_results"] = [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "audio_src": f"/audio/{item['file_name']}",
            "distance": float(item["distance"]),
            "similarity": distance_to_similarity(item["distance"], search_metric),
        }
        for item in results
    ]

    return trace


def search_similar_audio(query_file_path, metric=DEFAULT_METRIC, top_k=DEFAULT_TOP_K):
    """
    Tìm kiếm file giống nhất.
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        metric (str): "cosine" hoặc "euclidean"
        top_k (int): Số kết quả
        
    Returns:
        list: Top-K kết quả
    """
    trace = build_search_trace(
        query_file_path=query_file_path,
        metric=metric,
        top_k=top_k,
    )
    return trace["final_results"]


def trace_search_pipeline(query_file_path, top_k=DEFAULT_TOP_K, metric=DEFAULT_METRIC):
    """
    Tìm kiếm và in log chi tiết.
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        top_k (int): Số kết quả
        metric (str): "cosine" hoặc "euclidean"
    """
    trace = build_search_trace(
        query_file_path=query_file_path,
        metric=metric,
        top_k=top_k,
    )

    summary = trace["query_summary"]

    print("=== Query Summary ===")
    print(f"file={summary['file_name']}")
    print(f"sr={summary['sample_rate']}, duration_sec={summary['duration_sec']}, samples={summary['samples']}")
    print(f"silence_ratio={summary['silence_ratio']:.6f}")
    print(
        "features="
        f"energy={summary['energy']:.6f}, zcr={summary['zcr']:.6f}, centroid={summary['spectral_centroid']:.6f}, bandwidth={summary['spectral_bandwidth']:.6f}"
    )
    print(f"feature_vector_dim={summary['feature_vector_dim']}")
    print(
        f"frame_stats=frames:{summary['frame_count']}, "
        f"frame_energy_mean:{summary['frame_energy_mean']:.6f}, "
        f"frame_zcr_mean:{summary['frame_zcr_mean']:.6f}, "
        f"frame_silent_ratio:{summary['frame_silent_ratio']:.6f}"
    )

    print(f"\n=== Search Results (Top {top_k}) ===")
    for idx, item in enumerate(trace["final_results"], start=1):
        print(f"{idx}. {item['file_name']} | distance={item['distance']:.6f} | similarity={item['similarity']:.6f}")
