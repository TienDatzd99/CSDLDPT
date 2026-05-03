"""
Logic tìm kiếm và xếp hạng kết quả (retrieval + ranking).
- Stage 1: Vector search (nhanh)
- Stage 2: DTW re-ranking (chính xác)
"""
import os
from pathlib import Path
from src.feature_extraction import (
    preprocess_audio, extract_features, calculate_dtw_distance
)
from src.database import search_vector_candidates, upsert_audio_metadata, SessionLocal, AudioMetadata
from src.config import DURATION, DEFAULT_METRIC, DEFAULT_TOP_K, DEFAULT_DTW_POOL, AUDIO_DATASET_FOLDER


def distance_to_similarity(distance, metric):
    """
    Chuyển đổi distance thành similarity (0-1, cao = giống).
    
    Args:
        distance: Khoảng cách
        metric: "cosine", "euclidean", hoặc "dtw"
        
    Returns:
        float: Similarity (0-1)
    """
    if metric == "cosine":
        # cosine_distance = 1 - cosine_similarity
        return max(0.0, min(1.0, 1.0 - float(distance)))
    # Monotonic transform cho distance-based metrics
    return 1.0 / (1.0 + float(distance))


def index_audio_file(file_path):
    """
    Index 1 file âm thanh: trích xuất features và lưu vào DB.
    
    Args:
        file_path (str): Đường dẫn tới file WAV
    """
    y, sr, silence_ratio = preprocess_audio(file_path)
    energy, zcr, f0_mean, spectral_centroid, feature_vector, mfcc_matrix = extract_features(y, sr)

    file_name = os.path.basename(file_path)
    upsert_audio_metadata(
        file_name=file_name,
        duration=DURATION,
        silence_ratio=silence_ratio,
        energy_mean=energy,
        zcr_mean=zcr,
        pitch_mean=f0_mean,
        spectral_centroid=spectral_centroid,
        feature_vector=feature_vector,
        mfcc_matrix=mfcc_matrix.tolist(),
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


def build_search_trace(query_file_path, metric=DEFAULT_METRIC, top_k=DEFAULT_TOP_K, dtw_candidate_pool=DEFAULT_DTW_POOL):
    """
    Tìm kiếm và xếp hạng kết quả (2 giai đoạn).
    
    Stage 1: Vector search (pgvector cosine/euclidean)
    Stage 2: DTW re-ranking (nếu metric="dtw")
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        metric (str): "cosine", "euclidean", hoặc "dtw"
        top_k (int): Số kết quả cuối cùng
        dtw_candidate_pool (int): Số candidate cho DTW
        
    Returns:
        dict: Trace object chứa query_summary, vector_candidates, final_results
    """
    # Trích xuất query features
    y, sr, silence_ratio = preprocess_audio(query_file_path)
    energy, zcr, f0_mean, spectral_centroid, query_vector, query_mfcc_matrix = extract_features(y, sr)

    trace = {
        "query_summary": {
            "file_name": os.path.basename(query_file_path),
            "sample_rate": sr,
            "duration_sec": DURATION,
            "samples": len(y),
            "silence_ratio": silence_ratio,
            "energy": energy,
            "zcr": zcr,
            "pitch_mean": f0_mean,
            "spectral_centroid": spectral_centroid,
            "feature_vector_dim": len(query_vector),
            "mfcc_matrix_shape": tuple(query_mfcc_matrix.shape),
        },
        "vector_candidates": [],
        "final_results": [],
    }

    # ===== STAGE 1: Vector Search (nhanh) =====
    vector_metric = metric if metric in {"cosine", "euclidean"} else "cosine"
    candidate_limit = max(top_k, dtw_candidate_pool) if metric == "dtw" else top_k
    
    vector_candidates = search_vector_candidates(query_vector, top_k=candidate_limit, metric=vector_metric)
    
    trace["vector_candidates"] = [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "audio_src": f"/audio/{item['file_name']}",
            "distance": float(item["distance"]),
            "similarity": distance_to_similarity(item["distance"], vector_metric),
        }
        for item in vector_candidates
    ]

    # ===== STAGE 2: DTW Re-ranking (chính xác) =====
    if metric == "dtw":
        dtw_scored = []
        for candidate in vector_candidates:
            if not candidate["mfcc_matrix"]:
                continue
            dtw_distance = calculate_dtw_distance(query_mfcc_matrix, candidate["mfcc_matrix"])
            dtw_scored.append(
                {
                    "id": candidate["id"],
                    "file_name": candidate["file_name"],
                    "audio_src": f"/audio/{candidate['file_name']}",
                    "distance": float(dtw_distance),
                    "similarity": distance_to_similarity(dtw_distance, "dtw"),
                }
            )

        dtw_scored.sort(key=lambda x: x["similarity"], reverse=True)
        trace["final_results"] = dtw_scored[:top_k]
    else:
        # Không DTW: final_results = vector_candidates
        trace["final_results"] = trace["vector_candidates"][:top_k]

    return trace


def search_similar_audio(query_file_path, metric=DEFAULT_METRIC, top_k=DEFAULT_TOP_K, dtw_candidate_pool=DEFAULT_DTW_POOL):
    """
    Tìm kiếm file giống nhất (trả về final results).
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        metric (str): Metric tìm kiếm
        top_k (int): Số kết quả
        dtw_candidate_pool (int): Số candidate cho DTW
        
    Returns:
        list: Top-K kết quả (từ final_results)
    """
    trace = build_search_trace(
        query_file_path=query_file_path,
        metric=metric,
        top_k=top_k,
        dtw_candidate_pool=dtw_candidate_pool,
    )
    return trace["final_results"]


def trace_search_pipeline(query_file_path, top_k=DEFAULT_TOP_K, dtw_candidate_pool=DEFAULT_DTW_POOL):
    """
    Tìm kiếm và in log chi tiết (2 giai đoạn).
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        top_k (int): Số kết quả
        dtw_candidate_pool (int): Số candidate cho DTW
    """
    trace = build_search_trace(
        query_file_path=query_file_path,
        metric="dtw",
        top_k=top_k,
        dtw_candidate_pool=dtw_candidate_pool,
    )

    summary = trace["query_summary"]

    print("=== Query Summary ===")
    print(f"file={summary['file_name']}")
    print(f"sr={summary['sample_rate']}, duration_sec={summary['duration_sec']}, samples={summary['samples']}")
    print(f"silence_ratio={summary['silence_ratio']:.6f}")
    print(
        "features="
        f"energy={summary['energy']:.6f}, zcr={summary['zcr']:.6f}, pitch_mean={summary['pitch_mean']:.6f}, centroid={summary['spectral_centroid']:.6f}"
    )
    print(f"feature_vector_dim={summary['feature_vector_dim']}, mfcc_matrix_shape={summary['mfcc_matrix_shape']}")

    print("\n=== Stage 1: Vector Candidate Search ===")
    for idx, item in enumerate(trace["vector_candidates"][:top_k], start=1):
        print(f"{idx}. {item['file_name']} | distance={item['distance']:.6f} | similarity={item['similarity']:.6f}")

    print("\n=== Stage 2: Final Ranking ===")
    for idx, item in enumerate(trace["final_results"], start=1):
        print(f"{idx}. {item['file_name']} | distance={item['distance']:.6f} | similarity={item['similarity']:.6f}")
