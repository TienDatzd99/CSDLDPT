import argparse
import os
from pathlib import Path

from audio_search.audio_processor import DURATION, calculate_dtw_distance, extract_features, preprocess_audio
from audio_search.database import init_db, search_vector_candidates, upsert_audio_metadata


def distance_to_similarity(distance, metric):
    if metric == "cosine":
        # cosine_distance = 1 - cosine_similarity
        return max(0.0, min(1.0, 1.0 - float(distance)))
    # Monotonic transform for distance-based metrics.
    return 1.0 / (1.0 + float(distance))


def index_audio_file(file_path):
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
    folder = Path(folder_path)
    files = sorted(folder.glob(pattern))
    for file_path in files:
        index_audio_file(str(file_path))


def build_search_trace(query_file_path, metric="cosine", top_k=5, dtw_candidate_pool=30):
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

    if metric in {"cosine", "euclidean"}:
        trace["final_results"] = trace["vector_candidates"][:top_k]
        return trace

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
    return trace


def search_similar_audio(query_file_path, metric="cosine", top_k=5, dtw_candidate_pool=30):
    trace = build_search_trace(
        query_file_path=query_file_path,
        metric=metric,
        top_k=top_k,
        dtw_candidate_pool=dtw_candidate_pool,
    )
    return trace["final_results"]


def trace_search_pipeline(query_file_path, top_k=5, dtw_candidate_pool=30):
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



def main():
    parser = argparse.ArgumentParser(description="Audio search engine with Cosine/Euclidean/DTW")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index all wav files in a folder")
    index_parser.add_argument("--folder", required=True, help="Folder path containing wav files")

    search_parser = subparsers.add_parser("search", help="Search similar audio files")
    search_parser.add_argument("--query", required=True, help="Path to query wav file")
    search_parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean", "dtw"],
        default="cosine",
        help="Similarity metric",
    )
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument(
        "--dtw-candidate-pool",
        type=int,
        default=30,
        help="Number of vector candidates before DTW re-ranking",
    )

    trace_parser = subparsers.add_parser("trace", help="Show intermediate search stages for one query")
    trace_parser.add_argument("--query", required=True, help="Path to query wav file")
    trace_parser.add_argument("--top-k", type=int, default=5, help="Number of final results to show")
    trace_parser.add_argument(
        "--dtw-candidate-pool",
        type=int,
        default=30,
        help="Number of vector candidates before DTW re-ranking",
    )

    args = parser.parse_args()

    init_db()

    if args.command == "index":
        index_folder(args.folder)
        print(f"Indexed all wav files in: {args.folder}")
    elif args.command == "search":
        results = search_similar_audio(
            query_file_path=args.query,
            metric=args.metric,
            top_k=args.top_k,
            dtw_candidate_pool=args.dtw_candidate_pool,
        )
        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. {item['file_name']} | similarity={item['similarity']:.6f} | distance={item['distance']:.6f}"
            )
    elif args.command == "trace":
        trace_search_pipeline(
            query_file_path=args.query,
            top_k=args.top_k,
            dtw_candidate_pool=args.dtw_candidate_pool,
        )


if __name__ == "__main__":
    main()
