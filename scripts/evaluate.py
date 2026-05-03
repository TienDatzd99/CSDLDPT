import argparse
import random
from pathlib import Path

from src.database import init_db, list_indexed_file_names
from src.retrieval import search_similar_audio


def evaluate_self_retrieval(dataset_folder, sample_size=100, top_k=5, dtw_candidate_pool=30):
    folder = Path(dataset_folder)
    indexed_file_names = set(list_indexed_file_names())

    if not indexed_file_names:
        raise RuntimeError("Database is empty. Please run indexing first.")

    available_queries = []
    for file_name in indexed_file_names:
        candidate_path = folder / file_name
        if candidate_path.exists():
            available_queries.append(candidate_path)

    if not available_queries:
        raise RuntimeError("No query files found in dataset folder for indexed entries.")

    random.seed(42)
    if sample_size > len(available_queries):
        sample_size = len(available_queries)
    queries = random.sample(available_queries, sample_size)

    metrics = ["cosine", "euclidean", "dtw"]

    print(f"Evaluation sample size: {sample_size}")
    print(f"Top-K: {top_k}\n")

    for metric in metrics:
        hit_at_1 = 0
        hit_at_k = 0

        for query_path in queries:
            results = search_similar_audio(
                query_file_path=str(query_path),
                metric=metric,
                top_k=top_k,
                dtw_candidate_pool=dtw_candidate_pool,
            )

            ranked_names = [item["file_name"] for item in results]
            target = query_path.name

            if ranked_names and ranked_names[0] == target:
                hit_at_1 += 1
            if target in ranked_names:
                hit_at_k += 1

        h1 = hit_at_1 / sample_size
        hk = hit_at_k / sample_size
        print(f"Metric: {metric}")
        print(f"  Hit@1: {h1:.4f}")
        print(f"  Hit@{top_k}: {hk:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate male voice retrieval system")
    parser.add_argument("--dataset-folder", required=True, help="Folder path containing dataset wav files")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of random queries")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results for evaluation")
    parser.add_argument(
        "--dtw-candidate-pool",
        type=int,
        default=30,
        help="Number of vector candidates before DTW re-ranking",
    )

    args = parser.parse_args()

    init_db()
    evaluate_self_retrieval(
        dataset_folder=args.dataset_folder,
        sample_size=args.sample_size,
        top_k=args.top_k,
        dtw_candidate_pool=args.dtw_candidate_pool,
    )


if __name__ == "__main__":
    main()
