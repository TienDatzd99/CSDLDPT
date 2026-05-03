#!/usr/bin/env python3
"""
Script tìm kiếm: tìm top-K file giống nhất với query file.

Sử dụng:
    python scripts/run_query.py --query /path/to/query.wav --metric dtw --top-k 5
    
    hoặc:
    python scripts/run_query.py --query query.wav  # mặc định cosine, top-k=5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import trace_search_pipeline, search_similar_audio
from src.utils import print_search_results, save_search_results_to_csv
from src.config import DEFAULT_METRIC, DEFAULT_TOP_K, DEFAULT_DTW_POOL


def main():
    parser = argparse.ArgumentParser(description="Search for similar audio files")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query audio file"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "euclidean", "dtw"],
        default=DEFAULT_METRIC,
        help=f"Search metric (default: {DEFAULT_METRIC})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--dtw-pool",
        type=int,
        default=DEFAULT_DTW_POOL,
        help=f"DTW candidate pool size (default: {DEFAULT_DTW_POOL})"
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Save results to CSV file (optional)"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print detailed trace (2 stages)"
    )
    
    args = parser.parse_args()
    
    query_path = args.query
    if not Path(query_path).exists():
        print(f"❌ Query file not found: {query_path}")
        sys.exit(1)
    
    print(f"🔍 Searching for similar audio files...")
    print(f"  Query: {query_path}")
    print(f"  Metric: {args.metric}")
    print(f"  Top-K: {args.top_k}")
    
    if args.trace:
        print(f"\n{'='*60}")
        print(f"DETAILED TRACE (2-stage pipeline)")
        print(f"{'='*60}")
        trace_search_pipeline(
            query_path,
            top_k=args.top_k,
            dtw_candidate_pool=args.dtw_pool
        )
    else:
        print_search_results(
            query_path,
            metric=args.metric,
            top_k=args.top_k
        )
    
    if args.save_csv:
        save_search_results_to_csv(
            query_path,
            output_path=args.save_csv,
            metric=args.metric,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
