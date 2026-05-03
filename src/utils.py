"""
Hàm tiện ích chung (utilities) cho hệ thống.
"""
import csv
from pathlib import Path
from src.retrieval import build_search_trace
from src.config import DEFAULT_METRIC, DEFAULT_TOP_K, DEFAULT_DTW_POOL


def save_features_to_csv(output_path="result/features.csv"):
    """
    Xuất tất cả features từ database thành CSV.
    
    Args:
        output_path (str): Đường dẫn file CSV output
    """
    from src.database import SessionLocal, AudioMetadata
    
    session = SessionLocal()
    try:
        records = session.query(AudioMetadata).all()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'file_name', 'duration', 'silence_ratio', 'energy_mean', 'zcr_mean',
                'pitch_mean', 'spectral_centroid', 'spectral_bandwidth', 'feature_vector'
            ])
            
            for record in records:
                writer.writerow([
                    record.file_name,
                    record.duration,
                    f"{record.silence_ratio:.6f}",
                    f"{record.energy_mean:.6f}",
                    f"{record.zcr_mean:.6f}",
                    f"{record.pitch_mean:.6f}",
                    f"{record.spectral_centroid:.6f}",
                    f"{(record.spectral_bandwidth or 0.0):.6f}",
                    ",".join([f"{x:.6f}" for x in record.feature_vector])
                ])
        
        print(f"✓ Features saved to {output_path}")
    finally:
        session.close()


def save_search_results_to_csv(query_file_path, output_path="result/top5_result.csv",
                               metric=DEFAULT_METRIC, top_k=DEFAULT_TOP_K):
    """
    Tìm kiếm và xuất kết quả thành CSV.
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        output_path (str): Đường dẫn file CSV output
        metric (str): Metric tìm kiếm
        top_k (int): Số kết quả
    """
    trace = build_search_trace(query_file_path, metric=metric, top_k=top_k)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'file_name', 'similarity', 'distance', 'metric'])
        
        for rank, result in enumerate(trace["final_results"], start=1):
            writer.writerow([
                rank,
                result['file_name'],
                f"{result['similarity']:.6f}",
                f"{result['distance']:.6f}",
                metric
            ])
    
    print(f"✓ Search results saved to {output_path}")


def print_search_results(query_file_path, metric=DEFAULT_METRIC, top_k=DEFAULT_TOP_K):
    """
    In kết quả tìm kiếm dạng bảng.
    
    Args:
        query_file_path (str): Đường dẫn file truy vấn
        metric (str): Metric tìm kiếm
        top_k (int): Số kết quả
    """
    trace = build_search_trace(query_file_path, metric=metric, top_k=top_k)
    
    print(f"\n{'='*60}")
    print(f"Query: {trace['query_summary']['file_name']}")
    print(f"Metric: {metric}, Top-K: {top_k}")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'File Name':<25} {'Similarity':<12} {'Distance':<12}")
    print(f"{'-'*60}")
    
    for rank, result in enumerate(trace["final_results"], start=1):
        print(f"{rank:<5} {result['file_name']:<25} {result['similarity']:<12.6f} {result['distance']:<12.6f}")
    
    print(f"{'='*60}\n")
