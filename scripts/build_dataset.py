#!/usr/bin/env python3
"""
Script xây dựng dataset: index tất cả file WAV từ thư mục vào database.

Sử dụng:
    python scripts/build_dataset.py --folder /path/to/audio/folder
    
    hoặc mặc định:
    python scripts/build_dataset.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import init_db
from src.retrieval import index_folder
from src.config import AUDIO_DATASET_FOLDER


def main():
    parser = argparse.ArgumentParser(description="Index audio files into database")
    parser.add_argument(
        "--folder",
        type=str,
        default=AUDIO_DATASET_FOLDER,
        help=f"Path to audio folder (default: {AUDIO_DATASET_FOLDER})"
    )
    
    args = parser.parse_args()
    folder_path = args.folder
    
    print(f"🔄 Initializing database...")
    init_db()
    print(f"✓ Database ready")
    
    print(f"📂 Indexing folder: {folder_path}")
    index_folder(folder_path)
    print(f"✓ Indexing complete!")


if __name__ == "__main__":
    main()
