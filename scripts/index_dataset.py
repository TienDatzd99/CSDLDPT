#!/usr/bin/env python3
"""Script đơn giản để index một thư mục âm thanh vào DB.
Sử dụng: python scripts/index_dataset.py --folder /path/to/male_dataset_500
"""
import argparse
from src.retrieval import index_folder


def main():
    parser = argparse.ArgumentParser(description="Index WAV folder into DB")
    parser.add_argument("--folder", "-f", required=False, default="../male_dataset_500", help="Folder chứa file .wav")
    args = parser.parse_args()
    print(f"Indexing folder: {args.folder}")
    index_folder(args.folder)
    print("Indexing finished")


if __name__ == '__main__':
    main()
