#!/bin/bash
# Index all WAV files from dataset folder into PostgreSQL database
# Usage: bash scripts/index_dataset.sh [dataset_path]

DATASET_PATH="${1:-.venv/lib/python*/site-packages/../../../male_dataset_500}"

cd "$(dirname "$0")/.." || exit
source .venv/bin/activate 2>/dev/null || echo "⚠ Virtual environment not activated"

python3 -c "
from src.retrieval import index_folder
import os

dataset = os.path.expanduser('$DATASET_PATH')
if not os.path.exists(dataset):
    dataset = './male_dataset_500'

print(f'📂 Indexing from: {dataset}')
index_folder(dataset)
print('✅ Indexing complete')
"
