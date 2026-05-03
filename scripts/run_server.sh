#!/usr/bin/env bash
# Script khởi động database (docker-compose) và server uvicorn
set -e
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR/audio_search" || exit 1

# Start postgres container if docker-compose exists
if [ -f docker-compose.yml ]; then
  docker-compose up -d
fi

# Activate venv if present
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
fi

# Start uvicorn
uvicorn audio_search.app:app --reload --host 0.0.0.0 --port 8000
