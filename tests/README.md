# Tests Directory

Unit tests và integration tests cho hệ thống tìm kiếm âm thanh.

## Cấu trúc

```
tests/
├── unit/              # Unit tests cho individual modules
│   ├── test_audio_processor.py
│   ├── test_feature_extraction.py
│   ├── test_database.py
│   └── test_search_engine.py
├── integration/       # Integration tests cho end-to-end flows
│   ├── test_indexing_pipeline.py
│   ├── test_search_pipeline.py
│   └── test_api_endpoints.py
└── README.md
```

## Chạy tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy unit tests
pytest tests/unit/

# Chạy integration tests
pytest tests/integration/

# Chạy test file cụ thể
pytest tests/unit/test_audio_processor.py

# Chạy test function cụ thể
pytest tests/unit/test_audio_processor.py::test_preprocess_audio_normalization

# Chạy with coverage
pytest --cov=audio_search tests/
```

## Viết test

### Unit Test Template
```python
import pytest
from audio_search.audio_processor import preprocess_audio

def test_preprocess_audio_normalization():
    """Test that preprocess_audio normalizes duration to 5 seconds"""
    # Arrange
    test_file = "test_samples/short_audio.wav"
    
    # Act
    y, sr, silence_ratio = preprocess_audio(test_file)
    
    # Assert
    assert sr == 16000
    assert len(y) == 5 * 16000  # 5 seconds at 16kHz
    assert 0 <= silence_ratio <= 1
```

### Integration Test Template
```python
import pytest
from audio_search.search_engine import build_search_trace

def test_search_pipeline_end_to_end():
    """Test full indexing and search pipeline"""
    # Setup database, index sample files, search, verify results
```

## Test Fixtures

Common fixtures để tái sử dụng trong tests:
- Sample audio files (short_audio.wav, long_audio.wav, etc.)
- Mock database connection
- Pre-indexed test data
- Temporary directories

## CI/CD Integration

Tests chạy tự động trong GitHub Actions:
- Trigger: mỗi push/pull request
- Chạy: `pytest tests/ --cov=audio_search`
- Report: Coverage badge, test results

## Yêu cầu

- pytest >= 7.0
- pytest-cov (cho coverage reports)
- test-fixtures (sample audio files) trong `tests/fixtures/`
