-- Schema SQL cho hệ thống tìm kiếm giọng nói nam giới
-- Database: audio_db
-- Extensions: pgvector

-- Tạo extension pgvector nếu chưa tồn tại
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng lưu trữ metadata và features của file âm thanh
CREATE TABLE IF NOT EXISTS audio_metadata (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) UNIQUE NOT NULL,
    duration FLOAT NOT NULL DEFAULT 5.0,
    silence_ratio FLOAT,
    energy_mean FLOAT,
    zcr_mean FLOAT,
    pitch_mean FLOAT,
    spectral_centroid FLOAT,
    spectral_bandwidth FLOAT,
    
    -- Vector 24D: [energy, zcr, f0, centroid, 20x MFCC]
    -- Dùng cho pgvector search (Stage 1)
    feature_vector vector(24),
    
    -- MFCC matrix 157×20 (JSON format)
    -- Dùng cho DTW re-ranking (Stage 2)
    mfcc_matrix JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index trên file_name để tìm kiếm nhanh
CREATE INDEX IF NOT EXISTS idx_audio_metadata_file_name 
    ON audio_metadata(file_name);

-- Index trên feature_vector cho pgvector search (cosine/euclidean)
CREATE INDEX IF NOT EXISTS idx_audio_metadata_feature_vector 
    ON audio_metadata USING ivfflat (feature_vector vector_cosine_ops)
    WITH (lists = 100);
