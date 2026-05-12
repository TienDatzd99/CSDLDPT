"""
PostgreSQL + pgvector storage for audio features.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AudioMetadata(Base):
    """
    Model lưu trữ metadata và features của file âm thanh.
    
    Columns:
    - id: Khóa chính
    - file_name: Tên file
    - duration: Độ dài (giây)
    - silence_ratio: Tỉ lệ yên lặng
    - energy_mean: Năng lượng
    - zcr_mean: Zero crossing rate
    - pitch_mean: Tần số cơ bản
    - spectral_centroid: Tâm trọng quang phổ
    - spectral_bandwidth: Độ rộng phổ
    - feature_vector: Vector 28D (pgvector) - dùng cho Euclidean/Cosine search
    """
    __tablename__ = "audio_metadata"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    duration = Column(Float)
    silence_ratio = Column(Float)
    energy_mean = Column(Float)
    zcr_mean = Column(Float)
    pitch_mean = Column(Float)
    spectral_centroid = Column(Float)
    spectral_bandwidth = Column(Float)
    
    # Vector 28 chiều: [energy, zcr, spectral_centroid, spectral_bandwidth, + 24-band log-energy]
    # Dùng cho pgvector cosine/euclidean search
    feature_vector = Column(Vector(28))


def init_db():
    """Tạo bảng và pgvector extension nếu chưa tồn tại."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("ALTER TABLE IF EXISTS audio_metadata ADD COLUMN IF NOT EXISTS spectral_bandwidth DOUBLE PRECISION"))
    Base.metadata.create_all(bind=engine)


def upsert_audio_metadata(
    file_name,
    duration,
    silence_ratio,
    energy_mean,
    zcr_mean,
    pitch_mean,
    spectral_centroid,
    spectral_bandwidth,
    feature_vector,
):
    """
    Thêm hoặc cập nhật metadata của file âm thanh.
    
    Args:
        file_name: Tên file
        duration: Độ dài (giây)
        silence_ratio: Tỉ lệ yên lặng
        energy_mean: Năng lượng
        zcr_mean: ZCR
        pitch_mean: Pitch
        spectral_centroid: Spectral centroid
        spectral_bandwidth: Spectral bandwidth
        feature_vector: List 28 số (energy, zcr, centroid, bandwidth, + 24-band)
        
    Returns:
        AudioMetadata: Record đã được lưu
    """
    session = SessionLocal()
    try:
        record = session.query(AudioMetadata).filter(AudioMetadata.file_name == file_name).first()
        if record is None:
            record = AudioMetadata(file_name=file_name)
            session.add(record)

        record.duration = duration
        record.silence_ratio = silence_ratio
        record.energy_mean = energy_mean
        record.zcr_mean = zcr_mean
        record.pitch_mean = pitch_mean
        record.spectral_centroid = spectral_centroid
        record.spectral_bandwidth = spectral_bandwidth
        record.feature_vector = feature_vector

        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


def search_vector_candidates(query_vector, top_k=5, metric="cosine"):
    """
    Tìm kiếm vector candidates bằng pgvector (Euclidean/Cosine).
    
    Args:
        query_vector: Query vector (28D: energy, zcr, centroid, bandwidth, + 24-band)
        top_k: Số kết quả trả về
        metric: "cosine" hoặc "euclidean"
        
    Returns:
        list: Danh sách dict chứa {id, file_name, distance}
    """
    session = SessionLocal()
    try:
        if metric == "cosine":
            distance_expr = AudioMetadata.feature_vector.cosine_distance(query_vector)
        elif metric == "euclidean":
            distance_expr = AudioMetadata.feature_vector.l2_distance(query_vector)
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        rows = (
            session.query(AudioMetadata, distance_expr.label("distance"))
            .order_by(distance_expr)
            .limit(top_k)
            .all()
        )

        results = []
        for record, distance in rows:
            results.append({
                "id": record.id,
                "file_name": record.file_name,
                "distance": float(distance),
            })
        return results
    finally:
        session.close()


def list_indexed_file_names():
    """Return all indexed file names currently stored in the database."""
    session = SessionLocal()
    try:
        rows = session.query(AudioMetadata.file_name).order_by(AudioMetadata.file_name).all()
        return [row[0] for row in rows]
    finally:
        session.close()
