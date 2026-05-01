import os

from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/audio_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AudioMetadata(Base):
    __tablename__ = "audio_metadata"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    duration = Column(Float)
    silence_ratio = Column(Float)
    energy_mean = Column(Float)
    zcr_mean = Column(Float)
    pitch_mean = Column(Float)
    spectral_centroid = Column(Float)
    
    # Vector 24 chiều: [energy, zcr, f0, centroid, 20x mfcc]
    feature_vector = Column(Vector(24))
    
    # Lưu dưới dạng mảng JSON để phục vụ tính khoảng cách DTW (re-ranking)
    mfcc_matrix = Column(JSON)


def upsert_audio_metadata(
    file_name,
    duration,
    silence_ratio,
    energy_mean,
    zcr_mean,
    pitch_mean,
    spectral_centroid,
    feature_vector,
    mfcc_matrix,
):
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
        record.feature_vector = feature_vector
        record.mfcc_matrix = mfcc_matrix

        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


def search_vector_candidates(query_vector, top_k=5, metric="cosine"):
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

        return [
            {
                "id": row.id,
                "file_name": row.file_name,
                "distance": float(distance),
                "feature_vector": row.feature_vector,
                "mfcc_matrix": row.mfcc_matrix,
            }
            for row, distance in rows
        ]
    finally:
        session.close()


def list_indexed_file_names():
    session = SessionLocal()
    try:
        rows = session.query(AudioMetadata.file_name).all()
        return [row[0] for row in rows]
    finally:
        session.close()

def init_db():
    # Tạo extension pgvector nếu chưa có
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
