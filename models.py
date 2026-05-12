from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nrp = Column(String, unique=True, index=True)
    nama = Column(String, index=True)
    jurusan = Column(String)
    role = Column(String, default="student") 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    embeddings = relationship("FaceEmbedding", back_populates="user")
    attendances = relationship("Attendance", back_populates="user")

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    embedding_data = Column(String)
    
    user = relationship("User", back_populates="embeddings")

class Attendance(Base):
    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    schedule_id = Column(Integer, ForeignKey("schedules.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String) 
    confidence = Column(Float)
    
    user = relationship("User", back_populates="attendances")
    schedule = relationship("Schedule")

class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String, index=True)
    lecturer = Column(String)
    room = Column(String)
    day_of_week = Column(String) 
    start_time = Column(String) 
    end_time = Column(String) 
    is_active = Column(Boolean, default=False)
