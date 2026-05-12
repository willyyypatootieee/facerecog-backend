from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserCreate(BaseModel):
    nrp: str
    nama: str
    jurusan: str
    role: Optional[str] = "student"

class UserResponse(BaseModel):
    id: int
    nrp: str
    nama: str
    jurusan: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class AttendanceCreate(BaseModel):
    user_id: int
    status: str
    confidence: float

class AttendanceResponse(BaseModel):
    id: int
    user_id: int
    timestamp: datetime
    status: str
    confidence: float

    class Config:
        from_attributes = True

class ScheduleCreate(BaseModel):
    class_name: str
    lecturer: str
    room: str
    day_of_week: str
    start_time: str
    end_time: str

class ScheduleResponse(ScheduleCreate):
    id: int
    is_active: bool

    class Config:
        from_attributes = True
