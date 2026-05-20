from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime

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

class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class RecognitionUser(BaseModel):
    nrp: str
    nama: str
    jurusan: str

class AttendanceFaceResult(BaseModel):
    status: str
    nrp: Optional[str] = None
    name: Optional[str] = None
    user: Optional[RecognitionUser] = None
    box: FaceBox
    confidence: float
    liveness: float
    attendance_status: Optional[str] = None
    message: str
    cooldown_remaining: Optional[int] = None
    quality: List[str] = Field(default_factory=list)
    liveness_reasons: List[str] = Field(default_factory=list)

class AttendanceProcessResponse(BaseModel):
    status: str
    message: str
    processed_at: Optional[datetime] = None
    schedule_id: Optional[int] = None
    schedule: Optional[ScheduleResponse] = None
    recognized: List[AttendanceFaceResult] = Field(default_factory=list)

class AttendanceLogUser(BaseModel):
    nrp: Optional[str] = None
    nama: Optional[str] = None
    jurusan: Optional[str] = None

class AttendanceLogSchedule(BaseModel):
    id: Optional[int] = None
    class_name: Optional[str] = None
    lecturer: Optional[str] = None
    room: Optional[str] = None

class AttendanceLogResponse(BaseModel):
    id: int
    user_id: int
    user: AttendanceLogUser
    schedule_id: Optional[int] = None
    schedule: Optional[AttendanceLogSchedule] = None
    timestamp: datetime
    status: str
    confidence: Optional[float] = None

class CameraStatusResponse(BaseModel):
    index: int
    is_running: bool
    has_frame: bool
    clients: int

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

class ScheduleActionResponse(BaseModel):
    status: str
    message: str
    active_schedule: Optional[ScheduleResponse] = None

class ScheduleAttendanceResponse(BaseModel):
    schedule: ScheduleResponse
    attendances: List[AttendanceLogResponse]
