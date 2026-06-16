import cv2
import numpy as np
import base64
import json
import datetime
import binascii
import logging
from fastapi import FastAPI, HTTPException, Form, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List

from database import engine, get_db, Base
import models
import schemas
from config import settings
from face_engine import (
    register_face,
    remove_registered_face,
    recognize_faces,
    get_faces_and_embeddings,
    camera_stream,
    embeddings_db,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
    openapi_url="/api/swagger.json",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials="*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Face Recognition API is running"}

def now_local() -> datetime.datetime:
    return datetime.datetime.now(settings.timezone)

def now_utc_naive() -> datetime.datetime:
    return datetime.datetime.utcnow()

def utc_naive_to_local(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(settings.timezone)

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        img_data = base64.b64decode(base64_string, validate=True)
        if len(img_data) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="Image is too large.")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        return img
    except HTTPException:
        raise
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image format.")

def normalize_role(role: str) -> str:
    cleaned = role.strip().lower()
    if cleaned not in settings.allowed_roles:
        raise HTTPException(status_code=400, detail="Invalid role.")
    return cleaned

def face_box_to_dict(face_box) -> dict:
    return {
        "x": max(0, int(face_box[0])),
        "y": max(0, int(face_box[1])),
        "width": max(0, int(face_box[2])),
        "height": max(0, int(face_box[3])),
    }

def face_quality_issues(img_array: np.ndarray, face_box=None) -> List[str]:
    issues = []
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))

    if blur_score < settings.min_blur_score:
        issues.append("blurry_frame")
    if brightness < settings.min_brightness:
        issues.append("low_light")

    if face_box is not None:
        box = face_box_to_dict(face_box)
        if box["width"] < settings.min_face_size_px or box["height"] < settings.min_face_size_px:
            issues.append("face_too_small")

        if len(face_box) >= 8:
            right_eye_x, right_eye_y = float(face_box[4]), float(face_box[5])
            left_eye_x, left_eye_y = float(face_box[6]), float(face_box[7])
            eye_dx = left_eye_x - right_eye_x
            if eye_dx:
                tilt = abs(np.degrees(np.arctan2(left_eye_y - right_eye_y, eye_dx)))
                if tilt > settings.max_face_tilt_degrees:
                    issues.append("face_angle_too_extreme")

    return issues

def user_payload(user: models.User) -> dict:
    return {
        "nrp": user.nrp,
        "nama": user.nama,
        "jurusan": user.jurusan,
    }

def schedule_payload(schedule: Optional[models.Schedule]) -> Optional[dict]:
    if not schedule:
        return None
    return {
        "id": schedule.id,
        "class_name": schedule.class_name,
        "lecturer": schedule.lecturer,
        "room": schedule.room,
    }

def schedule_response_payload(schedule: Optional[models.Schedule]) -> Optional[dict]:
    if not schedule:
        return None
    return {
        "id": schedule.id,
        "class_name": schedule.class_name,
        "lecturer": schedule.lecturer,
        "room": schedule.room,
        "day_of_week": schedule.day_of_week,
        "start_time": schedule.start_time,
        "end_time": schedule.end_time,
        "is_active": schedule.is_active,
    }

def attendance_log_payload(record: models.Attendance) -> dict:
    return {
        "id": record.id,
        "user_id": record.user_id,
        "user": user_payload(record.user) if record.user else {"nrp": None, "nama": None, "jurusan": None},
        "schedule_id": record.schedule_id,
        "schedule": schedule_payload(record.schedule),
        "timestamp": utc_naive_to_local(record.timestamp),
        "status": record.status,
        "confidence": record.confidence,
    }

def attendance_response_message(statuses: set, checked_in_count: int) -> str:
    if checked_in_count:
        if statuses - {"checked_in"}:
            return "Attendance partially processed."
        return "Attendance marked."
    if "spoof_suspected" in statuses:
        return "Phone screen or spoof attempt suspected."
    if "poor_quality" in statuses:
        return "Face quality is not good enough for attendance."
    if statuses == {"duplicate"}:
        return "Already checked in. Latest real capture updated."
    if statuses == {"not_registered"}:
        return "Not registered."
    if not statuses:
        return "No faces found."
    return "No attendance was marked."

@app.post("/register")
async def register(
    nrp: str = Form(...),
    nama: str = Form(...),
    jurusan: str = Form(...),
    role: str = Form("student"),
    image: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Registers a new user mapping to nrp, and stores embedding.
    """
    nrp = nrp.strip()
    nama = nama.strip()
    jurusan = jurusan.strip()
    role = normalize_role(role)
    if not nrp or not nama or not jurusan:
        raise HTTPException(status_code=400, detail="nrp, nama, and jurusan are required.")
    
    img_array = decode_base64_image(image)
    faces_data = get_faces_and_embeddings(img_array)
    if not faces_data:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    if len(faces_data) > 1:
        raise HTTPException(status_code=400, detail="Registration image must contain exactly one face.")
    quality_issues = face_quality_issues(img_array, faces_data[0][0])
    if quality_issues:
        raise HTTPException(status_code=400, detail={"message": "Bad face quality.", "issues": quality_issues})

    previous_embedding = embeddings_db.get(nrp)
    success, emb_list = register_face(nrp, img_array)
    if not success:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    try:
        user = db.query(models.User).filter(models.User.nrp == nrp).first()
        if not user:
            user = models.User(nrp=nrp, nama=nama, jurusan=jurusan, role=role)
            db.add(user)
            db.flush()
        else:
            user.nama = nama
            user.jurusan = jurusan
            user.role = role

        face_emb = models.FaceEmbedding(user_id=user.id, embedding_data=json.dumps(emb_list))
        db.add(face_emb)
        db.commit()
    except Exception:
        db.rollback()
        if previous_embedding is not None:
            embeddings_db[nrp] = previous_embedding
        else:
            remove_registered_face(nrp)
        logger.exception("Registration failed for nrp=%s", nrp)
        raise HTTPException(status_code=500, detail="Registration failed.")
    else:
        db.refresh(user)

    return {"status": "success", "message": f"User {nama} ({nrp}) registered successfully."}

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.query(models.User.id).first()
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Database unavailable.")
    return {
        "status": "ok",
        "registered_embeddings": len(embeddings_db),
        "timezone": settings.timezone_name,
    }

def get_active_schedule(db: Session) -> Optional[models.Schedule]:
    """ Returns the currently active schedule object. """
    active_sched = db.query(models.Schedule).filter(models.Schedule.is_active == True).first()
    if active_sched:
        return active_sched
    
    now = now_local()
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    current_day = days[now.weekday()]
    curr_time = now.strftime("%H:%M")

    schedules = db.query(models.Schedule).filter(models.Schedule.day_of_week == current_day).all()
    for s in schedules:
        if s.start_time <= curr_time <= s.end_time:
            return s
            
    return None

@app.post("/attendance", response_model=schemas.AttendanceProcessResponse)
async def attendance(image: str = Form(...), db: Session = Depends(get_db)):
    """
    Marks attendance if active schedule allows, user found, and cooldown passed.
    Handles multiple faces. Parses 'Late' vs 'Present'.
    """
    active_schedule = get_active_schedule(db)
    if not active_schedule:
         return JSONResponse(status_code=403, content={"status": "fail", "message": "No active class schedule for face recognition right now.", "schedule": None, "recognized": []})

    img_array = decode_base64_image(image)
    
    recognized_faces = recognize_faces(img_array, threshold=settings.face_match_threshold)
    if not recognized_faces:
        return JSONResponse(status_code=400, content={
            "status": "fail",
            "message": "No faces found.",
            "schedule_id": active_schedule.id,
            "schedule": schedule_response_payload(active_schedule),
            "recognized": [],
        })

    now = now_local()
    processed_at = now
    current_utc = now_utc_naive()
    try:
        start_time_obj = datetime.datetime.strptime(active_schedule.start_time, "%H:%M").time()
        start_dt = datetime.datetime.combine(now.date(), start_time_obj, tzinfo=settings.timezone)
        if (now - start_dt).total_seconds() > settings.late_after_minutes * 60:
            attendance_status = "late"
        else:
            attendance_status = "present"
    except ValueError:
        attendance_status = "present"

    face_results = []
    checked_in_count = 0
    
    for result in recognized_faces:
        nrp = result.get("user")
        distance = result.get("distance", 1.0)
        liveness = result.get("liveness", 1.0)
        liveness_reasons = result.get("liveness_reasons", [])
        confidence = max(0, 1.0 - distance)
        box = face_box_to_dict(result.get("box", [0, 0, 0, 0]))
        quality_issues = face_quality_issues(img_array, result.get("box"))

        if quality_issues:
            face_results.append({
                "status": "poor_quality",
                "nrp": nrp,
                "name": None,
                "user": None,
                "box": box,
                "confidence": confidence,
                "liveness": liveness,
                "attendance_status": None,
                "message": "Face quality is not good enough for attendance.",
                "quality": quality_issues,
                "liveness_reasons": liveness_reasons,
            })
            continue

        if liveness < settings.liveness_threshold:
            logger.warning("Spoof suspected for nrp=%s liveness=%.2f", nrp, liveness)
            spoof_message = "Phone screen or spoof attempt suspected." if liveness_reasons else "Liveness score is below the required threshold."
            face_results.append({
                "status": "spoof_suspected",
                "nrp": nrp,
                "name": None,
                "user": None,
                "box": box,
                "confidence": confidence,
                "liveness": liveness,
                "attendance_status": None,
                "message": spoof_message,
                "liveness_reasons": liveness_reasons,
            })
            continue

        if not nrp:
            face_results.append({
                "status": "not_registered",
                "nrp": None,
                "name": None,
                "user": None,
                "box": box,
                "confidence": confidence,
                "liveness": liveness,
                "attendance_status": None,
                "message": "Not registered.",
                "liveness_reasons": liveness_reasons,
            })
            continue

        user = db.query(models.User).filter(models.User.nrp == nrp).first()
        if not user:
            face_results.append({
                "status": "not_registered",
                "nrp": nrp,
                "name": None,
                "user": None,
                "box": box,
                "confidence": confidence,
                "liveness": liveness,
                "attendance_status": None,
                "message": "Not registered.",
                "liveness_reasons": liveness_reasons,
            })
            continue

        cooldown_start = current_utc - datetime.timedelta(minutes=settings.attendance_cooldown_minutes)
        recent = db.query(models.Attendance).filter(
            models.Attendance.user_id == user.id,
            models.Attendance.schedule_id == active_schedule.id,
            models.Attendance.timestamp >= cooldown_start
        ).first()

        if recent:
            recent.timestamp = current_utc
            recent.confidence = confidence
            cooldown_until = recent.timestamp + datetime.timedelta(minutes=settings.attendance_cooldown_minutes)
            cooldown_remaining = max(0, int((cooldown_until - current_utc).total_seconds()))
            face_results.append({
                "status": "duplicate",
                "nrp": user.nrp,
                "name": user.nama,
                "user": user_payload(user),
                "box": box,
                "confidence": confidence,
                "liveness": liveness,
                "attendance_status": recent.status,
                "message": "Already checked in. Latest real capture updated.",
                "cooldown_remaining": cooldown_remaining,
                "liveness_reasons": liveness_reasons,
            })
            continue

        new_attendance = models.Attendance(
            user_id=user.id, 
            schedule_id=active_schedule.id,
            status=attendance_status, 
            confidence=confidence
        )
        db.add(new_attendance)
        checked_in_count += 1
        face_results.append({
            "status": "checked_in",
            "nrp": user.nrp,
            "name": user.nama,
            "user": user_payload(user),
            "box": box,
            "confidence": confidence,
            "liveness": liveness,
            "attendance_status": attendance_status,
            "message": f"Checked in as {attendance_status}.",
            "liveness_reasons": liveness_reasons,
        })
    
    db.commit()

    statuses = {item["status"] for item in face_results}
    if checked_in_count:
        response_status = "success" if len(statuses) == 1 else "partial"
    elif statuses == {"duplicate"}:
        response_status = "duplicate"
    else:
        response_status = "fail"
        
    return {
        "status": response_status,
        "message": attendance_response_message(statuses, checked_in_count),
        "processed_at": processed_at,
        "schedule_id": active_schedule.id,
        "schedule": schedule_response_payload(active_schedule),
        "recognized": face_results,
    }

@app.get("/admin/schedules", response_model=List[schemas.ScheduleResponse])
def get_schedules(db: Session = Depends(get_db)):
    return db.query(models.Schedule).all()

@app.post("/admin/schedules/{schedule_id}/force-open", response_model=schemas.ScheduleActionResponse)
def force_open_schedule(schedule_id: int, open: bool = True, db: Session = Depends(get_db)):
    sched = db.query(models.Schedule).filter(models.Schedule.id == schedule_id).first()
    if not sched:
        raise HTTPException(status_code=404, detail="Schedule not found")

    db.query(models.Schedule).update({"is_active": False}, synchronize_session=False)
    
    if open:
        sched.is_active = True
    
    db.commit()
    db.refresh(sched)
    status = "opened" if open else "closed"
    return {
        "status": "success",
        "message": f"Schedule {status} manually.",
        "active_schedule": schedule_response_payload(sched) if open else None,
    }

@app.post("/admin/schedules/{schedule_id}/open", response_model=schemas.ScheduleActionResponse)
def open_schedule(schedule_id: int, db: Session = Depends(get_db)):
    return force_open_schedule(schedule_id=schedule_id, open=True, db=db)

@app.post("/admin/schedules/close", response_model=schemas.ScheduleActionResponse)
def close_active_schedule(db: Session = Depends(get_db)):
    db.query(models.Schedule).update({"is_active": False}, synchronize_session=False)
    db.commit()
    return {
        "status": "success",
        "message": "All schedules closed.",
        "active_schedule": None,
    }

@app.get("/admin/attendances", response_model=List[schemas.AttendanceLogResponse])
def get_attendances(
    schedule_id: Optional[int] = None,
    date: Optional[datetime.date] = None,
    status: Optional[str] = None,
    nrp: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Attendance).join(models.User)
    if schedule_id is not None:
        query = query.filter(models.Attendance.schedule_id == schedule_id)
    if status:
        query = query.filter(models.Attendance.status == status.strip().lower())
    if nrp:
        query = query.filter(models.User.nrp == nrp.strip())
    if date:
        start_local = datetime.datetime.combine(date, datetime.time.min, tzinfo=settings.timezone)
        end_local = start_local + datetime.timedelta(days=1)
        start_utc = start_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        end_utc = end_local.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        query = query.filter(
            models.Attendance.timestamp >= start_utc,
            models.Attendance.timestamp < end_utc,
        )

    records = query.order_by(models.Attendance.timestamp.desc()).all()
    return [attendance_log_payload(record) for record in records]

@app.get("/admin/schedules/{schedule_id}/attendances", response_model=schemas.ScheduleAttendanceResponse)
def get_schedule_attendances(
    schedule_id: int,
    date: Optional[datetime.date] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    schedule = db.query(models.Schedule).filter(models.Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    attendances = get_attendances(
        schedule_id=schedule_id,
        date=date,
        status=status,
        nrp=None,
        db=db,
    )
    return {
        "schedule": schedule_response_payload(schedule),
        "attendances": attendances,
    }

from fastapi.responses import FileResponse
import os
from face_engine import FACES_DIR

@app.get("/users/{nrp}/face")
def get_user_face(nrp: str):
    """
    Returns the registered front face picture of the given NRP.
    """
    face_path = os.path.join(FACES_DIR, f"{nrp}.jpg")
    if os.path.exists(face_path):
        return FileResponse(face_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=404, detail="Face image not found for this user.")

@app.get("/api/swagger.json", include_in_schema=False)
def get_swagger_json(accept: Optional[str] = Header(None)):
    if accept and "text/html" in accept:
        return RedirectResponse(url="/api/docs")
    return JSONResponse(content=app.openapi())

@app.get("/docs", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url="/api/docs")

@app.get("/api/swagger", include_in_schema=False)
def swagger_redirect():
    return RedirectResponse(url="/api/docs")

def generate_frames():
    import time
    camera_stream.add_client()
    try:
        while True:
            frame_bytes = camera_stream.get_jpeg_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
    finally:
        camera_stream.remove_client()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/camera/start", response_model=schemas.CameraStatusResponse)
def start_camera(index: Optional[int] = None):
    if index is not None:
        camera_stream.index = index
    if not camera_stream.start():
        raise HTTPException(status_code=503, detail=f"Could not start camera {camera_stream.index}.")
    return camera_stream.status()

@app.post("/camera/stop", response_model=schemas.CameraStatusResponse)
def stop_camera():
    camera_stream.stop()
    return camera_stream.status()

@app.get("/camera/status", response_model=schemas.CameraStatusResponse)
def camera_status():
    return camera_stream.status()

@app.post("/set_camera/{index}")
def set_camera(index: int):
    if not camera_stream.set_index(index):
        raise HTTPException(status_code=503, detail=f"Could not switch to camera {index}.")
    return {"status": "success", "message": f"Camera index set to {index}"}

@app.on_event("shutdown")
def shutdown_event():
    camera_stream.stop()