import cv2
import numpy as np
import base64
import json
import datetime
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List

from database import engine, get_db, Base
import models
import schemas
from face_engine import register_face, recognize_faces, camera_stream, embeddings_db

app = FastAPI(title="Face Recognition Attendance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Face Recognition API is running"}

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

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
    img_array = decode_base64_image(image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    success, emb_list = register_face(nrp, img_array)
    if not success:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    user = db.query(models.User).filter(models.User.nrp == nrp).first()
    if not user:
        user = models.User(nrp=nrp, nama=nama, jurusan=jurusan, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)

    face_emb = models.FaceEmbedding(user_id=user.id, embedding_data=json.dumps(emb_list))
    db.add(face_emb)
    db.commit()

    return {"status": "success", "message": f"User {nama} ({nrp}) registered successfully."}

def get_active_schedule(db: Session) -> Optional[models.Schedule]:
    """ Returns the currently active schedule object. """
    active_sched = db.query(models.Schedule).filter(models.Schedule.is_active == True).first()
    if active_sched:
        return active_sched
    
    now = datetime.datetime.now()
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    current_day = days[now.weekday()]
    curr_time = now.strftime("%H:%M")

    schedules = db.query(models.Schedule).filter(models.Schedule.day_of_week == current_day).all()
    for s in schedules:
        if s.start_time <= curr_time <= s.end_time:
            return s
            
    return None

@app.post("/attendance")
async def attendance(image: str = Form(...), db: Session = Depends(get_db)):
    """
    Marks attendance if active schedule allows, user found, and cooldown passed.
    Handles multiple faces. Parses 'Late' vs 'Present'.
    """
    active_schedule = get_active_schedule(db)
    if not active_schedule:
         return JSONResponse(status_code=403, content={"status": "fail", "detail": "No active class schedule for face recognition right now."})

    img_array = decode_base64_image(image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    recognized_faces = recognize_faces(img_array, threshold=0.35)
    if not recognized_faces:
        raise HTTPException(status_code=400, detail="No faces found.")

    now = datetime.datetime.now()
    try:
        start_time_obj = datetime.datetime.strptime(active_schedule.start_time, "%H:%M").time()
        start_dt = datetime.datetime.combine(now.date(), start_time_obj)
        if (now - start_dt).total_seconds() > 15 * 60:
            attendance_status = "late"
        else:
            attendance_status = "present"
    except:
        attendance_status = "present"

    success_messages = []
    
    for result in recognized_faces:
        nrp = result.get("user")
        distance = result.get("distance", 1.0)
        liveness = result.get("liveness", 1.0)
        confidence = max(0, 1.0 - distance)

        # Liveness Anti-Spoofing check (threshold set to 0.70)
        if liveness < 0.70:
            print(f"Spoof detected for {nrp} - Score: {liveness}")
            continue

        if not nrp:
            continue # Skip unrecognized faces in bulk processing

        user = db.query(models.User).filter(models.User.nrp == nrp).first()
        if not user:
            continue

        # Cooldown check (prevent multiple entries within 1 hour)
        one_hour_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        recent = db.query(models.Attendance).filter(
            models.Attendance.user_id == user.id,
            models.Attendance.schedule_id == active_schedule.id,
            models.Attendance.timestamp >= one_hour_ago
        ).first()

        if recent:
            success_messages.append(f"Already checked in: {user.nama}")
            continue

        # Record attendance with schedule ID
        new_attendance = models.Attendance(
            user_id=user.id, 
            schedule_id=active_schedule.id,
            status=attendance_status, 
            confidence=confidence
        )
        db.add(new_attendance)
        success_messages.append(f"Checked in [{attendance_status}]: {user.nama}")
    
    db.commit()

    if not success_messages:
        return JSONResponse(status_code=404, content={"status": "fail", "detail": "No recognized users could be checked in."})
        
    return {"status": "success", "message": "Attendance processed.", "details": success_messages}

@app.get("/admin/schedules", response_model=List[schemas.ScheduleResponse])
def get_schedules(db: Session = Depends(get_db)):
    return db.query(models.Schedule).all()

@app.post("/admin/schedules/{schedule_id}/force-open")
def force_open_schedule(schedule_id: int, open: bool = True, db: Session = Depends(get_db)):
    # Reset all first
    db.query(models.Schedule).update({"is_active": False})
    
    if open:
        sched = db.query(models.Schedule).filter(models.Schedule.id == schedule_id).first()
        if not sched:
            raise HTTPException(status_code=404, detail="Schedule not found")
        sched.is_active = True
    
    db.commit()
    status = "opened" if open else "closed"
    return {"status": "success", "message": f"Schedule {status} manually."}

@app.get("/admin/attendances")
def get_attendances(db: Session = Depends(get_db)):
    records = db.query(models.Attendance).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "user_id": r.user_id,
            "nrp": getattr(r.user, 'nrp', 'Unknown'),
            "user": getattr(r.user, 'nama', 'Unknown'),
            "schedule_id": r.schedule_id,
            "timestamp": r.timestamp,
            "status": r.status,
            "confidence": r.confidence,
            "class_name": getattr(r.schedule, 'class_name', 'Unknown') if r.schedule else 'Unknown'
        })
    return results

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

@app.get("/api/swagger.json")
def get_swagger_json():
    return app.openapi()

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

@app.post("/set_camera/{index}")
def set_camera(index: int):
    camera_stream.set_index(index)
    return {"status": "success", "message": f"Camera index set to {index}"}

@app.on_event("shutdown")
def shutdown_event():
    camera_stream.stop()