import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional

from face_engine import register_face, recognize_face, camera_stream

app = FastAPI()

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
async def register(name: str = Form(...), image: str = Form(...)):
    """
    Registers a new user given a base64 encoded image and a name.
    """
    img_array = decode_base64_image(image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    success = register_face(name, img_array)
    if success:
        return {"status": "success", "message": f"User {name} registered successfully."}
    else:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

@app.post("/attendance")
async def attendance(image: str = Form(...)):
    """
    Marks attendance by recognizing the face in the provided base64 encoded image.
    """
    img_array = decode_base64_image(image)
    if img_array is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    result = recognize_face(img_array, threshold=0.35)
    emb = result.get("embedding")

    if emb is None:
        raise HTTPException(status_code=400, detail="No face found.")

    name = result.get("user")
    if name:
        return {"status": "success", "message": f"Welcome, {name}!", "user": name, "embedding": emb}
    else:
        return JSONResponse(status_code=404, content={"status": "fail", "detail": "Face not recognized.", "embedding": emb})

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
