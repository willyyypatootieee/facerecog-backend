import os
import cv2
import pickle
import numpy as np
import urllib.request
import threading
import logging
from config import settings

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "embeddings.pkl")

for d in [DATA_DIR, FACES_DIR, MODELS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

if not os.path.exists(DETECTOR_PATH):
    print("Downloading YuNet Face Detector...")
    urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)

if not os.path.exists(RECOGNIZER_PATH):
    print("Downloading SFace Face Recognizer...")
    urllib.request.urlretrieve(RECOGNIZER_URL, RECOGNIZER_PATH)

detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320), 0.9, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")
logger = logging.getLogger(__name__)

import json
from database import SessionLocal
import models

def load_embeddings_from_db():
    db = SessionLocal()
    embeddings = {}
    try:
        db_embeddings = db.query(models.FaceEmbedding).all()
        for db_emb in db_embeddings:
            user = db.query(models.User).filter(models.User.id == db_emb.user_id).first()
            if user and db_emb.embedding_data:
                emb_list = json.loads(db_emb.embedding_data)
                embeddings[user.nrp] = np.array(emb_list, dtype=np.float32)
    finally:
        db.close()
    return embeddings

embeddings_db = load_embeddings_from_db()

def get_faces_and_embeddings(img_array):
    """
    Extract multiple face embeddings from an image array.
    Returns a list of tuples: [(face_box, embedding), ...]
    """
    results = []
    try:
        height, width, _ = img_array.shape
        detector.setInputSize((width, height))
        
        _, faces = detector.detect(img_array)
        if faces is not None:
            for face in faces:
                aligned_face = recognizer.alignCrop(img_array, face)
                feature = recognizer.feature(aligned_face)
                results.append((face, feature[0]))
    except Exception:
        logger.exception("Error extracting face embeddings")
    return results

def get_embedding(img_array):
    """
    Extract a single (primary) face embedding.
    """
    faces_data = get_faces_and_embeddings(img_array)
    if len(faces_data) > 0:
        return faces_data[0][1] 
    return None

def register_face(name: str, img_array: np.ndarray) -> bool:
    """
    Registers a face in-memory. DB saving handled by router.
    """
    embedding = get_embedding(img_array)
    if embedding is not None:
        embeddings_db[name] = embedding
        cv2.imwrite(os.path.join(FACES_DIR, f"{name}.jpg"), img_array)
        return True, embedding.tolist()
    return False, None

def remove_registered_face(name: str):
    embeddings_db.pop(name, None)

def _crop_with_margin(img_array: np.ndarray, face_box, margin: float = 0.9) -> np.ndarray:
    height, width = img_array.shape[:2]
    x, y, w, h = [int(v) for v in face_box[:4]]
    pad_x = int(w * margin)
    pad_y = int(h * margin)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return img_array[y1:y2, x1:x2]

def _phone_spoof_signals(img_array: np.ndarray, face_box) -> list:
    signals = []
    crop = _crop_with_margin(img_array, face_box)
    if crop.size == 0:
        return signals

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_area = gray.shape[0] * gray.shape[1]
    if crop_area == 0:
        return signals

    edges = cv2.Canny(gray, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face_area = float(face_box[2] * face_box[3])
    face_center = (
        float(face_box[0]) + float(face_box[2]) / 2,
        float(face_box[1]) + float(face_box[3]) / 2,
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(face_area * 1.4, crop_area * 0.12):
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h else 0
        rectangularity = area / float(w * h) if w and h else 0

        if len(approx) == 4 and 0.45 <= aspect <= 2.4 and rectangularity > 0.55:
            signals.append("screen_like_rectangle")
            break

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    glare_ratio = float(np.mean((value > 235) & (saturation < 45)))
    if glare_ratio > settings.phone_spoof_glare_ratio:
        signals.append("screen_glare")

    edge_ratio = float(np.mean(edges > 0))
    if edge_ratio > settings.phone_spoof_edge_ratio:
        signals.append("screen_edge_pattern")

    full_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    full_edges = cv2.Canny(full_gray, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    full_edges = cv2.morphologyEx(full_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    full_contours, _ = cv2.findContours(full_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = img_array.shape[0] * img_array.shape[1]
    for contour in full_contours:
        area = cv2.contourArea(contour)
        if area < max(face_area * 2.5, frame_area * settings.phone_spoof_screen_ratio):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if not (x <= face_center[0] <= x + w and y <= face_center[1] <= y + h):
            continue

        aspect = w / h if h else 0
        rectangularity = area / float(w * h) if w and h else 0
        if 0.35 <= aspect <= 1.4 and rectangularity > 0.35:
            signals.append("phone_or_screen_around_face")
            break

    x, y, w, h = [int(v) for v in face_box[:4]]
    frame_h, frame_w = img_array.shape[:2]
    pad_x = int(w * 0.45)
    pad_y = int(h * 0.45)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y)
    context = img_array[y1:y2, x1:x2]
    face = img_array[max(0, y):min(frame_h, y + h), max(0, x):min(frame_w, x + w)]
    if context.size and face.size:
        context_gray = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        context_brightness = float(np.mean(context_gray))
        face_brightness = float(np.mean(face_gray))
        dark_context_ratio = float(np.mean(context_gray < 35))
        if dark_context_ratio > 0.28 and face_brightness > context_brightness + 25:
            signals.append("bright_face_dark_phone_border")

    return sorted(set(signals))

def check_liveness(img_array: np.ndarray, face_box) -> dict:
    """
    Lightweight anti-spoofing heuristic.
    This is not a replacement for a trained liveness model, but it blocks common
    phone-screen attempts by detecting rectangular screens, glare, and dense
    display edge patterns around the face.
    """
    signals = _phone_spoof_signals(img_array, face_box)
    penalty = min(0.85, len(signals) * 0.35)
    if len(signals) >= settings.phone_spoof_min_reasons:
        penalty = max(penalty, 0.45)
    score = max(0.0, 0.95 - penalty)
    return {
        "score": score,
        "is_spoof": score < settings.liveness_threshold,
        "reasons": signals,
    }

def recognize_faces(img_array: np.ndarray, threshold: float = 0.35) -> list:
    """
    Recognizes all faces in an image using SFace.
    Returns a list of dicts including liveness score.
    """
    face_features = get_faces_and_embeddings(img_array)
    results = []

    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for face_box, embedding in face_features:
        emb_list = embedding.tolist()
        
        # Check liveness for this face
        liveness_result = check_liveness(img_array, face_box)
        liveness_score = liveness_result["score"]

        if not embeddings_db:
            results.append({
                "user": None,
                "embedding": emb_list,
                "distance": 1.0,
                "box": face_box.tolist(),
                "liveness": liveness_score,
                "liveness_reasons": liveness_result["reasons"],
            })
            continue
        
        best_match = None
        best_dist = float("inf")

        for nrp, db_emb in embeddings_db.items():
            dist = cosine_distance(embedding, db_emb)
            if dist < best_dist:
                best_dist = dist
                best_match = nrp
                
        logger.info(
            "Face recognized - best_match=%s distance=%.4f liveness=%.2f",
            best_match,
            best_dist,
            liveness_score,
        )
        
        if best_dist < threshold:
            results.append({
                "user": best_match,
                "embedding": emb_list,
                "distance": float(best_dist),
                "box": face_box.tolist(),
                "liveness": liveness_score,
                "liveness_reasons": liveness_result["reasons"],
            })
        else:
            results.append({
                "user": None,
                "embedding": emb_list,
                "distance": float(best_dist),
                "box": face_box.tolist(),
                "liveness": liveness_score,
                "liveness_reasons": liveness_result["reasons"],
            })

    return results

class CameraStream:
    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.lock = threading.Lock()
        self.is_running = False
        self.thread = None
        self.latest_frame = None
        self.client_count = 0
        self.count_lock = threading.Lock()

    def add_client(self):
        with self.count_lock:
            self.client_count += 1
            if self.client_count == 1:
                self.start()

    def remove_client(self):
        with self.count_lock:
            self.client_count = max(0, self.client_count - 1)
            if self.client_count == 0:
                self.stop()

    def start(self):
        if self.is_running:
            return True
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.warning("Failed to open camera %s with DSHOW, falling back to default", self.index)
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.cap = None
            self.is_running = False
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                break
        
    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_jpeg_bytes(self):
        frame = self.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return buffer.tobytes()
        return None

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.thread = None
        self.cap = None
        self.latest_frame = None

    def set_index(self, index):
        self.index = index
        if self.is_running:
            self.stop()
            return self.start()
        return True

    def status(self):
        return {
            "index": self.index,
            "is_running": self.is_running,
            "has_frame": self.latest_frame is not None,
            "clients": self.client_count,
        }

camera_stream = CameraStream(index=0)
