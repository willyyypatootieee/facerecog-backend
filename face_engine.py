import os
import cv2
import pickle
import numpy as np
import urllib.request
import threading

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

import json
from database import SessionLocal
import models

def load_embeddings_from_db():
    db = SessionLocal()
    embeddings = {}
    db_embeddings = db.query(models.FaceEmbedding).all()
    for db_emb in db_embeddings:
        user = db.query(models.User).filter(models.User.id == db_emb.user_id).first()
        if user and db_emb.embedding_data:
            emb_list = json.loads(db_emb.embedding_data)
            embeddings[user.nrp] = np.array(emb_list, dtype=np.float32)
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
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
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

def check_liveness(img_array: np.ndarray, face_box) -> float:
    """
    Placeholder for Liveness Detection (Anti-Spoofing).
    In production, you load an ONNX model (e.g., MiniFASNet) here, crop the face using face_box,
    and run inference to get a real/spoof score.
    
    Returns a liveness confidence score (0.0 to 1.0).
    """
    # 1. Crop face from img_array using face_box coordinates
    # x, y, w, h = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])
    # face_crop = img_array[y:y+h, x:x+w]
    
    # 2. Run your Anti-Spoofing ONNX model inference here
    # liveness_score = anti_spoof_model.predict(face_crop)
    
    # Returning a mock score of 0.95 (Real) for now.
    # Change to random.uniform(0.0, 1.0) to test spoof rejections.
    return 0.95

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
        liveness_score = check_liveness(img_array, face_box)

        if not embeddings_db:
            results.append({"user": None, "embedding": emb_list, "distance": 1.0, "box": face_box.tolist(), "liveness": liveness_score})
            continue
        
        best_match = None
        best_dist = float("inf")

        for nrp, db_emb in embeddings_db.items():
            dist = cosine_distance(embedding, db_emb)
            if dist < best_dist:
                best_dist = dist
                best_match = nrp
                
        print(f"Face recognized - Best match: {best_match} with distance: {best_dist}, Liveness: {liveness_score}")
        
        if best_dist < threshold:
            results.append({"user": best_match, "embedding": emb_list, "distance": float(best_dist), "box": face_box.tolist(), "liveness": liveness_score})
        else:
            results.append({"user": None, "embedding": emb_list, "distance": float(best_dist), "box": face_box.tolist(), "liveness": liveness_score})

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
            return
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.index} with DSHOW, falling back to default")
            self.cap = cv2.VideoCapture(self.index)
        
        self.is_running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

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
            self.thread.join()
        if self.cap:
            self.cap.release()

    def set_index(self, index):
        self.index = index
        if self.is_running:
            self.stop()
            self.start()

camera_stream = CameraStream(index=0)
