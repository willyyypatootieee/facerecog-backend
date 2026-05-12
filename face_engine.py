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

# Download ONNX models if they don't exist
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

# Initialize models
detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320), 0.9, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")

# Load DB
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = {}

def save_db():
    with open(DB_PATH, "wb") as f:
        pickle.dump(embeddings_db, f)

def get_embedding(img_array):
    """
    Extract face embedding from an image array using OpenCV SFace.
    """
    try:
        height, width, _ = img_array.shape
        detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = detector.detect(img_array)
        if faces is not None and len(faces) > 0:
            # Get the first face
            face = faces[0]
            # Align face
            aligned_face = recognizer.alignCrop(img_array, face)
            # Extract feature
            feature = recognizer.feature(aligned_face)
            return feature[0]
    except Exception as e:
        print(f"Error extracting embedding: {e}")
    return None

def register_face(name: str, img_array: np.ndarray) -> bool:
    """
    Registers a face. Returns True if successful.
    """
    embedding = get_embedding(img_array)
    if embedding is not None:
        embeddings_db[name] = embedding
        save_db()
        cv2.imwrite(os.path.join(FACES_DIR, f"{name}.jpg"), img_array)
        return True
    return False

def recognize_face(img_array: np.ndarray, threshold: float = 0.363) -> dict:
    """
    Recognizes a face using SFace. SFace cosine threshold is ~0.363.
    Returns a dict with 'user' (name or None) and 'embedding' (list or None).
    """
    embedding = get_embedding(img_array)
    if embedding is None:
        return {"user": None, "embedding": None}

    emb_list = embedding.tolist()

    if not embeddings_db:
        return {"user": None, "embedding": emb_list}
    
    best_match = None
    best_dist = float("inf")

    def cosine_distance(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for name, db_emb in embeddings_db.items():
        dist = cosine_distance(embedding, db_emb)
        if dist < best_dist:
            best_dist = dist
            best_match = name
            
    print(f"Best match: {best_match} with distance: {best_dist}")
    
    # SFace threshold for cosine distance is typically around 0.363
    # We use 0.35 by default in the API to be slightly more accurate.
    if best_dist < threshold:
        return {"user": best_match, "embedding": emb_list}
    return {"user": None, "embedding": emb_list}

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
