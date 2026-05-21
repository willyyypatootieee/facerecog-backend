import os
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class Settings:
    app_name = "Face Recognition Attendance API"
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/database.db")
    timezone_name = os.getenv("APP_TIMEZONE", "Asia/Jakarta")
    face_match_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.35"))
    liveness_threshold = float(os.getenv("LIVENESS_THRESHOLD", "0.70"))
    late_after_minutes = int(os.getenv("LATE_AFTER_MINUTES", "15"))
    attendance_cooldown_minutes = int(os.getenv("ATTENDANCE_COOLDOWN_MINUTES", "60"))
    max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
    min_face_size_px = int(os.getenv("MIN_FACE_SIZE_PX", "80"))
    min_blur_score = float(os.getenv("MIN_BLUR_SCORE", "50"))
    min_brightness = float(os.getenv("MIN_BRIGHTNESS", "40"))
    max_face_tilt_degrees = float(os.getenv("MAX_FACE_TILT_DEGREES", "20"))
    phone_spoof_screen_ratio = float(os.getenv("PHONE_SPOOF_SCREEN_RATIO", "0.18"))
    phone_spoof_glare_ratio = float(os.getenv("PHONE_SPOOF_GLARE_RATIO", "0.025"))
    phone_spoof_edge_ratio = float(os.getenv("PHONE_SPOOF_EDGE_RATIO", "0.12"))
    phone_spoof_min_reasons = int(os.getenv("PHONE_SPOOF_MIN_REASONS", "1"))
    allowed_roles = {"student", "lecturer", "admin"}

    @property
    def timezone(self):
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            if self.timezone_name == "Asia/Jakarta":
                return timezone(timedelta(hours=7), name="Asia/Jakarta")
            return timezone.utc


settings = Settings()
