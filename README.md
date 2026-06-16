# Face Recognition Attendance Backend

FastAPI backend for student attendance using face recognition. It stores users,
face embeddings, schedules, and attendance records in a local SQLite database.

## Requirements

- Python 3.11 or newer
- `pip`
- Optional: Docker and Docker Compose

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Initialize the database and seed class schedules:

```powershell
python init_db.py
```

Start the API:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open these URLs:

- API root: [http://localhost:8000](http://localhost:8000)
- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI JSON: [http://localhost:8000/api/swagger.json](http://localhost:8000/api/swagger.json)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

## Run With Docker

Build and start the backend:

```powershell
docker compose up --build
```

The API will run at [http://localhost:8000](http://localhost:8000).

Stop the backend:

```powershell
docker compose down
```

The `data/` folder is mounted into the container, so the database, face images,
and downloaded ONNX models persist between runs.

## Configuration

Configuration is read from environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./data/database.db` | SQLAlchemy database URL |
| `APP_TIMEZONE` | `Asia/Jakarta` | Timezone used for schedule and late checks |
| `CORS_ORIGINS` | `*` | Comma-separated allowed frontend origins |
| `FACE_MATCH_THRESHOLD` | `0.35` | Lower means stricter face matching |
| `LIVENESS_THRESHOLD` | `0.70` | Minimum liveness score required |
| `LATE_AFTER_MINUTES` | `15` | Minutes after class start before status becomes `late` |
| `ATTENDANCE_COOLDOWN_MINUTES` | `60` | Prevent duplicate check-ins within this window |
| `MAX_IMAGE_BYTES` | `4194304` | Maximum decoded base64 image size |
| `MIN_FACE_SIZE_PX` | `80` | Minimum detected face width and height |
| `MIN_BLUR_SCORE` | `50` | Minimum Laplacian blur score |
| `MIN_BRIGHTNESS` | `40` | Minimum grayscale brightness |
| `MAX_FACE_TILT_DEGREES` | `20` | Maximum allowed eye-line tilt |
| `PHONE_SPOOF_GLARE_RATIO` | `0.025` | Screen glare threshold for phone-spoof detection |
| `PHONE_SPOOF_EDGE_RATIO` | `0.12` | Dense screen-edge threshold for phone-spoof detection |
| `PHONE_SPOOF_MIN_REASONS` | `1` | Number of phone-spoof signals needed to reject a face |

Example local run with custom CORS:

```powershell
$env:CORS_ORIGINS="http://localhost:3000,http://localhost:5173"
$env:APP_TIMEZONE="Asia/Jakarta"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Example Docker Compose environment section:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - APP_TIMEZONE=Asia/Jakarta
  - CORS_ORIGINS=http://localhost:3000
```

## Main Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | API status message |
| `GET` | `/health` | Health check with database access |
| `POST` | `/register` | Register or update a user face |
| `POST` | `/attendance` | Recognize faces and mark attendance |
| `GET` | `/admin/schedules` | List schedules |
| `POST` | `/admin/schedules/{schedule_id}/force-open` | Manually open or close a schedule |
| `POST` | `/admin/schedules/{schedule_id}/open` | Open one class schedule and close all others |
| `POST` | `/admin/schedules/close` | Close the currently open class schedule |
| `GET` | `/admin/schedules/{schedule_id}/attendances` | List logs for one class schedule |
| `GET` | `/admin/attendances` | List attendance records |
| `GET` | `/users/{nrp}/face` | Get a registered face image |
| `POST` | `/camera/start` | Start camera capture |
| `POST` | `/camera/stop` | Stop camera capture |
| `GET` | `/camera/status` | Check camera lifecycle status |
| `GET` | `/video_feed` | MJPEG camera stream |
| `POST` | `/set_camera/{index}` | Change camera index |

## Register a Face

`/register` expects form data:

- `nrp`
- `nama`
- `jurusan`
- `role`, optional: `student`, `lecturer`, or `admin`
- `image`, a base64-encoded image string

The image may be plain base64 or a browser data URL such as
`data:image/jpeg;base64,...`.

## Mark Attendance

`/attendance` expects form data:

- `image`, a base64-encoded image string

Attendance only works when there is an active schedule. A schedule is active if:

- It was manually opened through `/admin/schedules/{schedule_id}/force-open`, or
- The current day and time match a schedule in the database.

The response returns one result per detected face, including unknown faces,
duplicates, spoof/quality failures, and successful check-ins.

```json
{
  "status": "success",
  "message": "Attendance processed.",
  "schedule_id": 1,
  "recognized": [
    {
      "status": "checked_in",
      "nrp": "5325600084",
      "name": "Student Name",
      "user": {
        "nrp": "5325600084",
        "nama": "Student Name",
        "jurusan": "Multimedia"
      },
      "box": {
        "x": 10,
        "y": 20,
        "width": 120,
        "height": 160
      },
      "confidence": 0.93,
      "liveness": 0.95,
      "attendance_status": "late",
      "message": "Checked in as late.",
      "cooldown_remaining": null,
      "quality": [],
      "liveness_reasons": []
    }
  ]
}
```

Unregistered people are returned per face with:

```json
{
  "status": "not_registered",
  "message": "Not registered."
}
```

Phone-screen attempts are returned as `spoof_suspected` with reasons such as
`screen_like_rectangle`, `phone_or_screen_around_face`, `screen_glare`,
`screen_edge_pattern`, or `bright_face_dark_phone_border`.

The top-level `message` is status-aware. A rejected frame will no longer say
`Attendance processed`; it returns messages like `Phone screen or spoof attempt
suspected`, `Face quality is not good enough for attendance`, or `Not
registered`.

For a real duplicate scan, the existing attendance row timestamp is refreshed so
the admin log reflects the latest valid live capture. Spoof and poor-quality
frames do not refresh logs.

## Attendance Filters

Admin attendance logs can be filtered by query params:

```txt
/admin/attendances?schedule_id=1&date=2026-05-19&status=late&nrp=5325600084
```

You can also fetch logs for one class directly:

```txt
/admin/schedules/1/attendances?date=2026-05-19&status=late
```

## Development Notes

- Face detection uses YuNet and face recognition uses SFace from OpenCV Zoo.
- Model files are stored under `data/models/`.
- Registered face images are stored under `data/faces/`.
- Face embeddings are stored in the database and loaded into memory at startup.
- Current liveness detection is still a placeholder in `face_engine.py`; replace it
  with a real anti-spoofing model before using this for production attendance.

## Common Issues

If no schedule is active, `/attendance` returns:

```json
{
  "status": "fail",
  "detail": "No active class schedule for face recognition right now."
}
```

Open a schedule manually from Swagger docs or make sure your local time matches
one of the seeded schedules.

If the camera stream does not work on Docker, run locally instead. Containers
usually need extra host/device configuration to access a laptop webcam.
