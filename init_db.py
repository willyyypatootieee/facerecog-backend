import os
from database import engine, Base, SessionLocal
import models

# Create all tables
Base.metadata.create_all(bind=engine)

def seed_schedules():
    db = SessionLocal()
    if db.query(models.Schedule).count() == 0:
        schedules = [
            models.Schedule(class_name="Praktikum Basis Data", lecturer="Akhmad Alimudin", room="SAW-09.01", day_of_week="Senin", start_time="11:20", end_time="13:50"),
            models.Schedule(class_name="Kewarganegaraan", lecturer="Lucky Pradigta Setiya Raharja/Moh. Zikky", room="SAW-06.10", day_of_week="Selasa", start_time="08:00", end_time="09:40"),
            models.Schedule(class_name="Workshop Animasi 2D", lecturer="Rendra Suprobo Aji", room="SAW-09.06", day_of_week="Selasa", start_time="09:40", end_time="12:10"),
            models.Schedule(class_name="Fisika Multimedia 2", lecturer="Tri Budi Santoso", room="B 204", day_of_week="Selasa", start_time="13:00", end_time="14:40"),
            models.Schedule(class_name="Workshop Desain Kreatif", lecturer="Rizki Dwi Irianti", room="SAW-09.07", day_of_week="Selasa", start_time="13:50", end_time="16:20"),
            models.Schedule(class_name="Matematika 2", lecturer="Rizki Dwi Irianti", room="PS-04.08", day_of_week="Rabu", start_time="08:50", end_time="10:30"),
            models.Schedule(class_name="Workshop Storytelling Interaktif", lecturer="Ashafidz Fauzan Dianta", room="SAW-09.01", day_of_week="Rabu", start_time="10:30", end_time="13:50"),
            models.Schedule(class_name="Basis Data", lecturer="Akhmad Alimudin", room="PS-04.19", day_of_week="Kamis", start_time="08:00", end_time="09:40"),
            models.Schedule(class_name="Workshop Pemrograman Struktur Data", lecturer="Muhammad Iqbal Izzul Haq", room="SAW-09.06", day_of_week="Kamis", start_time="13:50", end_time="16:20"),
        ]
        db.add_all(schedules)
        db.commit()
    db.close()

if __name__ == "__main__":
    seed_schedules()
    print("Database initialized and schedules seeded.")
