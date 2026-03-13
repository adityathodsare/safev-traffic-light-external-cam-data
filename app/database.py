import sqlite3
import json
from datetime import datetime

conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

# Create tables with traffic light fields
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_number INTEGER UNIQUE,
    image_path TEXT,
    original_image TEXT,
    objects TEXT,
    person_count INTEGER DEFAULT 0,
    car_count INTEGER DEFAULT 0,
    traffic_light_detected BOOLEAN DEFAULT 0,
    traffic_light_color TEXT,
    traffic_light_countdown INTEGER,
    detection_count INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Create index for faster queries
cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_sequence ON detections(sequence_number)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)
""")

conn.commit()

def save_detection(sequence_number, image_path, original_image, objects, 
                   person_count, car_count, traffic_light_info):
    """Save detection results to database"""
    
    objects_json = json.dumps(objects)
    
    cursor.execute("""
        INSERT INTO detections 
        (sequence_number, image_path, original_image, objects, 
         person_count, car_count, traffic_light_detected, 
         traffic_light_color, traffic_light_countdown, detection_count)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        sequence_number, 
        image_path, 
        original_image,
        objects_json,
        person_count,
        car_count,
        traffic_light_info['detected'],
        traffic_light_info['color'],
        traffic_light_info['countdown'],
        len(objects)
    ))
    
    conn.commit()
    return cursor.lastrowid

def get_latest():
    """Get the most recent detection"""
    cursor.execute(
        "SELECT * FROM detections ORDER BY sequence_number DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row:
        return {
            "id": row[0],
            "sequence_number": row[1],
            "image_path": row[2],
            "original_image": row[3],
            "objects": json.loads(row[4]),
            "person_count": row[5],
            "car_count": row[6],
            "traffic_light_detected": bool(row[7]),
            "traffic_light_color": row[8],
            "traffic_light_countdown": row[9],
            "detection_count": row[10],
            "timestamp": row[11]
        }
    return None

def get_history(limit=50):
    """Get detection history"""
    cursor.execute(
        "SELECT * FROM detections ORDER BY sequence_number DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "sequence_number": r[1],
            "image_path": r[2],
            "original_image": r[3],
            "objects": json.loads(r[4]),
            "person_count": r[5],
            "car_count": r[6],
            "traffic_light_detected": bool(r[7]),
            "traffic_light_color": r[8],
            "traffic_light_countdown": r[9],
            "detection_count": r[10],
            "timestamp": r[11]
        })
    return history

def get_detection_by_sequence(seq_num):
    """Get detection by sequence number"""
    cursor.execute(
        "SELECT * FROM detections WHERE sequence_number = ?",
        (seq_num,)
    )
    row = cursor.fetchone()
    if row:
        return {
            "id": row[0],
            "sequence_number": row[1],
            "image_path": row[2],
            "original_image": row[3],
            "objects": json.loads(row[4]),
            "person_count": row[5],
            "car_count": row[6],
            "traffic_light_detected": bool(row[7]),
            "traffic_light_color": row[8],
            "traffic_light_countdown": row[9],
            "detection_count": row[10],
            "timestamp": row[11]
        }
    return None