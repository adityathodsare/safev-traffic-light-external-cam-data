import sqlite3
import json
from datetime import datetime

conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

# Create tables with enhanced fields
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_number INTEGER UNIQUE,
    image_path TEXT,
    original_image TEXT,
    objects TEXT,
    objects_detailed TEXT,
    person_count INTEGER DEFAULT 0,
    vehicle_count INTEGER DEFAULT 0,
    traffic_light_count INTEGER DEFAULT 0,
    traffic_light_detected BOOLEAN DEFAULT 0,
    traffic_light_color TEXT,
    traffic_light_countdown INTEGER,
    traffic_light_countdown_detected BOOLEAN DEFAULT 0,
    traffic_light_confidence TEXT,
    detection_count INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Create indexes
cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequence ON detections(sequence_number)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_traffic_light ON detections(traffic_light_detected)")

conn.commit()

def save_detection(sequence_number, image_path, original_image, result):
    """Save detection results to database"""
    
    objects_json = json.dumps(result['objects'])
    objects_detailed_json = json.dumps(result['objects_detailed'])
    
    tl_info = result['traffic_light']
    
    # Check if sequence number already exists
    cursor.execute(
        "SELECT id FROM detections WHERE sequence_number = ?",
        (sequence_number,)
    )
    existing = cursor.fetchone()
    
    if existing:
        # Update existing record
        cursor.execute("""
            UPDATE detections 
            SET image_path=?, original_image=?, objects=?, objects_detailed=?,
                person_count=?, vehicle_count=?, traffic_light_count=?,
                traffic_light_detected=?, traffic_light_color=?, traffic_light_countdown=?,
                traffic_light_countdown_detected=?, traffic_light_confidence=?, detection_count=?
            WHERE sequence_number = ?
        """, (
            image_path,
            original_image,
            objects_json,
            objects_detailed_json,
            result['counts']['people'],
            result['counts']['vehicles'],
            result['counts']['traffic_lights'],
            tl_info['detected'],
            tl_info['color'],
            tl_info['countdown'],
            tl_info.get('countdown_detected', False),
            tl_info.get('confidence', 'low'),
            result['counts']['total'],
            sequence_number
        ))
    else:
        # Insert new record
        cursor.execute("""
            INSERT INTO detections 
            (sequence_number, image_path, original_image, objects, objects_detailed,
             person_count, vehicle_count, traffic_light_count,
             traffic_light_detected, traffic_light_color, traffic_light_countdown,
             traffic_light_countdown_detected, traffic_light_confidence, detection_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sequence_number,
            image_path,
            original_image,
            objects_json,
            objects_detailed_json,
            result['counts']['people'],
            result['counts']['vehicles'],
            result['counts']['traffic_lights'],
            tl_info['detected'],
            tl_info['color'],
            tl_info['countdown'],
            tl_info.get('countdown_detected', False),
            tl_info.get('confidence', 'low'),
            result['counts']['total']
        ))
    
    conn.commit()
    return cursor.lastrowid

def get_latest():
    """Get the most recent detection"""
    cursor.execute("SELECT * FROM detections ORDER BY sequence_number DESC LIMIT 1")
    row = cursor.fetchone()
    
    if row:
        return {
            "id": row[0],
            "sequence_number": row[1],
            "image_path": row[2],
            "original_image": row[3],
            "objects": json.loads(row[4]) if row[4] else [],
            "objects_detailed": json.loads(row[5]) if row[5] else [],
            "person_count": row[6] or 0,
            "vehicle_count": row[7] or 0,
            "traffic_light_count": row[8] or 0,
            "traffic_light_detected": bool(row[9]) if row[9] else False,
            "traffic_light_color": row[10] or "unknown",
            "traffic_light_countdown": row[11],
            "traffic_light_countdown_detected": bool(row[12]) if row[12] else False,
            "traffic_light_confidence": row[13] or "low",
            "detection_count": row[14] or 0,
            "timestamp": row[15]
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
            "objects": json.loads(r[4]) if r[4] else [],
            "person_count": r[6] or 0,
            "vehicle_count": r[7] or 0,
            "traffic_light_count": r[8] or 0,
            "traffic_light_detected": bool(r[9]) if r[9] else False,
            "traffic_light_color": r[10] or "unknown",
            "traffic_light_countdown": r[11],
            "traffic_light_countdown_detected": bool(r[12]) if r[12] else False,
            "detection_count": r[14] or 0,
            "timestamp": r[15]
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
            "objects": json.loads(row[4]) if row[4] else [],
            "objects_detailed": json.loads(row[5]) if row[5] else [],
            "person_count": row[6] or 0,
            "vehicle_count": row[7] or 0,
            "traffic_light_count": row[8] or 0,
            "traffic_light_detected": bool(row[9]) if row[9] else False,
            "traffic_light_color": row[10] or "unknown",
            "traffic_light_countdown": row[11],
            "traffic_light_countdown_detected": bool(row[12]) if row[12] else False,
            "traffic_light_confidence": row[13] or "low",
            "detection_count": row[14] or 0,
            "timestamp": row[15]
        }
    return None

def get_statistics():
    """Get detailed statistics"""
    cursor.execute("""
        SELECT 
            COUNT(*) as total_detections,
            SUM(person_count) as total_people,
            SUM(vehicle_count) as total_vehicles,
            SUM(traffic_light_count) as total_traffic_lights,
            SUM(CASE WHEN traffic_light_detected = 1 THEN 1 ELSE 0 END) as detections_with_tl,
            SUM(CASE WHEN traffic_light_color = 'red' THEN 1 ELSE 0 END) as red_lights,
            SUM(CASE WHEN traffic_light_color = 'yellow' THEN 1 ELSE 0 END) as yellow_lights,
            SUM(CASE WHEN traffic_light_color = 'green' THEN 1 ELSE 0 END) as green_lights,
            SUM(CASE WHEN traffic_light_countdown_detected = 1 THEN 1 ELSE 0 END) as countdown_detected,
            AVG(detection_count) as avg_objects
        FROM detections
    """)
    
    stats = cursor.fetchone()
    
    return {
        "total_detections": stats[0] or 0,
        "total_people": stats[1] or 0,
        "total_vehicles": stats[2] or 0,
        "total_traffic_lights": stats[3] or 0,
        "detections_with_traffic_light": stats[4] or 0,
        "red_lights": stats[5] or 0,
        "yellow_lights": stats[6] or 0,
        "green_lights": stats[7] or 0,
        "countdown_detected": stats[8] or 0,
        "avg_objects_per_detection": round(stats[9] or 0, 2)
    }

def reset_database():
    """Reset the database (for testing)"""
    cursor.execute("DROP TABLE IF EXISTS detections")
    cursor.execute("""
    CREATE TABLE detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sequence_number INTEGER UNIQUE,
        image_path TEXT,
        original_image TEXT,
        objects TEXT,
        objects_detailed TEXT,
        person_count INTEGER DEFAULT 0,
        vehicle_count INTEGER DEFAULT 0,
        traffic_light_count INTEGER DEFAULT 0,
        traffic_light_detected BOOLEAN DEFAULT 0,
        traffic_light_color TEXT,
        traffic_light_countdown INTEGER,
        traffic_light_countdown_detected BOOLEAN DEFAULT 0,
        traffic_light_confidence TEXT,
        detection_count INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequence ON detections(sequence_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)")
    conn.commit()
    print("Database reset successfully")