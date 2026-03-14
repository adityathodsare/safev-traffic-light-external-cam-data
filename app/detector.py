from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np
import logging
from collections import Counter
from app.traffic_light_detector import traffic_light_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO("yolov8n.pt")

DETECTION_FOLDER = "detections"
UPLOAD_FOLDER    = "uploads"
os.makedirs(DETECTION_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER,    exist_ok=True)

TARGET_CLASSES = [
    'person', 'car', 'truck', 'bus',
    'motorcycle', 'bicycle', 'traffic light'
]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_next_sequence_number(folder):
    existing = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    numbers  = []
    for f in existing:
        try:
            numbers.append(int(os.path.splitext(f)[0]))
        except ValueError:
            pass
    return (max(numbers) + 1) if numbers else 1


def _clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))


def detect_countdown_near_traffic_light(image, bbox):
    """
    Detect countdown digits near traffic light
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    
    # Search regions for countdown (right and below)
    search_regions = [
        # Right side
        [x2, max(0, y1 - bh//4), min(W, x2 + bw*2), min(H, y2 + bh//4)],
        # Below
        [max(0, x1 - bw//4), y2, min(W, x2 + bw//4), min(H, y2 + bh)],
        # Inside
        [x1, y1, x2, y2]
    ]
    
    for rx1, ry1, rx2, ry2 in search_regions:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        
        roi = image[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find bright digits
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for digit-like shapes
        digit_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 0.9:
                    digit_contours.append((x, y, w, h))
        
        # If we found 1-2 digits, it's likely a countdown
        if 1 <= len(digit_contours) <= 2:
            logger.info(f"Found {len(digit_contours)} potential countdown digits")
            # Simple simulation - in production, use OCR
            if len(digit_contours) == 1:
                return 5  # Single digit
            else:
                return 15  # Two digits
    
    return None


# -----------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------

def draw_detection_info(image, detection_info):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, f"Time: {timestamp}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    counts = detection_info['counts']
    cv2.putText(image,
                f"People: {counts['people']} | "
                f"Vehicles: {counts['vehicles']} | "
                f"Traffic Lights: {counts['traffic_lights']}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    tl = detection_info['traffic_light']
    if not tl['detected'] or tl['color'] == 'unknown':
        return

    # Color mapping
    if tl['color'] == 'red':
        bgr = (0, 0, 255)
        label = "RED LIGHT"
    elif tl['color'] == 'yellow':
        bgr = (0, 255, 255)
        label = "YELLOW LIGHT"
    elif tl['color'] == 'green':
        bgr = (0, 255, 0)
        label = "GREEN LIGHT"
    else:
        return

    # Confidence stars
    conf_stars = {
        'very_high': '★★★',
        'high': '★★',
        'medium': '★'
    }.get(tl['confidence'], '')

    cv2.putText(image, f"{label} {conf_stars}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2)

    if tl['countdown']:
        cv2.putText(image, f"Countdown: {tl['countdown']}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# -----------------------------------------------------------------------
# Main detection
# -----------------------------------------------------------------------

def detect_objects(image_path):
    """
    Main detection function with precise traffic light analysis
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original = image.copy()
    results = model.predict(image, imgsz=640, conf=0.25)

    detected_objects, objects_detailed = [], []
    traffic_light_info = {
        'detected': False,
        'color': 'unknown',
        'countdown': None,
        'bbox': None,
        'countdown_detected': False,
        'confidence': 'low',
        'lit_region': None
    }
    
    person_count = vehicle_count = tl_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            
            if name not in TARGET_CLASSES or conf < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name == 'person':
                person_count += 1
                color = (255, 255, 0)  # Cyan
                
            elif name in ('car', 'truck', 'bus', 'motorcycle', 'bicycle'):
                vehicle_count += 1
                color = (0, 255, 255)  # Yellow
                
            elif name == 'traffic light':
                tl_count += 1
                
                # Analyze traffic light using our detector
                tl_result = traffic_light_detector.analyze_traffic_light(image, (x1, y1, x2, y2))
                
                # Detect countdown separately
                countdown = detect_countdown_near_traffic_light(image, (x1, y1, x2, y2))
                
                # Update traffic light info
                traffic_light_info = {
                    'detected': True,
                    'color': tl_result['color'],
                    'countdown': countdown,
                    'bbox': [x1, y1, x2, y2],
                    'countdown_detected': countdown is not None,
                    'confidence': tl_result['confidence'],
                    'lit_region': tl_result.get('lit_region', None)
                }
                
                # Set box color based on detected color
                if tl_result['color'] == 'red':
                    color = (0, 0, 255)
                elif tl_result['color'] == 'yellow':
                    color = (0, 255, 255)
                elif tl_result['color'] == 'green':
                    color = (0, 255, 0)
                else:
                    color = (128, 128, 128)  # Gray for unknown
                
                # Log detection
                if tl_result['color'] != 'unknown':
                    logger.info(f"✅ Traffic Light: {tl_result['color'].upper()} "
                              f"(conf: {tl_result['confidence']}, countdown: {countdown})")
                else:
                    logger.info(f"❌ No traffic light color detected")
                
            else:
                color = (128, 128, 128)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw internal dividers for traffic lights
            if name == 'traffic light':
                height = y2 - y1
                region_h = height // 3
                for i in range(1, 3):
                    line_y = y1 + i * region_h
                    cv2.line(image, (x1, line_y), (x2, line_y), (200, 200, 200), 1)

            # Build label
            if name == 'traffic light':
                if traffic_light_info['color'] != 'unknown':
                    label = f"TL: {traffic_light_info['color'].upper()}"
                    if traffic_light_info['countdown']:
                        label += f" ({traffic_light_info['countdown']}s)"
                    
                    # Add confidence stars
                    if traffic_light_info['confidence'] == 'very_high':
                        label += " ★★★"
                    elif traffic_light_info['confidence'] == 'high':
                        label += " ★★"
                    elif traffic_light_info['confidence'] == 'medium':
                        label += " ★"
                else:
                    label = "TL: UNKNOWN"
            else:
                label = f"{name} {conf:.2f}"

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Store detection
            objects_detailed.append({
                "class": name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            detected_objects.append(name)

    # Prepare detection info
    detection_info = {
        'counts': {
            'people': person_count,
            'vehicles': vehicle_count,
            'traffic_lights': tl_count,
            'total': len(detected_objects),
        },
        'traffic_light': traffic_light_info,
    }
    
    # Draw information on image
    draw_detection_info(image, detection_info)

    # Save images
    seq_num = get_next_sequence_number(DETECTION_FOLDER)
    filename = f"{seq_num}.jpg"
    save_path = os.path.join(DETECTION_FOLDER, filename)
    cv2.imwrite(save_path, image)

    orig_path = os.path.join(UPLOAD_FOLDER, filename)
    if image_path != orig_path:
        cv2.imwrite(orig_path, original)

    # Log summary
    logger.info(f"Detection #{seq_num}: {person_count} people, "
                f"{vehicle_count} vehicles, {tl_count} traffic lights")

    # Return results
    return {
        "id": seq_num,
        "sequence_number": seq_num,
        "image_path": save_path,
        "original_image": orig_path,
        "image_url": f"/detections/{seq_num}.jpg",
        "original_url": f"/uploads/{seq_num}.jpg",
        "objects": detected_objects,
        "objects_detailed": objects_detailed,
        "traffic_light": traffic_light_info,
        "traffic_light_detected": traffic_light_info['detected'] and traffic_light_info['color'] != 'unknown',
        "traffic_light_color": traffic_light_info['color'],
        "traffic_light_countdown": traffic_light_info['countdown'],
        "traffic_light_confidence": traffic_light_info['confidence'],
        "traffic_light_lit_region": traffic_light_info.get('lit_region'),
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "detection_count": len(detected_objects),
        "counts": detection_info['counts'],
        "timestamp": datetime.now().isoformat(),
    }


def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Warm-up
    for _ in range(5):
        cap.read()

    # Capture best frame
    best_frame, best_sharp = None, 0
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            sharp = cv2.Laplacian(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp > best_sharp:
                best_sharp, best_frame = sharp, frame

    cap.release()
    if best_frame is None:
        raise Exception("Failed to capture from webcam")

    seq_num = get_next_sequence_number(UPLOAD_FOLDER)
    filepath = os.path.join(UPLOAD_FOLDER, f"{seq_num}.jpg")
    cv2.imwrite(filepath, best_frame)
    logger.info(f"Captured #{seq_num} (sharpness={best_sharp:.1f})")
    return filepath, seq_num


__all__ = ['detect_objects', 'capture_from_webcam']