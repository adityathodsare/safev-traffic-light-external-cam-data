from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np
from app.traffic_light_detector import traffic_light_detector

# Load lightweight model
model = YOLO("yolov8n.pt")

# Create folders if they don't exist
DETECTION_FOLDER = "detections"
UPLOAD_FOLDER = "uploads"
os.makedirs(DETECTION_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Target classes (only what we need)
TARGET_CLASSES = ['person', 'car', 'traffic light']

def get_next_sequence_number(folder):
    """Get next sequential number for file naming"""
    existing_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        try:
            num = int(os.path.splitext(f)[0])
            numbers.append(num)
        except:
            continue
    
    return max(numbers) + 1 if numbers else 1

def detect_objects(image_path):
    """
    Detect objects with focus on traffic light color and countdown
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Run YOLO detection
    results = model(image)
    
    detected_objects = []
    traffic_light_info = {
        'detected': False,
        'color': 'unknown',
        'countdown': None,
        'bbox': None
    }
    
    person_count = 0
    car_count = 0
    
    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get class name
            cls = int(box.cls[0])
            name = model.names[cls]
            confidence = float(box.conf[0])
            
            # Only process target classes
            if name not in TARGET_CLASSES:
                continue
            
            # Only track if confidence is good enough
            if confidence > 0.5:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Count objects
                if name == 'person':
                    person_count += 1
                elif name == 'car':
                    car_count += 1
                
                # Special handling for traffic light
                if name == 'traffic light':
                    # Analyze traffic light color and countdown
                    tl_analysis = traffic_light_detector.analyze_traffic_light(image, (x1, y1, x2, y2))
                    
                    traffic_light_info = {
                        'detected': True,
                        'color': tl_analysis['color'],
                        'countdown': tl_analysis['countdown'],
                        'bbox': [x1, y1, x2, y2]
                    }
                    
                    # Color for bounding box based on traffic light color
                    if tl_analysis['color'] == 'red':
                        color = (0, 0, 255)  # Red
                    elif tl_analysis['color'] == 'yellow':
                        color = (0, 255, 255)  # Yellow
                    elif tl_analysis['color'] == 'green':
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 255, 0)  # Cyan for unknown
                else:
                    color = (255, 255, 255)  # White for other objects
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                if name == 'traffic light':
                    label = f"Traffic Light: {traffic_light_info['color']}"
                    if traffic_light_info['countdown']:
                        label += f" ({traffic_light_info['countdown']}s)"
                else:
                    label = f"{name}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
                
                detected_objects.append({
                    "class": name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
    
    # Get sequential number for saving
    seq_num = get_next_sequence_number(DETECTION_FOLDER)
    
    # Add timestamp and info to image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        image,
        f"Time: {timestamp}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Add counts
    cv2.putText(
        image,
        f"People: {person_count} | Cars: {car_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Add traffic light status
    if traffic_light_info['detected']:
        status = f"Traffic Light: {traffic_light_info['color']}"
        if traffic_light_info['countdown']:
            status += f" | Countdown: {traffic_light_info['countdown']}s"
        cv2.putText(
            image,
            status,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
    
    # Save annotated image with sequential name
    filename = f"{seq_num}.jpg"
    save_path = os.path.join(DETECTION_FOLDER, filename)
    cv2.imwrite(save_path, image)
    
    # Save original image with sequential name
    orig_filename = f"{seq_num}.jpg"
    orig_save_path = os.path.join(UPLOAD_FOLDER, orig_filename)
    if image_path != orig_save_path:  # Avoid copying if already in uploads
        cv2.imwrite(orig_save_path, cv2.imread(image_path))
    
    return {
        "id": seq_num,
        "image_path": save_path,
        "original_image": orig_save_path,
        "objects": [obj["class"] for obj in detected_objects],
        "objects_detailed": detected_objects,
        "traffic_light": traffic_light_info,
        "counts": {
            "people": person_count,
            "cars": car_count,
            "total": len(detected_objects)
        },
        "timestamp": timestamp
    }

def capture_from_webcam():
    """Capture image from laptop webcam with sequential naming"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Capture frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Failed to capture image from webcam")
    
    # Get sequential number
    seq_num = get_next_sequence_number(UPLOAD_FOLDER)
    
    # Save original image with sequential name
    filename = f"{seq_num}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    
    return filepath, seq_num