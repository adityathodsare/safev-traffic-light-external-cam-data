from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np
import logging
from app.traffic_light_detector import traffic_light_detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = YOLO("yolov8n.pt")

# Create folders
DETECTION_FOLDER = "detections"
UPLOAD_FOLDER = "uploads"
os.makedirs(DETECTION_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Target classes
TARGET_CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light']

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

def draw_detection_info(image, detection_info):
    """Draw all detection information on image"""
    
    # Add timestamp
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
    y_offset = 60
    cv2.putText(
        image,
        f"People: {detection_info['counts']['people']} | "
        f"Vehicles: {detection_info['counts']['vehicles']} | "
        f"Traffic Lights: {detection_info['counts']['traffic_lights']}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Add traffic light info if detected
    if detection_info['traffic_light']['detected']:
        y_offset += 30
        tl_info = detection_info['traffic_light']
        
        # Color code the text based on traffic light color
        if tl_info['color'] == 'red':
            text_color = (0, 0, 255)
            color_text = f"🔴 RED LIGHT"
        elif tl_info['color'] == 'yellow':
            text_color = (0, 255, 255)
            color_text = f"🟡 YELLOW LIGHT"
        elif tl_info['color'] == 'green':
            text_color = (0, 255, 0)
            color_text = f"🟢 GREEN LIGHT"
        else:
            text_color = (255, 255, 255)
            color_text = f"⚪ TRAFFIC LIGHT"
        
        cv2.putText(
            image,
            color_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )
        
        # Add countdown info if detected
        if tl_info['countdown']:
            y_offset += 30
            cv2.putText(
                image,
                f"⏱️ Countdown: {tl_info['countdown']} seconds",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
        elif tl_info['detected']:
            y_offset += 30
            cv2.putText(
                image,
                "⏱️ Countdown: Not detected",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (128, 128, 128),
                2
            )

def detect_objects(image_path):
    """
    Main detection function - this must be named exactly 'detect_objects'
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        original_image = image.copy()
        
        # Run YOLO detection
        results = model(image)
        
        # Initialize detection containers
        detected_objects = []
        objects_detailed = []
        traffic_light_info = {
            'detected': False,
            'color': 'unknown',
            'countdown': None,
            'bbox': None,
            'countdown_detected': False,
            'confidence': 'low'
        }
        
        # Counters
        person_count = 0
        vehicle_count = 0
        traffic_light_count = 0
        
        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get detection info
                cls = int(box.cls[0])
                name = model.names[cls]
                confidence = float(box.conf[0])
                
                # Only process target classes with good confidence
                if name not in TARGET_CLASSES or confidence < 0.5:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Update counters
                if name == 'person':
                    person_count += 1
                    color = (255, 255, 0)  # Cyan for people
                elif name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                    vehicle_count += 1
                    color = (0, 255, 255)  # Yellow for vehicles
                elif name == 'traffic light':
                    traffic_light_count += 1
                    
                    # Analyze traffic light using your detector
                    tl_analysis = traffic_light_detector.analyze_traffic_light(image, (x1, y1, x2, y2))
                    
                    traffic_light_info = {
                        'detected': True,
                        'color': tl_analysis['color'],
                        'countdown': tl_analysis['countdown'],
                        'bbox': [x1, y1, x2, y2],
                        'countdown_detected': tl_analysis['countdown_detected'],
                        'confidence': tl_analysis['confidence']
                    }
                    
                    # Color based on traffic light state
                    if tl_analysis['color'] == 'red':
                        color = (0, 0, 255)
                    elif tl_analysis['color'] == 'yellow':
                        color = (0, 255, 255)
                    elif tl_analysis['color'] == 'green':
                        color = (0, 255, 0)
                    else:
                        color = (255, 255, 255)
                else:
                    color = (128, 128, 128)  # Gray for other objects
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create detailed label
                if name == 'traffic light':
                    label = f"Traffic Light"
                    if tl_analysis['color'] != 'unknown':
                        label += f": {tl_analysis['color'].upper()}"
                    if tl_analysis['countdown']:
                        label += f" ({tl_analysis['countdown']}s)"
                else:
                    label = f"{name} {confidence:.2f}"
                
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
                
                # Store detection
                objects_detailed.append({
                    "class": name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
                
                detected_objects.append(name)
        
        # Prepare detection info
        detection_info = {
            'counts': {
                'people': person_count,
                'vehicles': vehicle_count,
                'traffic_lights': traffic_light_count,
                'total': len(detected_objects)
            },
            'traffic_light': traffic_light_info
        }
        
        # Draw all information on image
        draw_detection_info(image, detection_info)
        
        # Get sequence number
        seq_num = get_next_sequence_number(DETECTION_FOLDER)
        
        # Save annotated image
        filename = f"{seq_num}.jpg"
        save_path = os.path.join(DETECTION_FOLDER, filename)
        cv2.imwrite(save_path, image)
        
        # Save original image
        orig_filename = f"{seq_num}.jpg"
        orig_save_path = os.path.join(UPLOAD_FOLDER, orig_filename)
        if image_path != orig_save_path:
            cv2.imwrite(orig_save_path, original_image)
        
        # Log detection results
        logger.info(f"Detection #{seq_num}: {person_count} people, {vehicle_count} vehicles, "
                    f"{traffic_light_count} traffic lights")
        if traffic_light_info['detected']:
            logger.info(f"   Traffic Light: {traffic_light_info['color']}, "
                       f"Countdown: {traffic_light_info['countdown'] if traffic_light_info['countdown'] else 'NOT DETECTED'}")
        
        return {
            "id": seq_num,
            "image_path": save_path,
            "original_image": orig_save_path,
            "objects": detected_objects,
            "objects_detailed": objects_detailed,
            "traffic_light": traffic_light_info,
            "counts": detection_info['counts'],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}")
        raise

def capture_from_webcam():
    """Capture image from webcam"""
    try:
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
        
        # Get sequence number
        seq_num = get_next_sequence_number(UPLOAD_FOLDER)
        
        # Save original image
        filename = f"{seq_num}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        
        logger.info(f"Captured image #{seq_num}")
        
        return filepath, seq_num
    
    except Exception as e:
        logger.error(f"Error in capture_from_webcam: {str(e)}")
        raise

# Explicitly export the functions
__all__ = ['detect_objects', 'capture_from_webcam']