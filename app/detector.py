from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np
import logging
from collections import Counter
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

def detect_traffic_light_region(image, bbox):
    """
    Extract traffic light regions for individual light analysis
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    # Extract the traffic light
    tl_roi = image[y1:y2, x1:x2]
    
    if tl_roi.size == 0:
        return {}
    
    # Traffic lights are usually vertical with 3 lights
    height = y2 - y1
    light_height = height // 3
    
    # Extract each light region
    lights = {}
    
    if light_height > 0:
        lights['top'] = tl_roi[0:light_height, :]
    if 2*light_height <= height:
        lights['middle'] = tl_roi[light_height:2*light_height, :]
    if 3*light_height <= height:
        lights['bottom'] = tl_roi[2*light_height:3*light_height, :]
    
    return lights

def analyze_lit_lights(lights):
    """
    Analyze which lights are actually lit
    """
    lit_lights = {}
    
    for position, light_roi in lights.items():
        if light_roi.size > 0:
            # Convert to grayscale to check brightness
            gray = cv2.cvtColor(light_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness statistics
            mean_brightness = np.mean(gray)
            max_brightness = np.max(gray)
            
            # A lit light will have high brightness values
            if mean_brightness > 80 and max_brightness > 150:
                # Determine color of this light
                hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
                
                # Check for red
                red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                red_pixels = cv2.countNonZero(red_mask)
                
                # Check for yellow
                yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
                yellow_pixels = cv2.countNonZero(yellow_mask)
                
                # Check for green
                green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))
                green_pixels = cv2.countNonZero(green_mask)
                
                total_pixels = light_roi.shape[0] * light_roi.shape[1]
                
                if total_pixels > 0:
                    red_pct = (red_pixels / total_pixels) * 100
                    yellow_pct = (yellow_pixels / total_pixels) * 100
                    green_pct = (green_pixels / total_pixels) * 100
                    
                    colors = {'red': red_pct, 'yellow': yellow_pct, 'green': green_pct}
                    dominant_color = max(colors, key=colors.get)
                    max_pct = colors[dominant_color]
                    
                    if max_pct > 20:
                        lit_lights[position] = {
                            'color': dominant_color,
                            'brightness': mean_brightness,
                            'confidence': max_pct
                        }
                        
                        logger.debug(f"Lit light at {position}: {dominant_color} ({max_pct:.1f}%)")
    
    return lit_lights

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
        
        # Add confidence indicator
        confidence_symbol = "⭐⭐⭐" if tl_info['confidence'] == 'very_high' else \
                           "⭐⭐" if tl_info['confidence'] == 'high' else \
                           "⭐" if tl_info['confidence'] == 'medium' else "?"
        
        cv2.putText(
            image,
            f"{color_text} {confidence_symbol}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )
        
        # Add countdown info
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
        
        # Add lit lights info
        if 'lit_lights' in tl_info and tl_info['lit_lights']:
            y_offset += 30
            lit_summary = ", ".join([f"{pos}:{info['color']}" for pos, info in tl_info['lit_lights'].items()])
            cv2.putText(
                image,
                f"Lit: {lit_summary}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

def detect_objects(image_path):
    """
    Main detection function with enhanced traffic light analysis
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        original_image = image.copy()
        
        # Run YOLO detection with higher resolution for better accuracy
        results = model.predict(image, imgsz=640, conf=0.25)
        
        # Initialize detection containers
        detected_objects = []
        objects_detailed = []
        traffic_light_info = {
            'detected': False,
            'color': 'unknown',
            'countdown': None,
            'bbox': None,
            'countdown_detected': False,
            'confidence': 'low',
            'lit_lights': {}
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
                
                # Only process target classes
                if name not in TARGET_CLASSES or confidence < 0.3:
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
                    
                    # Analyze individual lights
                    lights = detect_traffic_light_region(image, (x1, y1, x2, y2))
                    lit_lights = analyze_lit_lights(lights)
                    
                    # Use traffic light detector for full analysis with expanded search
                    # Expand bounding box to include potential countdown displays
                    expanded_x2 = min(image.shape[1], x2 + int((x2-x1)*1.5))
                    tl_analysis = traffic_light_detector.analyze_traffic_light(
                        image, [x1, y1, expanded_x2, y2]
                    )
                    
                    # Override color with lit light analysis if available
                    if lit_lights:
                        lit_colors = [info['color'] for info in lit_lights.values()]
                        if lit_colors:
                            color_counter = Counter(lit_colors)
                            primary_color = color_counter.most_common(1)[0][0]
                            
                            max_confidence = max([info['confidence'] for info in lit_lights.values()])
                            if max_confidence > 30 and primary_color != 'unknown':
                                tl_analysis['color'] = primary_color
                                tl_analysis['confidence'] = 'very_high'
                    
                    traffic_light_info = {
                        'detected': True,
                        'color': tl_analysis['color'],
                        'countdown': tl_analysis['countdown'],
                        'bbox': [x1, y1, x2, y2],
                        'countdown_detected': tl_analysis['countdown_detected'],
                        'confidence': tl_analysis['confidence'],
                        'lit_lights': lit_lights
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
                    color = (128, 128, 128)
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create detailed label
                if name == 'traffic light':
                    label = f"Traffic Light"
                    if tl_analysis['color'] != 'unknown':
                        label += f": {tl_analysis['color'].upper()}"
                    if tl_analysis['countdown']:
                        label += f" ({tl_analysis['countdown']}s)"
                    
                    # Add confidence indicator
                    if tl_analysis['confidence'] == 'very_high':
                        label += " ★★★"
                    elif tl_analysis['confidence'] == 'high':
                        label += " ★★"
                    elif tl_analysis['confidence'] == 'medium':
                        label += " ★"
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
            logger.info(f"   Traffic Light: {traffic_light_info['color']} "
                       f"(confidence: {traffic_light_info['confidence']})")
            logger.info(f"   Countdown: {traffic_light_info['countdown'] if traffic_light_info['countdown'] else 'NOT DETECTED'}")
            if traffic_light_info['lit_lights']:
                lit_summary = ", ".join([f"{pos}:{info['color']}" for pos, info in traffic_light_info['lit_lights'].items()])
                logger.info(f"   Lit lights: {lit_summary}")
        
        return {
            "id": seq_num,
            "sequence_number": seq_num,
            "image_path": save_path,
            "original_image": orig_save_path,
            "image_url": f"/detections/{seq_num}.jpg",
            "original_url": f"/uploads/{seq_num}.jpg",
            "objects": detected_objects,
            "objects_detailed": objects_detailed,
            "traffic_light": traffic_light_info,
            "traffic_light_detected": traffic_light_info['detected'],
            "traffic_light_color": traffic_light_info['color'],
            "traffic_light_countdown": traffic_light_info['countdown'],
            "traffic_light_confidence": traffic_light_info['confidence'],
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "detection_count": len(detected_objects),
            "counts": detection_info['counts'],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}")
        raise

def capture_from_webcam():
    """Capture image from webcam with enhanced quality"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set higher resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Warm up the camera
        for _ in range(5):
            cap.read()
        
        # Capture multiple frames and take the sharpest one
        best_frame = None
        best_sharpness = 0
        
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                # Calculate sharpness using Laplacian variance
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = frame
        
        cap.release()
        
        if best_frame is None:
            raise Exception("Failed to capture image from webcam")
        
        # Get sequence number
        seq_num = get_next_sequence_number(UPLOAD_FOLDER)
        
        # Save original image
        filename = f"{seq_num}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, best_frame)
        
        logger.info(f"Captured image #{seq_num} (sharpness: {best_sharpness:.2f})")
        
        return filepath, seq_num
    
    except Exception as e:
        logger.error(f"Error in capture_from_webcam: {str(e)}")
        raise

# Explicitly export the functions
__all__ = ['detect_objects', 'capture_from_webcam']