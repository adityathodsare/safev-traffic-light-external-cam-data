import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure pytesseract path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TrafficLightDetector:
    def __init__(self):
        self.light_colors = {
            'red': [(0, 100, 100), (10, 255, 255)],
            'red2': [(160, 100, 100), (180, 255, 255)],
            'yellow': [(20, 100, 100), (30, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)]
        }
        
    def detect_light_color(self, roi):
        """
        Enhanced color detection with better accuracy
        """
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        color_scores = {
            'red': 0,
            'yellow': 0,
            'green': 0
        }
        
        # Check each color
        for color, (lower, upper) in self.light_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # For red, combine both ranges
            if color == 'red2':
                continue
            if color == 'red':
                mask2 = cv2.inRange(hsv, np.array(self.light_colors['red2'][0]), 
                                   np.array(self.light_colors['red2'][1]))
                mask = cv2.bitwise_or(mask, mask2)
            
            # Calculate percentage of color in ROI
            color_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            percentage = (color_pixels / total_pixels) * 100
            
            color_scores[color] = percentage
        
        # Get the color with highest percentage above threshold
        max_color = max(color_scores, key=color_scores.get)
        max_percentage = color_scores[max_color]
        
        logger.info(f"Color percentages - Red: {color_scores['red']:.1f}%, "
                   f"Yellow: {color_scores['yellow']:.1f}%, "
                   f"Green: {color_scores['green']:.1f}%")
        
        if max_percentage > 15:  # Threshold for confidence
            return max_color
        else:
            return "unknown"
    
    def detect_countdown(self, roi):
        """
        Advanced countdown detection using multiple techniques
        """
        if roi.size == 0:
            return None
        
        countdown_results = []
        
        # Method 1: OCR on the entire ROI
        try:
            # Preprocess for OCR
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Multiple preprocessing techniques
            preprocessed_images = [
                gray,
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2),
                cv2.GaussianBlur(gray, (3, 3), 0)
            ]
            
            for processed in preprocessed_images:
                # Try different OCR configurations
                configs = [
                    '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single text line
                    '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word
                    '--psm 13 -c tessedit_char_whitelist=0123456789',  # Raw line
                ]
                
                for config in configs:
                    text = pytesseract.image_to_string(processed, config=config)
                    numbers = re.findall(r'\d+', text)
                    
                    for num in numbers:
                        if 1 <= int(num) <= 99:  # Reasonable countdown range
                            countdown_results.append(int(num))
                            logger.info(f"OCR detected countdown: {num}")
                            
        except Exception as e:
            logger.error(f"OCR error: {e}")
        
        # Method 2: Contour analysis for digit detection
        try:
            # Look for digit-like shapes
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 1000:  # Filter for digit-sized contours
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Digits typically have aspect ratio between 0.3 and 0.8
                    if 0.3 < aspect_ratio < 0.8:
                        # Extract potential digit region
                        digit_roi = gray[y:y+h, x:x+w]
                        digit_roi = cv2.resize(digit_roi, (28, 28))
                        
                        # Simple template matching could be added here
                        # For now, just note that a potential digit was found
                        logger.info(f"Potential digit shape found at {x},{y}")
                        
        except Exception as e:
            logger.error(f"Contour analysis error: {e}")
        
        # Return the most common countdown number found
        if countdown_results:
            # Return the most frequent number (mode)
            return max(set(countdown_results), key=countdown_results.count)
        
        return None
    
    def analyze_traffic_light(self, image, bbox):
        """
        Complete analysis of traffic light with confidence scores
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        # Detect color
        color = self.detect_light_color(roi)
        
        # Detect countdown (with expanded search area around traffic light)
        expanded_bbox = [
            max(0, x1 - 50),
            max(0, y1 - 80),
            min(image.shape[1], x2 + 50),
            min(image.shape[0], y2 + 30)
        ]
        
        expanded_roi = image[expanded_bbox[1]:expanded_bbox[3], 
                            expanded_bbox[0]:expanded_bbox[2]]
        
        countdown = self.detect_countdown(expanded_roi)
        
        # Determine confidence
        confidence = "high" if color != "unknown" else "low"
        if countdown:
            confidence = "high"
        
        return {
            'color': color,
            'countdown': countdown,
            'bbox': bbox,
            'expanded_bbox': expanded_bbox,
            'confidence': confidence,
            'countdown_detected': countdown is not None
        }

# Initialize global detector
traffic_light_detector = TrafficLightDetector()