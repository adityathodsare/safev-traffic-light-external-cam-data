import cv2
import numpy as np
import re

class TrafficLightDetector:
    def __init__(self):
        self.light_colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0)
        }
        
    def detect_light_color(self, roi):
        """
        Detect traffic light color from ROI (Region of Interest)
        """
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255)],  # Red range 1
            'red2': [(160, 100, 100), (180, 255, 255)],  # Red range 2
            'yellow': [(20, 100, 100), (30, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)]
        }
        
        max_pixels = 0
        detected_color = "unknown"
        
        # Check each color
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixels = cv2.countNonZero(mask)
            
            # For red, combine both ranges
            if color == 'red2':
                continue
            if color == 'red':
                # Also check second red range
                mask2 = cv2.inRange(hsv, np.array(color_ranges['red2'][0]), 
                                   np.array(color_ranges['red2'][1]))
                pixels += cv2.countNonZero(mask2)
            
            if pixels > max_pixels and pixels > 50:  # Threshold to avoid noise
                max_pixels = pixels
                detected_color = color.replace('2', '')  # Remove '2' from red2
        
        return detected_color
    
    def detect_countdown(self, roi):
        """
        Detect countdown numbers in traffic light
        """
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to highlight numbers
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for number-like contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Numbers usually have aspect ratio between 0.3 and 0.8
                if 0.3 < aspect_ratio < 0.8:
                    # Extract number region
                    number_roi = thresh[y:y+h, x:x+w]
                    
                    # Here you'd use OCR to read the number
                    # For now, we'll simulate with a simple check
                    # You can integrate EasyOCR or Tesseract here
                    
                    # Placeholder for actual OCR
                    # Using pytesseract would be better
                    try:
                        import pytesseract
                        number_text = pytesseract.image_to_string(
                            number_roi, 
                            config='--psm 8 -c tessedit_char_whitelist=0123456789'
                        )
                        number = re.sub(r'\D', '', number_text)
                        if number:
                            return int(number)
                    except:
                        pass
        
        return None
    
    def analyze_traffic_light(self, image, bbox):
        """
        Complete analysis of traffic light
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        # Detect color
        color = self.detect_light_color(roi)
        
        # Detect countdown
        countdown = self.detect_countdown(roi)
        
        return {
            'color': color,
            'countdown': countdown,
            'bbox': [x1, y1, x2, y2]
        }

# Initialize global detector
traffic_light_detector = TrafficLightDetector()