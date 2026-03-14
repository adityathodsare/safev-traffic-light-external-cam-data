import cv2
import numpy as np
import logging
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficLightDetector:
    def __init__(self):
        # Precise HSV ranges for traffic light colors
        self.color_ranges = {
            'red': [
                (0, 150, 150), (10, 255, 255),     # Lower red
                (160, 150, 150), (180, 255, 255)   # Upper red
            ],
            'yellow': [(20, 150, 150), (35, 255, 255)],
            'green': [(40, 100, 100), (85, 255, 255)]
        }
        
        # Store recent detections for temporal smoothing
        self.recent_colors = deque(maxlen=5)
        self.recent_countdowns = deque(maxlen=5)
        
    def preprocess_roi(self, roi):
        """Enhanced preprocessing for better detection"""
        if roi.size == 0:
            return None
        
        # Increase size for better analysis
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to preserve edges while reducing noise
        roi = cv2.bilateralFilter(roi, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return roi

    def find_brightest_circle(self, roi, color_mask):
        """Find the brightest circular region in the masked area"""
        # Apply mask to find regions of interest
        masked = cv2.bitwise_and(roi, roi, mask=color_mask)
        if np.sum(masked) == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Find circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20,
            param1=50, 
            param2=30, 
            minRadius=5, 
            maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Return the brightest circle
            brightest = max(circles, key=lambda c: np.mean(gray[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]))
            return brightest
        
        return None

    def detect_light_color(self, roi):
        """
        Precise color detection using circularity and brightness
        """
        if roi.size == 0:
            return "unknown"
        
        # Preprocess the ROI
        roi = self.preprocess_roi(roi)
        if roi is None:
            return "unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        color_scores = {'red': 0, 'yellow': 0, 'green': 0}
        color_circles = {}
        
        # Analyze each color
        for color, ranges in self.color_ranges.items():
            if color == 'red':
                # Combine both red ranges
                mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower, upper = ranges
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find circular regions
            circles = self.find_brightest_circle(roi, mask)
            if circles is not None:
                x, y, r = circles
                # Extract the circular region
                circle_roi = roi[y-r:y+r, x-r:x+r]
                if circle_roi.size > 0:
                    # Calculate brightness of the circle
                    circle_hsv = cv2.cvtColor(circle_roi, cv2.COLOR_BGR2HSV)
                    brightness = np.mean(circle_hsv[:,:,2])
                    
                    # Calculate color purity in the circle
                    circle_mask = mask[y-r:y+r, x-r:x+r]
                    color_pixels = cv2.countNonZero(circle_mask)
                    total_pixels = circle_mask.shape[0] * circle_mask.shape[1]
                    
                    if total_pixels > 0:
                        purity = (color_pixels / total_pixels) * 100
                        # Score combines brightness and purity
                        score = (brightness / 255.0) * purity
                        color_scores[color] = score
                        color_circles[color] = (x, y, r, score)
                        
                        logger.debug(f"{color} circle - brightness: {brightness:.1f}, purity: {purity:.1f}%, score: {score:.1f}")
        
        # Get the color with highest score
        if max(color_scores.values()) > 20:  # Minimum threshold
            best_color = max(color_scores, key=color_scores.get)
            
            # Add to recent colors for temporal smoothing
            self.recent_colors.append(best_color)
            
            # Return the most common color from recent detections
            if len(self.recent_colors) > 0:
                from collections import Counter
                color_counter = Counter(self.recent_colors)
                return color_counter.most_common(1)[0][0]
            return best_color
        
        return "unknown"

    def extract_digits(self, roi):
        """
        Extract and recognize digits from countdown display
        """
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply adaptive thresholding to isolate digits
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        
        # Invert if necessary (digits are usually bright on dark background)
        if np.mean(gray) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort digit candidates
        digit_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 1000:  # Typical digit size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Digits typically have aspect ratio between 0.3 and 0.8
                if 0.3 < aspect_ratio < 0.9:
                    # Extract the digit region
                    digit_roi = binary[y:y+h, x:x+w]
                    digit_roi = cv2.resize(digit_roi, (20, 30))  # Standardize size
                    digit_regions.append((x, digit_roi))
        
        # Sort by x-coordinate (left to right)
        digit_regions.sort(key=lambda r: r[0])
        
        if len(digit_regions) == 0:
            return None
        
        # Simple digit recognition based on contour features
        # In production, you'd use a trained CNN here
        digits = []
        for _, digit_img in digit_regions:
            # Count white pixels (simplified recognition)
            white_pixels = cv2.countNonZero(digit_img)
            height, width = digit_img.shape
            total_pixels = height * width
            density = white_pixels / total_pixels
            
            # Very simple heuristic (replace with actual OCR)
            if density > 0.6:
                digits.append(8)
            elif density > 0.4:
                digits.append(0)
            else:
                digits.append(1)
        
        # Combine digits into a number
        if len(digits) == 1:
            number = digits[0]
        elif len(digits) == 2:
            number = digits[0] * 10 + digits[1]
        else:
            number = None
        
        if number:
            logger.info(f"Detected countdown: {number}")
            return number
        
        return None

    def detect_countdown(self, image, bbox):
        """
        Detect countdown digits near the traffic light
        """
        x1, y1, x2, y2 = bbox
        
        # Expand search area to catch external countdown displays
        search_areas = [
            # Below the traffic light (most common)
            [
                max(0, x1 - 30),
                min(image.shape[0], y2),
                min(image.shape[1], x2 + 30),
                min(image.shape[0], y2 + 100)
            ],
            # To the right of the traffic light
            [
                min(image.shape[1], x2),
                max(0, y1 - 20),
                min(image.shape[1], x2 + 80),
                min(image.shape[0], y2 + 20)
            ],
            # Inside the traffic light (for integrated displays)
            [x1, y1, x2, y2]
        ]
        
        for area_bbox in search_areas:
            ax1, ay1, ax2, ay2 = area_bbox
            if ay2 > ay1 and ax2 > ax1:
                area_roi = image[ay1:ay2, ax1:ax2]
                if area_roi.size > 0:
                    digits = self.extract_digits(area_roi)
                    if digits:
                        # Add to recent countdowns for smoothing
                        self.recent_countdowns.append(digits)
                        
                        # Return the most common recent value
                        if len(self.recent_countdowns) > 0:
                            from collections import Counter
                            countdown_counter = Counter(self.recent_countdowns)
                            return countdown_counter.most_common(1)[0][0]
                        return digits
        
        return None

    def analyze_traffic_light(self, image, bbox):
        """
        Complete analysis of traffic light
        """
        x1, y1, x2, y2 = bbox
        
        # Extract traffic light region
        roi = image[y1:y2, x1:x2]
        
        # Detect color
        color = self.detect_light_color(roi)
        
        # Detect countdown
        countdown = self.detect_countdown(image, bbox)
        
        # Determine confidence
        confidence = "low"
        if color != "unknown":
            confidence = "medium"
        if countdown:
            confidence = "high"
        if color != "unknown" and countdown:
            confidence = "very_high"
        
        result = {
            'color': color,
            'countdown': countdown,
            'bbox': bbox,
            'expanded_bbox': [x1-30, y1, x2+30, y2+100],
            'confidence': confidence,
            'countdown_detected': countdown is not None
        }
        
        logger.info(f"Traffic light analysis - Color: {color}, Countdown: {countdown}, Confidence: {confidence}")
        
        return result

# Initialize global detector
traffic_light_detector = TrafficLightDetector()