import cv2
import numpy as np
import logging
from collections import deque, Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficLightDetector:
    def __init__(self):
        # Precise HSV ranges for traffic light colors
        self.color_ranges = {
            'red': [
                (np.array([0, 150, 150]), np.array([10, 255, 255])),    # Bright red
                (np.array([160, 150, 150]), np.array([180, 255, 255])), # Deep red
            ],
            'yellow': [
                (np.array([15, 150, 150]), np.array([35, 255, 255])),   # Standard yellow
            ],
            'green': [
                (np.array([40, 100, 100]), np.array([85, 255, 255])),   # Standard green
            ]
        }
        
        self.recent_colors = deque(maxlen=3)
        self.recent_countdowns = deque(maxlen=5)

    def detect_color_in_region(self, region_roi):
        """
        Detect color in a specific region using circle detection
        Returns color and confidence
        """
        if region_roi is None or region_roi.size == 0:
            return "unknown", 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=region_roi.shape[0] // 2,
            param1=50,
            param2=25,
            minRadius=3,
            maxRadius=min(region_roi.shape[0] // 2, 30)
        )
        
        if circles is None:
            return "unknown", 0
        
        circles = np.round(circles[0]).astype("int")
        
        best_color = "unknown"
        best_score = 0
        
        for (cx, cy, cr) in circles:
            if cy >= region_roi.shape[0] or cx >= region_roi.shape[1]:
                continue
            
            # Create circle mask
            circle_mask = np.zeros(region_roi.shape[:2], dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), cr, 255, -1)
            
            # Get color percentages in circle
            hsv = cv2.cvtColor(region_roi, cv2.COLOR_BGR2HSV)
            circle_pixels = cv2.countNonZero(circle_mask)
            
            if circle_pixels == 0:
                continue
            
            scores = {}
            for color, ranges in self.color_ranges.items():
                mask = np.zeros(hsv.shape[:2], np.uint8)
                for lower, upper in ranges:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
                mask = cv2.bitwise_and(mask, circle_mask)
                color_pixels = cv2.countNonZero(mask)
                scores[color] = (color_pixels / circle_pixels) * 100
            
            # Find best color
            for color, score in scores.items():
                if score > best_score and score > 40:  # Need >40% of circle
                    best_score = score
                    best_color = color
        
        return best_color, best_score

    def analyze_traffic_light(self, image, bbox):
        """
        Analyze traffic light by checking each region for circular lights
        Returns color and countdown
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Extract traffic light ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                'color': 'unknown',
                'countdown': None,
                'countdown_detected': False,
                'confidence': 'low'
            }
        
        # Split into three regions
        height = y2 - y1
        region_height = height // 3
        
        regions = {}
        if region_height > 0:
            regions['top'] = roi[0:region_height, :]
        if 2 * region_height <= height:
            regions['middle'] = roi[region_height:2*region_height, :]
        if 3 * region_height <= height:
            regions['bottom'] = roi[2*region_height:3*region_height, :]
        
        # Check each region for lit circular light
        detected_color = "unknown"
        lit_region = None
        confidence = "low"
        
        # Expected colors for each position
        expected_colors = {
            'top': 'red',
            'middle': 'yellow',
            'bottom': 'green'
        }
        
        for position, region_roi in regions.items():
            color, score = self.detect_color_in_region(region_roi)
            
            if color != "unknown":
                # Verify color matches expected position
                if color == expected_colors[position]:
                    detected_color = color
                    lit_region = position
                    confidence = "very_high" if score > 60 else "high"
                    logger.info(f"✅ Found {color} light in {position} region (score: {score:.1f}%)")
                    break
                else:
                    logger.info(f"⚠️ Found {color} in {position} but expected {expected_colors[position]}")
        
        # If no color found, try brightness-based detection
        if detected_color == "unknown":
            # Check which region is brightest
            max_brightness = 0
            brightest_region = None
            
            for position, region_roi in regions.items():
                gray = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
                mean_bright = np.mean(gray)
                
                if mean_bright > max_brightness and mean_bright > 60:
                    max_brightness = mean_bright
                    brightest_region = position
            
            if brightest_region:
                detected_color = expected_colors[brightest_region]
                confidence = "medium"
                lit_region = brightest_region
                logger.info(f"📊 Brightness-based: {brightest_region} region is brightest -> {detected_color}")
        
        # For now, return without countdown (we'll handle it separately)
        return {
            'color': detected_color,
            'countdown': None,  # We'll detect countdown separately in detector.py
            'countdown_detected': False,
            'confidence': confidence,
            'lit_region': lit_region
        }


# Singleton
traffic_light_detector = TrafficLightDetector()