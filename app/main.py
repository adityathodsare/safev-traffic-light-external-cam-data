from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from app.detector import detect_objects, capture_from_webcam
from app.database import (
    save_detection, 
    get_latest, 
    get_history, 
    get_statistics,
    get_detection_by_sequence
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
DETECTION_FOLDER = "detections"
CAPTURE_INTERVAL = 20  # seconds

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Global variables for temporal smoothing
active_source = "webcam"
capture_task = None
last_countdown = None
last_color = "unknown"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global capture_task
    capture_task = asyncio.create_task(periodic_capture())
    logger.info("🚀 Periodic capture started (every 20 seconds)")
    yield
    if capture_task:
        capture_task.cancel()
        logger.info("👋 Periodic capture stopped")

app = FastAPI(
    title="SafeV Camera System",
    description="Traffic Light Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/detections", StaticFiles(directory="detections"), name="detections")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

async def periodic_capture():
    """Background task to capture and detect every 20 seconds"""
    global last_countdown, last_color
    
    while True:
        try:
            logger.info(f"📸 Capturing at {datetime.now()}")
            
            if active_source == "webcam":
                # Capture from webcam
                image_path, seq_num = capture_from_webcam()
                
                # Run detection
                result = detect_objects(image_path)
                
                # Temporal smoothing for countdown
                new_cd = result['traffic_light']['countdown']
                if new_cd is not None:
                    last_countdown = new_cd
                    last_color = result['traffic_light']['color']
                else:
                    # If no countdown detected, show previous - 1 (for decrementing display)
                    if last_countdown is not None and last_countdown > 0:
                        last_countdown -= 1
                        if last_countdown < 0:
                            last_countdown = None
                    
                    # Update result with smoothed values
                    result['traffic_light']['countdown'] = last_countdown
                    result['traffic_light']['color'] = last_color if last_color != "unknown" else result['traffic_light']['color']
                
                # Save to database
                save_detection(
                    result["id"],
                    result["image_path"],
                    result["original_image"],
                    result
                )
                
                logger.info(f"✅ Detection #{result['id']} complete")
                if result['traffic_light']['detected']:
                    logger.info(f"   Traffic Light: {result['traffic_light']['color']}")
                    if result['traffic_light']['countdown']:
                        logger.info(f"   Countdown: {result['traffic_light']['countdown']}s")
                
        except Exception as e:
            logger.error(f"❌ Periodic capture error: {str(e)}")
        
        await asyncio.sleep(CAPTURE_INTERVAL)

@app.get("/")
async def root():
    return {
        "name": "SafeV Traffic Light Detection",
        "version": "1.0.0",
        "capture_interval": f"{CAPTURE_INTERVAL} seconds",
        "active_source": active_source,
        "endpoints": {
            "GET /": "This info",
            "GET /webcam/capture": "Manual capture",
            "GET /latest": "Latest detection",
            "GET /history": "Detection history",
            "GET /stats": "Statistics",
            "GET /detection/{seq_num}": "Get detection by sequence number"
        }
    }

@app.get("/webcam/capture")
async def webcam_capture():
    """Manually trigger webcam capture"""
    global last_countdown, last_color
    
    try:
        image_path, seq_num = capture_from_webcam()
        result = detect_objects(image_path)
        
        # Temporal smoothing for manual capture
        new_cd = result['traffic_light']['countdown']
        if new_cd is not None:
            last_countdown = new_cd
            last_color = result['traffic_light']['color']
        else:
            result['traffic_light']['countdown'] = last_countdown
            result['traffic_light']['color'] = last_color if last_color != "unknown" else result['traffic_light']['color']
        
        save_detection(
            result["id"],
            result["image_path"],
            result["original_image"],
            result
        )
        
        # Add URLs for frontend
        result["image_url"] = f"/detections/{result['id']}.jpg"
        result["original_url"] = f"/uploads/{result['id']}.jpg"
        
        return result
    
    except Exception as e:
        logger.error(f"Manual capture failed: {str(e)}")
        raise HTTPException(500, f"Capture failed: {str(e)}")

@app.get("/latest")
async def latest_detection():
    """Get the most recent detection"""
    detection = get_latest()
    
    if detection is None:
        return {"message": "No detections yet"}
    
    # Add URLs
    detection["image_url"] = f"/detections/{detection['sequence_number']}.jpg"
    detection["original_url"] = f"/uploads/{detection['sequence_number']}.jpg"
    
    return detection

@app.get("/history")
async def detection_history(limit: int = 50):
    """Get detection history"""
    history = get_history(limit)
    
    for item in history:
        item["image_url"] = f"/detections/{item['sequence_number']}.jpg"
        item["original_url"] = f"/uploads/{item['sequence_number']}.jpg"
    
    return {
        "total": len(history),
        "detections": history
    }

@app.get("/detection/{seq_num}")
async def get_detection(seq_num: int):
    """Get detection by sequence number"""
    detection = get_detection_by_sequence(seq_num)
    
    if detection is None:
        raise HTTPException(404, f"Detection #{seq_num} not found")
    
    detection["image_url"] = f"/detections/{detection['sequence_number']}.jpg"
    detection["original_url"] = f"/uploads/{detection['sequence_number']}.jpg"
    
    return detection

@app.get("/stats")
async def get_stats():
    """Get detection statistics"""
    return get_statistics()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )