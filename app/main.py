from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from app.detector import detect_objects, capture_from_webcam
from app.database import (
    save_detection, get_latest, get_history, 
    get_detection_by_sequence
)

# Configuration
UPLOAD_FOLDER = "uploads"
DETECTION_FOLDER = "detections"
CAPTURE_INTERVAL = 20  # seconds

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Global variables
active_source = "webcam"
capture_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global capture_task
    capture_task = asyncio.create_task(periodic_capture())
    print("🚀 Periodic capture started (every 20 seconds)")
    yield
    if capture_task:
        capture_task.cancel()
        print("👋 Periodic capture stopped")

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
    while True:
        try:
            print(f"📸 Capturing at {datetime.now()}")
            
            if active_source == "webcam":
                # Capture from webcam
                image_path, seq_num = capture_from_webcam()
                
                # Run detection
                result = detect_objects(image_path)
                
                # Save to database
                save_detection(
                    result["id"],
                    result["image_path"],
                    result["original_image"],
                    result["objects"],
                    result["counts"]["people"],
                    result["counts"]["cars"],
                    result["traffic_light"]
                )
                
                print(f"✅ Detection #{result['id']} complete")
                print(f"   Traffic Light: {result['traffic_light']['color']}")
                if result['traffic_light']['countdown']:
                    print(f"   Countdown: {result['traffic_light']['countdown']}s")
                
        except Exception as e:
            print(f"❌ Periodic capture error: {e}")
        
        await asyncio.sleep(CAPTURE_INTERVAL)

@app.get("/")
async def root():
    return {
        "name": "SafeV Traffic Light Detection",
        "version": "1.0.0",
        "capture_interval": f"{CAPTURE_INTERVAL} seconds",
        "active_source": active_source
    }

@app.get("/webcam/capture")
async def webcam_capture():
    """Manually trigger webcam capture"""
    try:
        image_path, seq_num = capture_from_webcam()
        result = detect_objects(image_path)
        
        save_detection(
            result["id"],
            result["image_path"],
            result["original_image"],
            result["objects"],
            result["counts"]["people"],
            result["counts"]["cars"],
            result["traffic_light"]
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(500, f"Capture failed: {str(e)}")

@app.get("/latest")
async def latest_detection():
    """Get the most recent detection"""
    detection = get_latest()
    
    if detection is None:
        return {"message": "No detections yet"}
    
    # Add URLs
    detection["image_url"] = f"/{detection['image_path']}"
    detection["original_url"] = f"/{detection['original_image']}"
    
    return detection

@app.get("/history")
async def detection_history(limit: int = 50):
    """Get detection history"""
    history = get_history(limit)
    
    for item in history:
        item["image_url"] = f"/{item['image_path']}"
        item["original_url"] = f"/{item['original_image']}"
    
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
    
    detection["image_url"] = f"/{detection['image_path']}"
    detection["original_url"] = f"/{detection['original_image']}"
    
    return detection

@app.get("/stats")
async def get_stats():
    """Get quick statistics"""
    history = get_history(1000)
    
    total = len(history)
    traffic_lights = sum(1 for d in history if d['traffic_light_detected'])
    red_lights = sum(1 for d in history if d['traffic_light_color'] == 'red')
    green_lights = sum(1 for d in history if d['traffic_light_color'] == 'green')
    yellow_lights = sum(1 for d in history if d['traffic_light_color'] == 'yellow')
    
    return {
        "total_detections": total,
        "traffic_lights_detected": traffic_lights,
        "red_lights": red_lights,
        "yellow_lights": yellow_lights,
        "green_lights": green_lights,
        "total_people": sum(d['person_count'] for d in history),
        "total_cars": sum(d['car_count'] for d in history)
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )