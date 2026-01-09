"""
FastAPI Main Application
Production-grade object detection API
"""
import logging
import base64
import io
import uuid
import os
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

from app.core.inference import InferenceService
from app.core.metrics import MetricsTracker
from app.models.schemas import PredictResponse, MetricsResponse, HealthResponse, Detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OptiVision API",
    description="Edge-optimized real-time object detection with YOLOv8",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (initialized on startup)
inference_service: Optional[InferenceService] = None
metrics_tracker: Optional[MetricsTracker] = None


class Base64ImageRequest(BaseModel):
    """Request model for base64 encoded images"""
    image: str
    class_filter: Optional[str] = 'all'  # 'person', 'vehicle', 'animal', 'all'
    

@app.on_event("startup")
async def startup_event():
    """Initialize inference service and metrics tracker"""
    global inference_service, metrics_tracker
    
    try:
        # Read configuration from environment variables
        model_path = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
        conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.25"))
        iou_threshold = float(os.getenv("IOU_THRESHOLD", "0.45"))
        
        logger.info(f"Initializing InferenceService with model: {model_path}")
        logger.info(f"Configuration: conf={conf_threshold}, iou={iou_threshold}")
        
        inference_service = InferenceService(
            model_path=model_path,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_filter='all'  # Default: detect all classes
        )
        
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(window_size=1000)
        
        logger.info("✅ OptiVision API started successfully")
        logger.info(f"Model: {model_path}")
        logger.info(f"Input size: {inference_service.input_width}x{inference_service.input_height}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Base64ImageRequest):
    """
    Run object detection on base64 encoded image
    
    Args:
        request: Base64ImageRequest with image data and optional class_filter
        
    Returns:
        PredictResponse with detections and latency
    """
    try:
        # Update class filter if specified
        if request.class_filter and request.class_filter != inference_service.class_filter:
            inference_service.class_filter = request.class_filter
            logger.info(f"Class filter updated to: {request.class_filter}")
        # Decode base64 image
        image_data = base64.b64decode(request.image.split(',')[1] if ',' in request.image else request.image)
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Run inference with latency breakdown
        detections, latency_ms, latency_breakdown = inference_service.predict(image)
        
        # Record metrics with breakdown
        metrics_tracker.record_request(
            latency_ms, 
            len(detections),
            preprocess_ms=latency_breakdown['preprocess_ms'],
            inference_ms=latency_breakdown['inference_ms'],
            postprocess_ms=latency_breakdown['postprocess_ms']
        )
        
        # Log request
        logger.info(
            f"Request processed - Latency: {latency_ms:.2f}ms "
            f"(pre: {latency_breakdown['preprocess_ms']:.1f}ms, "
            f"inf: {latency_breakdown['inference_ms']:.1f}ms, "
            f"post: {latency_breakdown['postprocess_ms']:.1f}ms), "
            f"Detections: {len(detections)}, Image: {width}x{height}"
        )
        
        # Compute deterministic summary and metadata
        request_id = str(uuid.uuid4())
        summary = inference_service.compute_summary(detections, width, height)
        metadata = inference_service.compute_metadata(detections, width, height, latency_breakdown, request_id)
        
        # Build deterministic response (ALL fields always present)
        response = PredictResponse(
            detections=[Detection(**det) for det in detections],
            summary=summary,
            metadata=metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Run object detection on uploaded image file
    
    Args:
        file: Image file upload
        
    Returns:
        PredictResponse with detections and latency
    """
    try:
        # Read file
        contents = await file.read()
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Run inference with latency breakdown
        detections, latency_ms, latency_breakdown = inference_service.predict(image)
        
        # Record metrics with breakdown
        metrics_tracker.record_request(
            latency_ms, 
            len(detections),
            preprocess_ms=latency_breakdown['preprocess_ms'],
            inference_ms=latency_breakdown['inference_ms'],
            postprocess_ms=latency_breakdown['postprocess_ms']
        )
        
        logger.info(
            f"File processed - Latency: {latency_ms:.2f}ms "
            f"(pre: {latency_breakdown['preprocess_ms']:.1f}ms, "
            f"inf: {latency_breakdown['inference_ms']:.1f}ms, "
            f"post: {latency_breakdown['postprocess_ms']:.1f}ms), "
            f"Detections: {len(detections)}, Image: {width}x{height}"
        )
        
        # Compute deterministic summary and metadata
        request_id = str(uuid.uuid4())
        summary = inference_service.compute_summary(detections, width, height)
        metadata = inference_service.compute_metadata(detections, width, height, latency_breakdown, request_id)
        
        response = PredictResponse(
            detections=[Detection(**det) for det in detections],
            summary=summary,
            metadata=metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get aggregated inference metrics
    
    Returns:
        MetricsResponse with latency statistics and counts
    """
    try:
        metrics = metrics_tracker.get_metrics()
        return MetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/model")
async def get_model_info():
    """
    Get model metadata and configuration
    
    Returns:
        Model information including format, precision, and configuration
    """
    try:
        model_info = inference_service.get_model_info()
        
        # Enhanced model metadata
        return {
            "name": "YOLOv8n",
            "format": "ONNX",
            "precision": "FP32",
            "input_size": model_info['input_size'],
            "batch_size": 1,
            "num_classes": model_info['num_classes'],
            "confidence_threshold": model_info['confidence_threshold'],
            "iou_threshold": model_info['iou_threshold'],
            "model_path": model_info['model_path'],
            "backend": "ONNX Runtime (CPU)"
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with system status
    """
    try:
        model_info = inference_service.get_model_info()
        uptime = metrics_tracker.get_uptime_string()
        
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model=model_info['model_path'],
            model_type=model_info['model_type'],
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            model_loaded=False,
            model="unknown",
            model_type="unknown",
            uptime="0s"
        )


@app.get("/")
async def root():
    """Serve the demo UI"""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/detection")
async def detection_ui():
    """Alias for root - same detection UI"""
    return await root()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
