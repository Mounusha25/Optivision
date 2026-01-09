"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Detection(BaseModel):
    """Single object detection"""
    class_name: str = Field(..., description="Detected class name")
    class_id: int = Field(..., description="COCO class ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class DetectionSummary(BaseModel):
    """Deterministic summary of detections"""
    total_objects: int = Field(..., description="Total number of detected objects")
    counts_by_class: dict = Field(..., description="Object counts grouped by class (person/vehicle/animal)")
    dominant_class: Optional[str] = Field(None, description="Class with highest count")
    frame_occupancy_ratio: float = Field(..., description="Ratio of frame area occupied by bboxes")
    recent_activity: dict = Field(..., description="Sliding window of recent class activity")
    events: list = Field(..., description="List of state change events")


class LatencyBreakdown(BaseModel):
    """Detailed latency breakdown"""
    preprocess: float = Field(..., description="Preprocessing time in ms")
    inference: float = Field(..., description="ONNX inference time in ms")
    postprocess: float = Field(..., description="Postprocessing (NMS) time in ms")
    total: float = Field(..., description="Total end-to-end latency in ms")


class DetectionMetadata(BaseModel):
    """Structured metadata for machine-readable telemetry"""
    request_id: str = Field(..., description="Unique request identifier for tracing")
    mean_confidence: float = Field(..., description="Mean confidence across all detections")
    max_confidence: float = Field(..., description="Maximum confidence score")
    min_confidence: float = Field(..., description="Minimum confidence score")
    objects_per_megapixel: float = Field(..., description="Detection density metric")
    input_resolution: List[int] = Field(..., description="Input image resolution [width, height]")
    model_version: str = Field(..., description="Model identifier")
    latency_breakdown_ms: LatencyBreakdown = Field(..., description="Per-stage latency breakdown")


class PredictResponse(BaseModel):
    """
    Deterministic response from /predict endpoint
    Contract: ALL fields are always present
    """
    detections: List[Detection] = Field(..., description="List of detected objects")
    summary: DetectionSummary = Field(..., description="Aggregated detection summary")
    metadata: DetectionMetadata = Field(..., description="Structured metadata and telemetry")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MetricsResponse(BaseModel):
    """Response from /metrics endpoint"""
    avg_latency_ms: float = Field(..., description="Average latency across all requests")
    p50_latency_ms: float = Field(..., description="50th percentile latency")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    requests_served: int = Field(..., description="Total number of requests served")
    total_detections: int = Field(..., description="Total objects detected")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")


class HealthResponse(BaseModel):
    """Response from /health endpoint"""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded successfully")
    model: str = Field(..., description="Loaded model name")
    model_type: str = Field(..., description="Model type (ONNX/PyTorch)")
    uptime: str = Field(..., description="Human-readable uptime")
    version: str = Field(default="1.0.0", description="API version")
