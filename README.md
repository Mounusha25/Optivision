# OptiVision

**Real-time object detection inference service optimized for edge deployment with temporal intelligence and production-style monitoring.**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**🚀 [Live Demo](https://optivision-edge-inference-system.onrender.com)** • [Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Deployment](#deployment)

</div>

---

## Overview

**🌐 Live Demo**: [https://optivision-edge-inference-system.onrender.com](https://optivision-edge-inference-system.onrender.com)

This project exists to demonstrate how an ML model can be transformed into a reliable, real-time inference system with observable behavior and clean integration boundaries.

OptiVision is a deployment-ready object detection inference service built on YOLOv8n ONNX, delivering:

- **Low-latency inference** (<100ms) on CPU-constrained hardware
- **Temporal intelligence** with sliding window analysis and event detection
- **Real-time video streaming** with dual-loop architecture (~60fps rendering, ~20fps inference)
- **Comprehensive observability** with latency breakdown and performance metrics
- **Deployment support** via Docker containerization

### Design Philosophy

OptiVision prioritizes **real-time performance** and **operational visibility** for edge deployment scenarios. The system is intentionally scoped as a single-node inference service, avoiding distributed complexity while delivering production-style reliability.

### Scope & Non-Goals

OptiVision focuses on real-time ML inference and system observability. The following are intentionally out of scope for this project:

- Model training or fine-tuning
- Dataset curation
- Distributed serving or autoscaling
- Persistent storage or long-term analytics
- Full MLOps pipelines (CI/CD, retraining, registries)

These concerns are demonstrated in other projects in the portfolio.

---

## Features

### Core Capabilities

- **YOLOv8n ONNX Inference**: 12.3MB model optimized for CPU execution
- **Temporal Intelligence**: 
  - Sliding window analysis (30-frame buffer)
  - Event detection for object count changes (intentionally lightweight and scoped to short-term changes in object presence)
  - Recent activity tracking for person/vehicle/animal categories
- **Real-time Video Processing**:
  - ~60fps canvas rendering for smooth video display
  - ~20fps inference (CPU-bound) with intelligent backpressure control
  - Dual-loop architecture prevents UI blocking
- **Production-Style Observability**:
  - Latency breakdown (preprocess/inference/postprocess)
  - Performance metrics (P50/P95/P99)
  - Health monitoring endpoint
- **Class-Aware Detection**: Filter by person, vehicle, animal, or all objects

### Technical Highlights

| Component | Implementation | Performance |
|-----------|---------------|-------------|
| Model | YOLOv8n ONNX (12.3MB) | <100ms inference |
| Backend | FastAPI + ONNX Runtime | Async request handling |
| Frontend | Vanilla JavaScript | ~60fps video rendering |
| Temporal | Sliding window (deque) | 30-frame analysis |
| Deployment | Docker containerization | Deployment-ready |

---

## Architecture

### System Design

```
Client Browser
     |
     | HTTP (REST)
     v
┌─────────────────────────────────────┐
│      FastAPI Application             │
│  ┌───────────────────────────────┐   │
│  │ Inference Pipeline            │   │
│  │  - Preprocessing (~9ms)       │   │
│  │  - ONNX Inference (~41ms)     │   │
│  │  - Postprocessing (~7ms)      │   │
│  └───────────────────────────────┘   │
│  ┌───────────────────────────────┐   │
│  │ Temporal Intelligence         │   │
│  │  - Sliding Window (30 frames) │   │
│  │  - Event Detection            │   │
│  │  - Activity Tracking          │   │
│  └───────────────────────────────┘   │
│  ┌───────────────────────────────┐   │
│  │ Metrics & Observability       │   │
│  │  - Latency Breakdown          │   │
│  │  - Performance Tracking       │   │
│  └───────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Frontend Architecture

**Dual-Loop Design** for smooth real-time video:

- **Render Loop** (~60fps): Continuously draws video frames to canvas
- **Inference Loop** (~20fps, CPU-bound): Sends frames to backend API with backpressure control
- **Event System**: Updates detections overlay without blocking video stream

---

## 📊 API Contract (Deterministic)

### POST `/predict`

**All fields are always present (no optional magic).**

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "class_filter": "person"  // "person" | "vehicle" | "animal" | "all"
}
```

**Response:**
```json
{
  "detections": [
    {
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.89,
      "bbox": [120, 45, 340, 480]
    }
  ],
  "summary": {
    "total_objects": 3,
    "counts_by_class": {
      "person": 2,
      "vehicle": 1,
      "animal": 0
    },
    "dominant_class": "person",
    "frame_occupancy_ratio": 0.1847
  },
  "metadata": {
    "mean_confidence": 0.857,
    "max_confidence": 0.912,
    "min_confidence": 0.781,
    "objects_per_megapixel": 9.77,
    "input_resolution": [640, 480],
    "model_version": "yolov8n-onnx",
    "latency_breakdown_ms": {
      "preprocess": 9.2,
      "inference": 41.3,
      "postprocess": 6.6,
      "total": 57.1
    }
  },
  "timestamp": "2026-01-08T12:34:56.789Z"
}
```

**Why This Matters:**
- Downstream systems can make decisions without parsing raw bounding boxes
- Machine-readable telemetry for programmatic consumption
- No guesswork about schema structure

---

## Quick Start

### Prerequisites

- Python 3.9+ (tested with 3.9, 3.10, 3.11)
- 4GB RAM minimum
- Webcam (for real-time video demo)

### Local Development

```bash
# Clone repository
git clone https://github.com/Mounusha25/Optivision.git
cd OptiVision

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start backend server
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access Application

- **Frontend**: http://localhost:8000/ (served at root)
- **API Docs**: http://localhost:8000/docs (OpenAPI)
- **Health Check**: http://localhost:8000/health

---

## API Documentation

**Live API**: [https://optivision-edge-inference-system.onrender.com](https://optivision-edge-inference-system.onrender.com)

**OpenAPI Docs**: [https://optivision-edge-inference-system.onrender.com/docs](https://optivision-edge-inference-system.onrender.com/docs)

### Health Endpoint

**GET** `/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "models/yolov8n.onnx",
  "model_type": "ONNX",
  "uptime": "0h 15m 32s",
  "version": "1.0.0"
}
```

### Detection Endpoint

**POST** `/predict`

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "class_filter": "person"
}
```

**Parameters:**
- `image` (required): Base64-encoded image (JPEG/PNG)
- `class_filter` (optional): Filter detections by class
  - `"person"`: Detect only people
  - `"vehicle"`: Detect cars, trucks, buses, motorcycles
  - `"animal"`: Detect cats, dogs, horses, etc.
  - `"all"`: Detect all 80 COCO classes (default)

**Response:**
```json
{
  "detections": [
    {
      "class_name": "person",
      "class_id": 0,
      "confidence": 0.89,
      "bbox": [120, 45, 340, 480]
    }
  ],
  "summary": {
    "total_objects": 3,
    "counts_by_class": {"person": 2, "vehicle": 1, "animal": 0},
    "dominant_class": "person",
    "frame_occupancy_ratio": 0.18,
    "recent_activity": {
      "window_frames": 30,
      "class_presence": {"person": 28, "vehicle": 15, "animal": 0}
    },
    "events": [
      {"type": "count_increase", "class": "person", "change": 1}
    ]
  },
  "metadata": {
    "mean_confidence": 0.86,
    "input_resolution": [640, 480],
    "model_version": "yolov8n-onnx",
    "latency_breakdown_ms": {
      "preprocess": 9.2,
      "inference": 41.3,
      "postprocess": 6.6,
      "total": 57.1
    }
  },
  "timestamp": "2026-01-08T12:34:56.789Z"
}
```

### Metrics Endpoint

**GET** `/metrics`

```json
{
  "avg_latency_ms": 54.2,
  "p50_latency_ms": 52.8,
  "p95_latency_ms": 68.1,
  "p99_latency_ms": 74.3,
  "requests_served": 1547,
  "uptime_seconds": 342.6
}
```

---

## Deployment

### Docker Deployment

OptiVision includes a deployment-ready Docker configuration.

**Build and Run:**

```bash
# Build Docker image
docker build -f Dockerfile.render -t optivision:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e CONF_THRESHOLD=0.25 \
  -e IOU_THRESHOLD=0.45 \
  --name optivision \
  optivision:latest

# Test deployment
curl http://localhost:8000/health
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CONF_THRESHOLD` | 0.25 | Confidence threshold for detections |
| `IOU_THRESHOLD` | 0.45 | IoU threshold for NMS |
| `PORT` | 8000 | Server port |

### Cloud Deployment

#### Render

1. Fork/clone repository to GitHub
2. Create new **Web Service** on [render.com](https://render.com)
3. Connect GitHub repository
4. Configure:
   - **Environment**: Docker
   - **Dockerfile**: `Dockerfile.render`
   - **Instance Type**: Free or Starter ($7/mo recommended)
5. Add environment variables (CONF_THRESHOLD, IOU_THRESHOLD)
6. Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

#### Other Platforms

OptiVision can be deployed on any platform supporting Docker:
- **Railway**: Auto-detect Dockerfile
- **Fly.io**: `fly launch --dockerfile Dockerfile.render`
- **AWS ECS/Fargate**: Use Dockerfile.render
- **Google Cloud Run**: Deploy from container registry

---

## Project Structure

```
OptiVision/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── inference.py    # ONNX inference engine
│   │   │   └── metrics.py      # Performance tracking
│   │   ├── models/
│   │   │   └── schemas.py      # Pydantic response models
│   │   └── main.py             # FastAPI application
│   ├── models/
│   │   └── yolov8n.onnx       # Exported ONNX model
│   ├── tests/
│   │   ├── test_api.py        # API integration tests
│   │   └── test_inference.py  # Unit tests
│   └── requirements.txt
├── frontend/
│   └── index.html             # Real-time detection UI
├── models/
│   └── yolov8n.onnx          # ONNX model (production)
├── Dockerfile.render          # Production Docker configuration
├── DEPLOYMENT.md              # Deployment guide
└── README.md
```

---

## Configuration

### Adjust Detection Thresholds

Modify environment variables or update `backend/app/main.py`:

```python
# In main.py
conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.25"))
iou_threshold = float(os.getenv("IOU_THRESHOLD", "0.45"))
```

### Enable Additional Classes

Add custom class filters in `backend/app/core/inference.py`:

```python
CLASS_FILTERS = {
    "person": [0],
    "vehicle": [2, 3, 5, 7],
    "animal": [15, 16, 17, 18, 19, 20, 21, 22, 23],
    "custom": [0, 1, 2]  # Add your filter
}
```

---

## Performance Benchmarks

### Inference Latency (CPU)

| Component | Average | P95 | P99 |
|-----------|---------|-----|-----|
| **Total** | **57ms** | **68ms** | **74ms** |
| Preprocess | 9ms | 11ms | 13ms |
| Inference | 41ms | 49ms | 53ms |
| Postprocess | 7ms | 8ms | 9ms |

**Test Environment**: Apple M4, Python 3.11, ONNX Runtime 1.16+

### Throughput

- **Sequential**: 15-17 requests/second
- **Concurrent**: 8-12 requests/second (CPU bound)

### Model Comparison

| Model | Size | Latency | mAP | Use Case |
|-------|------|---------|-----|----------|
| YOLOv8n | 12.3MB | ~57ms | 37.3 | **Edge devices** (current) |
| YOLOv8s | 22MB | ~98ms | 44.9 | Balanced performance |
| YOLOv8m | 52MB | ~187ms | 50.2 | High accuracy |

---

## Testing

```bash
# Navigate to backend
cd backend

# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_api.py -v
pytest tests/test_inference.py -v
```

---

## Limitations

### Known Constraints

- **Small Object Detection**: YOLOv8n struggles with objects <32x32 pixels
- **Rare Classes**: Lower accuracy for uncommon COCO classes
- **Single Batch**: No batch inference (processes one image at a time)
- **CPU-Only**: No GPU acceleration in current deployment

### Design Decisions

- **Edge-First**: Optimized for CPU inference (<100ms)
- **Single-Node**: No distributed processing or horizontal scaling
- **Stateless**: In-memory metrics (no persistent storage)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Areas of interest:

- **Performance**: FP16/INT8 quantization, TensorRT backend
- **Features**: Batch inference, video file processing
- **Edge Devices**: Raspberry Pi, Jetson Nano compatibility
- **Documentation**: Tutorials, deployment guides

**Contribution Process:**

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push branch: `git push origin feature/your-feature`
5. Open Pull Request

---

## Contact

**GitHub**: [@Mounusha25](https://github.com/Mounusha25)  
**Repository**: [OptiVision](https://github.com/Mounusha25/Optivision)

---

<div align="center">

**Deployment-ready object detection inference service**

Built with FastAPI • ONNX Runtime • YOLOv8n

</div>
