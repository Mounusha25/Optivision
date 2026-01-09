# OptiVision - Render Deployment Guide

## Quick Deploy to Render

### 1. Prepare Repository
Ensure your GitHub repo is up to date with:
- `Dockerfile.render` (production-ready container)
- `backend/` with inference code
- `backend/models/yolov8n.onnx` (ONNX model)
- `frontend/index.html` (UI)

### 2. Create Render Service

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `optivision`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile.render`
   - **Instance Type**: `Free` (or `Starter` for better performance)

### 3. Set Environment Variables

In Render dashboard, add these environment variables:

```
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
PORT=8000
```

### 4. Deploy

Click **"Create Web Service"** - Render will:
- Build the Docker image
- Deploy the container
- Assign a public URL: `https://optivision-xxxx.onrender.com`

### 5. Test Deployment

Once deployed, test these endpoints:

**Health Check:**
```bash
curl https://optivision-xxxx.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/yolov8n.onnx",
  "input_size": "640x640"
}
```

**Frontend:**
Visit `https://optivision-xxxx.onrender.com` in browser

**API Test:**
```bash
curl -X POST https://optivision-xxxx.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONF_THRESHOLD` | 0.25 | Confidence threshold for detections |
| `IOU_THRESHOLD` | 0.45 | IoU threshold for NMS |
| `PORT` | 8000 | Server port (Render sets this automatically) |

### Resource Requirements

**Minimum (Free tier):**
- 512 MB RAM
- 0.1 CPU
- Works but may be slow

**Recommended (Starter - $7/month):**
- 2 GB RAM
- 0.5 CPU
- Much better inference performance

## Troubleshooting

### Build fails with "model not found"
Ensure `backend/models/yolov8n.onnx` exists in your repo

### Service crashes on startup
Check logs in Render dashboard - likely memory issue, upgrade to Starter tier

### Slow inference (>5s)
Free tier is CPU-limited, upgrade to Starter or use smaller images

## Production Notes

This deployment is:
- ✅ Containerized and reproducible
- ✅ Environment-configured
- ✅ Health-monitored
- ✅ Single-node (no unnecessary scaling)
- ✅ Interview-ready

This is **NOT**:
- ❌ Auto-scaling (not needed)
- ❌ Multi-region (overkill)
- ❌ CI/CD automated (optional bonus)
- ❌ Model retraining pipeline (different project)

## Interview Talking Points

When discussing deployment:

> "I deployed OptiVision as a containerized FastAPI service on Render. The Dockerfile includes the ONNX model, exposes health endpoints, and reads configuration from environment variables. This demonstrates reproducible ML deployment without overengineering. The service handles ~20fps inference on CPU and maintains deterministic responses."

Key terms to mention:
- Docker containerization
- Environment-based config
- Health endpoint monitoring
- Single-node deployment strategy
- Latency SLO tracking
