# OptiVision Deployment Guide

**✅ Live Deployment**: [https://optivision-edge-inference-system.onrender.com](https://optivision-edge-inference-system.onrender.com)

## Overview

This guide covers deploying OptiVision to production using Docker and cloud platforms. The application is containerized using `Dockerfile.render` and can be deployed to any platform supporting Docker.

---

## Prerequisites

- Git repository with OptiVision code
- Docker installed (for local testing)
- Cloud platform account (Render, Railway, etc.)

---

## Quick Deploy to Render

### Step 1: Prepare Repository

Ensure your repository contains:

```
OptiVision/
├── backend/           # FastAPI application
├── frontend/          # HTML/JS frontend
├── models/            # ONNX model files
├── Dockerfile.render  # Production Dockerfile
└── README.md
```

### Step 2: Create Render Service

1. Navigate to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `Mounusha25/Optivision`
4. Configure service:

   | Setting | Value |
   |---------|-------|
   | **Name** | `optivision` |
   | **Environment** | `Docker` |
   | **Dockerfile Path** | `Dockerfile.render` |
   | **Region** | Select closest to your location |
   | **Instance Type** | Free (512MB) or Starter (2GB - recommended) |

### Step 3: Environment Variables

Add the following environment variables in Render dashboard:

| Variable | Value | Description |
|----------|-------|-------------|
| `CONF_THRESHOLD` | `0.25` | Confidence threshold for detections |
| `IOU_THRESHOLD` | `0.45` | IoU threshold for NMS |

**Note**: `PORT` is automatically set by Render - don't override it.

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will automatically:
   - Clone your repository
   - Build Docker image from `Dockerfile.render`
   - Deploy container
   - Assign public URL: `https://optivision-xxxx.onrender.com`

Build time: ~5-10 minutes (first deployment)

### Step 5: Verify Deployment

**✅ Example Deployment**: [https://optivision-edge-inference-system.onrender.com](https://optivision-edge-inference-system.onrender.com)

**Health Check:**
```bash
curl https://optivision-edge-inference-system.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "models/yolov8n.onnx",
  "model_type": "ONNX",
  "uptime": "0h 2m 15s",
  "version": "1.0.0"
}
```

**Frontend:**
Open your app URL in browser to access the real-time detection interface.

**OpenAPI Documentation:**
Visit `/docs` endpoint for interactive API documentation.

**API Test:**
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "class_filter": "person"
  }'
```

---

## Local Docker Testing

Before deploying to production, test the Docker container locally.

### Build Image

```bash
docker build -f Dockerfile.render -t optivision:latest .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -e CONF_THRESHOLD=0.25 \
  -e IOU_THRESHOLD=0.45 \
  --name optivision \
  optivision:latest
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Frontend
open http://localhost:8000

# API test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

### Stop Container

```bash
docker stop optivision
docker rm optivision
```

---

## Alternative Platforms

### Railway

Railway provides automatic Docker deployment with zero configuration.

**Deploy:**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and initialize
railway login
railway init

# Deploy
railway up
```

**Configuration:**
- Railway auto-detects `Dockerfile.render`
- Set environment variables in Railway dashboard
- No manual port configuration needed

**Access**: Railway provides a public URL automatically

### Fly.io

Fly.io offers edge deployment with global distribution.

**Deploy:**

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login and launch
fly auth login
fly launch --dockerfile Dockerfile.render

# Deploy
fly deploy
```

**Configuration:**
- Edit `fly.toml` to set environment variables
- Configure regions for edge deployment
- Scale instances as needed

### Google Cloud Run

Serverless container deployment with automatic scaling.

**Deploy:**

```bash
# Build and push to Container Registry
docker build -f Dockerfile.render -t gcr.io/PROJECT_ID/optivision .
docker push gcr.io/PROJECT_ID/optivision

# Deploy to Cloud Run
gcloud run deploy optivision \
  --image gcr.io/PROJECT_ID/optivision \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Configuration Reference

### Environment Variables

| Variable | Type | Default | Description | Required |
|----------|------|---------|-------------|----------|
| `CONF_THRESHOLD` | float | `0.25` | Confidence threshold (0.0-1.0) | No |
| `IOU_THRESHOLD` | float | `0.45` | IoU threshold for NMS (0.0-1.0) | No |
| `MODEL_PATH` | string | `models/yolov8n.onnx` | Path to ONNX model | No |
| `PORT` | int | `8000` | Server port (auto-set by platforms) | No |

### Resource Requirements

| Tier | RAM | CPU | Performance | Cost | Recommended For |
|------|-----|-----|-------------|------|-----------------|
| **Free** | 512MB | 0.1 vCPU | 3-5 req/sec | $0 | Development/testing |
| **Starter** | 2GB | 0.5 vCPU | 10-15 req/sec | ~$7/mo | Production (low traffic) |
| **Standard** | 4GB | 1 vCPU | 20-30 req/sec | ~$25/mo | Production (moderate traffic) |

---

## Troubleshooting

### Common Issues

**Issue**: Build fails with "model not found"

**Solution**:
- Verify `models/yolov8n.onnx` exists in repository
- Check `.dockerignore` doesn't exclude models/
- Ensure model file is committed to Git

---

**Issue**: Service crashes on startup

**Solution**:
- Check logs in platform dashboard
- Verify all dependencies in `requirements.txt`
- Test Docker image locally first
- Upgrade to larger instance if memory issue

---

**Issue**: Slow inference (>5 seconds)

**Causes**:
- Free tier CPU throttling
- Cold start (first request after idle)
- Large image size

**Solutions**:
- Upgrade to Starter tier or higher
- Implement request warming
- Reduce input image resolution

---

## Monitoring

### Health Checks

The `/health` endpoint provides application status:

```bash
curl https://your-app.onrender.com/health
```

### Performance Metrics

Access performance data via `/metrics`:

```bash
curl https://your-app.onrender.com/metrics
```

**Metrics include:**
- Average latency (P50/P95/P99)
- Request count
- Uptime

### Logging

Application logs are available through platform dashboards:

- **Render**: Logs tab in service dashboard
- **Railway**: Deployments → View logs
- **Fly.io**: `fly logs`

---

## Security

### HTTPS

All major platforms provide HTTPS by default with automatic SSL/TLS certificates.

### API Security

For production deployments, consider:

1. **Rate Limiting**: Implement request throttling
2. **Authentication**: Add API key validation
3. **CORS**: Configure allowed origins
4. **Input Validation**: Validate image size/format

---

## Cost Optimization

### Platform Comparison

| Platform | Free Tier | Starter Tier | Notes |
|----------|-----------|--------------|-------|
| **Render** | 512MB, sleeps after inactivity | $7/mo, always-on | Best for hobby projects |
| **Railway** | $5 free credit/mo | Pay-as-you-go | Good for variable traffic |
| **Fly.io** | 3 shared VMs free | $5+/mo per VM | Best for global edge deployment |

### Tips

1. **Use Free Tier**: For development and low-traffic demos
2. **Right-Size Instances**: Start small, scale up as needed
3. **Monitor Usage**: Track request volume and adjust accordingly
4. **Auto-Sleep**: Enable sleep mode for development environments

---

## Support

For deployment issues:

1. Check platform documentation
2. Review application logs
3. Test Docker container locally
4. Open issue on [GitHub](https://github.com/Mounusha25/Optivision/issues)

---

<div align="center">

**OptiVision Deployment Guide**

For more information, visit the [GitHub repository](https://github.com/Mounusha25/Optivision)

</div>
