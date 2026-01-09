# OptiVision - Local Docker Test

## Build the container
```bash
docker build -f Dockerfile.render -t optivision:latest .
```

## Run locally
```bash
docker run -p 8000:8000 \
  -e CONF_THRESHOLD=0.25 \
  -e IOU_THRESHOLD=0.45 \
  optivision:latest
```

## Test endpoints

Health check:
```bash
curl http://localhost:8000/health
```

Frontend:
```bash
open http://localhost:8000
```

## Expected health response
```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "models/yolov8n.onnx",
  "model_type": "ONNX",
  "uptime": "0h 0m 15s",
  "version": "1.0.0"
}
```

## Deploy to Render

See [DEPLOYMENT.md](DEPLOYMENT.md) for full Render deployment guide.
