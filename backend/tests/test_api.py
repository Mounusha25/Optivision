"""
API contract tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint returns expected structure"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "model_type" in data
    assert "uptime" in data
    assert "version" in data


def test_metrics_endpoint():
    """Test /metrics endpoint returns expected structure"""
    response = client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "avg_latency_ms" in data
    assert "p50_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "p99_latency_ms" in data
    assert "requests_served" in data
    assert "total_detections" in data
    assert "uptime_seconds" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert data["name"] == "OptiVision API"


def test_predict_invalid_image():
    """Test /predict with invalid image data"""
    response = client.post(
        "/predict",
        json={"image": "invalid_base64"}
    )
    assert response.status_code == 400


def test_predict_valid_structure():
    """Test /predict response structure (would need valid image)"""
    # Would need to create a valid base64 image for full test
    pass
