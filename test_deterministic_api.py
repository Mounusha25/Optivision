#!/usr/bin/env python3
"""
Test script to verify OptiVision's deterministic API contract.
Validates that /predict returns detections, summary, and metadata.
"""

import sys
import json
import base64
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core.inference import InferenceService
import numpy as np
import cv2


def create_test_image():
    """Create a simple test image (640x640 white square)."""
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    # Add a black rectangle to ensure some detection data
    cv2.rectangle(img, (100, 100), (300, 400), (0, 0, 0), -1)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def test_deterministic_response():
    """Test that the API returns a deterministic response structure."""
    
    print("🧪 Testing OptiVision Deterministic API Contract\n")
    print("=" * 60)
    
    # Initialize service
    service = InferenceService(model_path="backend/models/yolov8n.onnx")
    print("✅ InferenceService initialized")
    
    # Create test image (NumPy array)
    test_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (100, 100), (300, 400), (128, 128, 128), -1)
    print("✅ Test image created (640x640)\n")
    
    # Run inference
    detections, latency_ms, latency_breakdown = service.predict(test_image)
    
    # Compute summary and metadata using the new methods
    width, height = 640, 640
    summary = service.compute_summary(detections, width, height)
    metadata = service.compute_metadata(detections, width, height, latency_breakdown)
    
    # Combine into result structure (matching API response)
    result = {
        "detections": detections,
        "summary": summary.dict() if hasattr(summary, 'dict') else summary,
        "metadata": metadata.dict() if hasattr(metadata, 'dict') else metadata
    }
    
    # Validate response structure
    print("📊 Validating Response Structure:")
    print("-" * 60)
    
    # Check detections
    assert "detections" in result, "❌ Missing 'detections' key"
    print(f"✅ detections: {len(result['detections'])} objects detected")
    
    # Check summary
    assert "summary" in result, "❌ Missing 'summary' key"
    summary_dict = result["summary"]
    
    required_summary_fields = ["total_objects", "counts_by_class", "dominant_class", "frame_occupancy_ratio"]
    for field in required_summary_fields:
        assert field in summary_dict, f"❌ Missing summary field: {field}"
        print(f"✅ summary.{field}: {summary_dict[field]}")
    
    # Check metadata
    assert "metadata" in result, "❌ Missing 'metadata' key"
    metadata_dict = result["metadata"]
    
    required_metadata_fields = [
        "request_id", "mean_confidence", "max_confidence", "min_confidence",
        "objects_per_megapixel", "input_resolution", "model_version",
        "latency_breakdown_ms"
    ]
    
    for field in required_metadata_fields:
        assert field in metadata_dict, f"❌ Missing metadata field: {field}"
        value = metadata_dict[field]
        # Format display based on type
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"✅ metadata.{field}: {value:.4f}" if isinstance(value, float) and field != "model_version" else f"✅ metadata.{field}: {value}")
        else:
            print(f"✅ metadata.{field}: {value}")
    
    # Check latency breakdown
    latency = metadata_dict["latency_breakdown_ms"]
    required_latency_fields = ["preprocess", "inference", "postprocess", "total"]
    
    for field in required_latency_fields:
        assert field in latency, f"❌ Missing latency field: {field}"
        print(f"   ✅ latency_breakdown.{field}: {latency[field]:.2f}ms")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED - API CONTRACT IS DETERMINISTIC\n")
    
    # Print sample JSON
    print("📄 Sample JSON Response:")
    print("-" * 60)
    sample_response = {
        "detections": result["detections"][:2] if len(result["detections"]) > 0 else [],
        "summary": result["summary"],
        "metadata": result["metadata"]
    }
    print(json.dumps(sample_response, indent=2, default=str))
    

if __name__ == "__main__":
    try:
        test_deterministic_response()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
