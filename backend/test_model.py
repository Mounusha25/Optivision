"""
Quick test script to verify ONNX model and inference
Run this after exporting the model
"""
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from app.core.inference import InferenceService


def test_inference():
    """Test inference with a sample image"""
    
    # Check if model exists
    model_path = Path("models/yolov8n.onnx")
    if not model_path.exists():
        print("❌ Model not found!")
        print("Run: python export_model.py")
        return
    
    print("✅ Model found")
    
    # Initialize service
    print("🔧 Initializing inference service...")
    service = InferenceService(str(model_path))
    
    # Create test image
    print("🎨 Creating test image...")
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    # Run inference
    print("🚀 Running inference...")
    detections, latency = service.predict(test_image)
    
    # Print results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"Latency: {latency:.2f} ms")
    print(f"Detections: {len(detections)}")
    
    if detections:
        print("\nDetected objects:")
        for i, det in enumerate(detections[:5], 1):  # Show first 5
            print(f"  {i}. {det['class_name']} - {det['confidence']:.2%}")
    
    # Model info
    info = service.get_model_info()
    print("\n" + "="*60)
    print("🤖 MODEL INFO")
    print("="*60)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_inference()
