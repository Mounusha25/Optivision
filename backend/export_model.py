"""
Export YOLOv8 model to ONNX format
Run this script to generate the ONNX model file
"""
from ultralytics import YOLO
from pathlib import Path

def export_yolov8_to_onnx():
    """Export YOLOv8n model to ONNX format"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Exporting YOLOv8n to ONNX...")
    print("This will download the pretrained model if not already cached")
    
    # Load YOLOv8n model (smallest, fastest)
    model = YOLO('yolov8n.pt')
    
    # Export to ONNX
    # Arguments:
    # - format='onnx': Export format
    # - imgsz=640: Input image size
    # - simplify=True: Simplify the ONNX model
    # - opset=12: ONNX opset version
    model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        opset=12
    )
    
    print("✅ Export complete!")
    print(f"Model saved as: yolov8n.onnx")
    print("\nMove the file to models/ directory:")
    print("  mkdir -p models && mv yolov8n.onnx models/")
    
    # Model comparison info
    print("\n" + "="*60)
    print("📊 YOLOv8 Model Comparison:")
    print("="*60)
    print("Model    | Size   | mAPval | Speed (CPU)")
    print("---------+--------+--------+--------------")
    print("YOLOv8n  | 6.3MB  | 37.3   | ~45ms  ← CURRENT")
    print("YOLOv8s  | 22MB   | 44.9   | ~100ms")
    print("YOLOv8m  | 52MB   | 50.2   | ~200ms")
    print("="*60)
    print("\nFor edge deployment, YOLOv8n is optimal (speed vs accuracy)")

if __name__ == "__main__":
    export_yolov8_to_onnx()
