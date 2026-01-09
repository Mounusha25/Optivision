"""
YOLOv8 ONNX Inference Service
Production-grade object detection with latency tracking
"""
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Dict
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Central inference service for YOLOv8 ONNX model
    Handles preprocessing, inference, and postprocessing with class-aware filtering
    """
    
    # Class groupings for production deployment
    CLASS_GROUPS = {
        'person': [0],  # person
        'vehicle': [2, 3, 5, 7],  # car, motorcycle, bus, truck
        'animal': [16, 17, 18, 19, 20, 21, 22, 23],  # cat, dog, horse, etc.
        'all': None  # No filtering
    }
    
    # Class-specific confidence thresholds for production reliability
    CLASS_THRESHOLDS = {
        0: 0.30,   # person - high reliability, slightly higher threshold
        2: 0.25,   # car
        3: 0.25,   # motorcycle
        5: 0.25,   # bus
        7: 0.25,   # truck
        16: 0.35,  # cat - smaller objects, higher threshold
        17: 0.35,  # dog
    }
    
    # Deployment mode configurations (latency/accuracy tradeoffs)
    MODES = {
        'low_latency': {
            'confidence': 0.35,  # Higher threshold = fewer detections = faster postprocess
            'iou': 0.50,         # More aggressive NMS = fewer boxes
            'description': 'Optimized for speed (<50ms target)'
        },
        'balanced': {
            'confidence': 0.25,  # Standard threshold
            'iou': 0.45,         # Standard NMS
            'description': 'Balanced speed/accuracy (default)'
        },
        'high_accuracy': {
            'confidence': 0.20,  # Lower threshold = more detections
            'iou': 0.40,         # Less aggressive NMS = more boxes
            'description': 'Optimized for recall (may be slower)'
        }
    }
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, class_filter: str = 'all', 
                 mode: str = 'balanced', slo_target_ms: float = 100.0):
        """
        Initialize ONNX inference session
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Base confidence threshold (overridden by mode)
            iou_threshold: IoU threshold for NMS (overridden by mode)
            class_filter: Filter mode ('person', 'vehicle', 'animal', 'all')
            mode: Deployment mode ('low_latency', 'balanced', 'high_accuracy')
            slo_target_ms: Service Level Objective for total latency (ms)
        """
        self.model_path = Path(model_path)
        self.mode = mode
        self.slo_target_ms = slo_target_ms
        
        # Apply mode-specific thresholds (override defaults)
        if mode in self.MODES:
            mode_config = self.MODES[mode]
            self.confidence_threshold = mode_config['confidence']
            self.iou_threshold = mode_config['iou']
            logger.info(f"Mode: {mode} - {mode_config['description']}")
        else:
            self.confidence_threshold = confidence_threshold
            self.iou_threshold = iou_threshold
            logger.warning(f"Unknown mode '{mode}', using custom thresholds")
        
        self.class_filter = class_filter
        
        # Sliding window for recent activity (last 30 frames)
        from collections import deque
        self.recent_frames = deque(maxlen=30)
        
        # State tracking for events
        self.previous_counts = {'person': 0, 'vehicle': 0, 'animal': 0}
        
        # Load COCO class names (YOLOv8 default)
        self.class_names = self._load_class_names()
        
        # Initialize ONNX Runtime session
        logger.info(f"Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']  # CPU-only for edge deployment
        )
        
        # Get model input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2] if len(input_shape) > 2 else 640
        self.input_width = input_shape[3] if len(input_shape) > 3 else 640
        
        logger.info(f"Model loaded successfully - Input shape: {input_shape}")
        logger.info(f"Input: {self.input_name}, Output: {self.output_name}")
        
    def _load_class_names(self) -> List[str]:
        """Load COCO class names"""
        # COCO 80 classes
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for YOLOv8 inference
        Maintains aspect ratio with letterboxing
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image, scale_x, scale_y for bbox rescaling
        """
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling factors
        scale = min(self.input_width / original_width, self.input_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create letterboxed image
        letterboxed = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # Calculate padding
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        
        # Place resized image in center
        letterboxed[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # Convert to RGB and normalize
        letterboxed = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        letterboxed = letterboxed.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        letterboxed = letterboxed.transpose(2, 0, 1)
        letterboxed = np.expand_dims(letterboxed, axis=0)
        
        # Calculate inverse scaling factors for bbox rescaling
        scale_x = original_width / self.input_width
        scale_y = original_height / self.input_height
        
        return letterboxed, scale_x, scale_y
    
    def postprocess(
        self, 
        outputs: np.ndarray, 
        scale_x: float, 
        scale_y: float,
        original_width: int,
        original_height: int
    ) -> List[Dict]:
        """
        Postprocess YOLO outputs with NMS
        
        Args:
            outputs: Raw model outputs
            scale_x: Horizontal scaling factor
            scale_y: Vertical scaling factor
            original_width: Original image width
            original_height: Original image height
            
        Returns:
            List of detections with class, confidence, and bbox
        """
        # YOLOv8 output format: [batch, 84, 8400] or [batch, num_classes + 4, num_anchors]
        # First 4 values are bbox (cx, cy, w, h), rest are class scores
        
        predictions = outputs[0]
        
        # Transpose if needed: [84, 8400] -> [8400, 84]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get class with max confidence
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Apply class-specific confidence thresholds
        mask = np.zeros(len(confidences), dtype=bool)
        for i, (class_id, conf) in enumerate(zip(class_ids, confidences)):
            # Get class-specific threshold or use default
            threshold = self.CLASS_THRESHOLDS.get(int(class_id), self.confidence_threshold)
            mask[i] = conf >= threshold
        
        # Filter by class group if specified
        if self.class_filter != 'all' and self.class_filter in self.CLASS_GROUPS:
            allowed_classes = self.CLASS_GROUPS[self.class_filter]
            if allowed_classes is not None:
                class_mask = np.isin(class_ids, allowed_classes)
                mask = mask & class_mask
        
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale boxes to original image size
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)
        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)
        
        # Apply NMS
        indices = self._nms(
            np.column_stack([x1, y1, x2, y2]),
            confidences,
            self.iou_threshold
        )
        
        # Build final detections
        detections = []
        for idx in indices:
            class_id = int(class_ids[idx])
            detections.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'confidence': float(confidences[idx]),
                'bbox': [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])]
            })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: Bounding boxes in format [x1, y1, x2, y2]
            scores: Confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def predict(self, image: np.ndarray) -> Tuple[List[Dict], float, Dict[str, float]]:
        """
        Run inference on image with latency breakdown
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (detections, total_latency_ms, latency_breakdown)
        """
        start_time = time.perf_counter()
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Preprocess
        preprocess_start = time.perf_counter()
        input_tensor, scale_x, scale_y = self.preprocess(image)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Run inference
        inference_start = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Postprocess
        postprocess_start = time.perf_counter()
        detections = self.postprocess(outputs, scale_x, scale_y, original_width, original_height)
        postprocess_time = (time.perf_counter() - postprocess_start) * 1000
        
        # Calculate total latency
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        latency_breakdown = {
            'preprocess_ms': preprocess_time,
            'inference_ms': inference_time,
            'postprocess_ms': postprocess_time
        }
        
        return detections, total_latency_ms, latency_breakdown
    
    def get_model_info(self) -> Dict:
        """Get model metadata"""
        return {
            'model_path': str(self.model_path),
            'model_type': 'ONNX',
            'input_size': f'{self.input_width}x{self.input_height}',
            'num_classes': len(self.class_names),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'class_filter': self.class_filter,
            'available_filters': list(self.CLASS_GROUPS.keys())
        }
    
    def compute_summary(self, detections: List[Dict], image_width: int, image_height: int) -> Dict:
        """
        Compute deterministic detection summary
        
        Args:
            detections: List of detection dictionaries
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Summary dictionary with counts and metrics
        """
        if not detections:
            # Update window even with no detections
            self.recent_frames.append(set())
            
            current_counts = {'person': 0, 'vehicle': 0, 'animal': 0}
            events = self._detect_count_events(current_counts)
            
            return {
                'total_objects': 0,
                'counts_by_class': current_counts,
                'dominant_class': None,
                'frame_occupancy_ratio': 0.0,
                'recent_activity': self.compute_recent_activity(),
                'events': events
            }
        
        # Count by class groups
        counts = {'person': 0, 'vehicle': 0, 'animal': 0}
        total_bbox_area = 0
        
        for det in detections:
            class_id = det['class_id']
            
            # Map to groups
            if class_id in self.CLASS_GROUPS['person']:
                counts['person'] += 1
            elif class_id in self.CLASS_GROUPS['vehicle']:
                counts['vehicle'] += 1
            elif class_id in self.CLASS_GROUPS['animal']:
                counts['animal'] += 1
            
            # Calculate bbox area
            x1, y1, x2, y2 = det['bbox']
            bbox_area = (x2 - x1) * (y2 - y1)
            total_bbox_area += bbox_area
        
        # Determine dominant class
        dominant_class = max(counts.items(), key=lambda x: x[1])[0] if any(counts.values()) else None
        
        # Frame occupancy ratio
        image_area = image_width * image_height
        frame_occupancy_ratio = total_bbox_area / image_area if image_area > 0 else 0.0
        
        # Detect events before updating state
        events = self._detect_count_events(counts)
        
        # Update sliding window with current frame's classes
        current_classes = set(det['class_name'] for det in detections)
        self.recent_frames.append(current_classes)
        
        return {
            'total_objects': len(detections),
            'counts_by_class': counts,
            'dominant_class': dominant_class,
            'frame_occupancy_ratio': round(frame_occupancy_ratio, 4),
            'recent_activity': self.compute_recent_activity(),
            'events': events
        }
    
    def compute_recent_activity(self) -> Dict:
        """
        Compute recent activity from sliding window
        
        Returns:
            Dictionary with window size and class presence counts
        """
        counts = {"person": 0, "vehicle": 0, "animal": 0}
        
        for frame_classes in self.recent_frames:
            for cls in frame_classes:
                # Map individual classes to groups
                if cls == "person":
                    counts["person"] += 1
                elif cls in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                    counts["vehicle"] += 1
                elif cls in ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]:
                    counts["animal"] += 1
        
        return {
            "window_frames": len(self.recent_frames),
            "class_presence": counts
        }
    
    def _detect_count_events(self, current_counts: Dict[str, int]) -> List[str]:
        """
        Detect count change events by comparing with previous frame
        
        Args:
            current_counts: Current frame counts by class group
            
        Returns:
            List of event strings
        """
        events = []
        
        for group in ['person', 'vehicle', 'animal']:
            curr = current_counts.get(group, 0)
            prev = self.previous_counts.get(group, 0)
            
            if curr > prev:
                events.append(f"{group}_count_increased")
            elif curr < prev:
                events.append(f"{group}_count_decreased")
        
        # Update state for next comparison
        self.previous_counts = current_counts.copy()
        
        return events
    
    def compute_metadata(self, detections: List[Dict], image_width: int, image_height: int, 
                        latency_breakdown: Dict[str, float], request_id: str = None) -> Dict:
        """
        Compute structured metadata for machine-readable telemetry
        
        Args:
            detections: List of detection dictionaries
            image_width: Original image width
            image_height: Original image height
            latency_breakdown: Dictionary with preprocess/inference/postprocess times
            request_id: Optional unique request identifier (generated if not provided)
            
        Returns:
            Metadata dictionary with confidence stats and performance metrics
        """
        import uuid
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        if not detections:
            return {
                'request_id': request_id,
                'mean_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'objects_per_megapixel': 0.0,
                'input_resolution': [image_width, image_height],
                'model_version': 'yolov8n-onnx',
                'latency_breakdown_ms': {
                    'preprocess': round(latency_breakdown['preprocess_ms'], 2),
                    'inference': round(latency_breakdown['inference_ms'], 2),
                    'postprocess': round(latency_breakdown['postprocess_ms'], 2),
                    'total': round(sum(latency_breakdown.values()), 2)
                }
            }
        
        # Confidence statistics
        confidences = [det['confidence'] for det in detections]
        mean_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        # Objects per megapixel
        megapixels = (image_width * image_height) / 1_000_000
        objects_per_megapixel = len(detections) / megapixels if megapixels > 0 else 0.0
        
        # Calculate total latency and SLO compliance
        total_latency = sum(latency_breakdown.values())
        slo_violated = total_latency > self.slo_target_ms
        
        return {
            'request_id': request_id,
            'mean_confidence': round(float(mean_confidence), 3),
            'max_confidence': round(float(max_confidence), 3),
            'min_confidence': round(float(min_confidence), 3),
            'objects_per_megapixel': round(objects_per_megapixel, 2),
            'input_resolution': [image_width, image_height],
            'model_version': 'yolov8n-onnx',
            'deployment_mode': self.mode,
            'latency_breakdown_ms': {
                'preprocess': round(latency_breakdown['preprocess_ms'], 2),
                'inference': round(latency_breakdown['inference_ms'], 2),
                'postprocess': round(latency_breakdown['postprocess_ms'], 2),
                'total': round(total_latency, 2)
            },
            'slo': {
                'target_ms': self.slo_target_ms,
                'violated': slo_violated,
                'margin_ms': round(self.slo_target_ms - total_latency, 2)
            }
        }
