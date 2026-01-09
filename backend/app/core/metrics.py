"""
Metrics tracking and observability
"""
import time
from typing import List
from collections import deque
import numpy as np
from datetime import datetime


class MetricsTracker:
    """
    Track inference metrics for observability
    Maintains rolling window of latency measurements
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker
        
        Args:
            window_size: Maximum number of latency measurements to keep
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.preprocess_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.postprocess_times = deque(maxlen=window_size)
        self.fps_history = deque(maxlen=100)  # Track last 100 FPS measurements
        self.last_request_time = None
        self.requests_served = 0
        self.total_detections = 0
        self.start_time = time.time()
        
    def record_request(self, latency_ms: float, num_detections: int, 
                      preprocess_ms: float = 0, inference_ms: float = 0, postprocess_ms: float = 0):
        """
        Record a single inference request with latency breakdown
        
        Args:
            latency_ms: Total inference latency in milliseconds
            num_detections: Number of objects detected
            preprocess_ms: Time spent in preprocessing
            inference_ms: Time spent in model inference
            postprocess_ms: Time spent in postprocessing
        """
        self.latencies.append(latency_ms)
        self.preprocess_times.append(preprocess_ms)
        self.inference_times.append(inference_ms)
        self.postprocess_times.append(postprocess_ms)
        
        # Calculate FPS
        current_time = time.time()
        if self.last_request_time is not None:
            time_delta = current_time - self.last_request_time
            if time_delta > 0:
                fps = 1.0 / time_delta
                self.fps_history.append(fps)
        self.last_request_time = current_time
        
        self.requests_served += 1
        self.total_detections += num_detections
        
    def get_metrics(self) -> dict:
        """
        Calculate aggregate metrics with latency breakdown
        
        Returns:
            Dictionary with latencies, FPS, breakdown, and counts
        """
        if not self.latencies:
            return {
                'avg_latency_ms': 0.0,
                'p50_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0,
                'latency_breakdown': {
                    'preprocess_ms': 0.0,
                    'inference_ms': 0.0,
                    'postprocess_ms': 0.0
                },
                'fps': {
                    'current': 0.0,
                    'avg': 0.0,
                    'min': 0.0,
                    'max': 0.0
                },
                'requests_served': self.requests_served,
                'total_detections': self.total_detections,
                'uptime_seconds': time.time() - self.start_time
            }
        
        latency_array = np.array(self.latencies)
        
        # Calculate latency breakdown
        breakdown = {
            'preprocess_ms': float(np.mean(self.preprocess_times)) if self.preprocess_times else 0.0,
            'inference_ms': float(np.mean(self.inference_times)) if self.inference_times else 0.0,
            'postprocess_ms': float(np.mean(self.postprocess_times)) if self.postprocess_times else 0.0
        }
        
        # Calculate FPS statistics
        fps_stats = {
            'current': float(self.fps_history[-1]) if self.fps_history else 0.0,
            'avg': float(np.mean(self.fps_history)) if self.fps_history else 0.0,
            'min': float(np.min(self.fps_history)) if self.fps_history else 0.0,
            'max': float(np.max(self.fps_history)) if self.fps_history else 0.0
        }
        
        return {
            'avg_latency_ms': float(np.mean(latency_array)),
            'p50_latency_ms': float(np.percentile(latency_array, 50)),
            'p95_latency_ms': float(np.percentile(latency_array, 95)),
            'p99_latency_ms': float(np.percentile(latency_array, 99)),
            'latency_breakdown': breakdown,
            'fps': fps_stats,
            'requests_served': self.requests_served,
            'total_detections': self.total_detections,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def get_uptime_string(self) -> str:
        """Get human-readable uptime"""
        uptime_seconds = time.time() - self.start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"
    
    def reset(self):
        """Reset all metrics"""
        self.latencies.clear()
        self.requests_served = 0
        self.total_detections = 0
        self.start_time = time.time()
