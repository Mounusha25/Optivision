#!/usr/bin/env python3
"""Quick server test script"""
import requests
import time

print("Testing OptiVision Server...")
print("-" * 50)

# Test 1: Check if server is running
try:
    response = requests.get("http://localhost:8000/", timeout=5)
    print(f"✅ Server is running")
    print(f"   Status: {response.status_code}")
    print(f"   Content-Type: {response.headers.get('Content-Type')}")
    
    if 'html' in response.headers.get('Content-Type', ''):
        print(f"   Frontend served: {len(response.text)} bytes")
        if '<title>' in response.text:
            title = response.text.split('<title>')[1].split('</title>')[0]
            print(f"   Page title: {title}")
    
except Exception as e:
    print(f"❌ Server not responding: {e}")
    exit(1)

# Test 2: Check API endpoint
print("\n" + "-" * 50)
print("Testing /predict endpoint...")

# Create a small test image (1x1 black pixel in base64)
test_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"image": test_image},
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API working")
        print(f"   Detections: {len(data.get('detections', []))}")
        print(f"   Summary present: {bool(data.get('summary'))}")
        print(f"   Metadata present: {bool(data.get('metadata'))}")
        
        # Check for new temporal features
        summary = data.get('summary', {})
        if 'recent_activity' in summary:
            print(f"   ✅ recent_activity: PRESENT")
            activity = summary['recent_activity']
            print(f"      - window_frames: {activity.get('window_frames', 0)}")
            print(f"      - window_duration_s: {activity.get('window_duration_s', 0)}s")
        else:
            print(f"   ❌ recent_activity: MISSING")
        
        if 'events' in summary:
            print(f"   ✅ events: PRESENT ({len(summary['events'])} events)")
        else:
            print(f"   ❌ events: MISSING")
        
        if data.get('metadata'):
            latency = data['metadata'].get('latency_breakdown', {})
            print(f"   Inference time: {latency.get('inference_ms', 0):.1f}ms")
    else:
        print(f"❌ API error: {response.status_code}")
        print(f"   {response.text}")
        
except Exception as e:
    print(f"❌ API request failed: {e}")

print("\n" + "-" * 50)
print("RESULT: Everything working ✅" if response.status_code == 200 else "RESULT: Issues found ❌")
