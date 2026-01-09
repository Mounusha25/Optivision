"""
Load testing script
Simulates concurrent requests to measure system performance
"""
import asyncio
import aiohttp
import time
import base64
import io
import numpy as np
import cv2
from statistics import mean, median
from pathlib import Path


async def create_test_image():
    """Create a test image as base64"""
    # Create a simple test image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


async def send_request(session, url, image_data, request_id):
    """Send a single prediction request"""
    start_time = time.time()
    
    try:
        async with session.post(
            f"{url}/predict",
            json={"image": image_data},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            result = await response.json()
            latency = (time.time() - start_time) * 1000
            
            return {
                'request_id': request_id,
                'status': response.status,
                'latency_ms': latency,
                'server_latency_ms': result.get('latency_ms', 0),
                'detections': len(result.get('detections', []))
            }
    except Exception as e:
        return {
            'request_id': request_id,
            'status': 'error',
            'error': str(e),
            'latency_ms': (time.time() - start_time) * 1000
        }


async def load_test(url, num_requests=10, concurrency=2):
    """
    Run load test
    
    Args:
        url: API base URL
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
    """
    print(f"🚀 Starting load test")
    print(f"   URL: {url}")
    print(f"   Total requests: {num_requests}")
    print(f"   Concurrency: {concurrency}")
    print("-" * 60)
    
    # Create test image
    image_data = await create_test_image()
    
    # Track results
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Create batches of concurrent requests
        for batch_start in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - batch_start)
            
            # Create tasks for this batch
            tasks = [
                send_request(session, url, image_data, batch_start + i)
                for i in range(batch_size)
            ]
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Progress update
            completed = len(results)
            print(f"Progress: {completed}/{num_requests} requests completed")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 LOAD TEST RESULTS")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 200]
    failed = [r for r in results if r['status'] != 200]
    
    print(f"\nRequests:")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if successful:
        client_latencies = [r['latency_ms'] for r in successful]
        server_latencies = [r['server_latency_ms'] for r in successful]
        
        print(f"\nClient Latency (total round-trip):")
        print(f"  Average: {mean(client_latencies):.2f} ms")
        print(f"  Median:  {median(client_latencies):.2f} ms")
        print(f"  Min:     {min(client_latencies):.2f} ms")
        print(f"  Max:     {max(client_latencies):.2f} ms")
        
        print(f"\nServer Latency (inference only):")
        print(f"  Average: {mean(server_latencies):.2f} ms")
        print(f"  Median:  {median(server_latencies):.2f} ms")
        print(f"  Min:     {min(server_latencies):.2f} ms")
        print(f"  Max:     {max(server_latencies):.2f} ms")
        
        total_detections = sum(r['detections'] for r in successful)
        print(f"\nDetections:")
        print(f"  Total: {total_detections}")
        print(f"  Average per request: {total_detections / len(successful):.1f}")
    
    if failed:
        print(f"\n⚠️  Failed Requests:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"  - Request {r['request_id']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test OptiVision API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--requests", type=int, default=10, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent requests")
    
    args = parser.parse_args()
    
    asyncio.run(load_test(args.url, args.requests, args.concurrency))
