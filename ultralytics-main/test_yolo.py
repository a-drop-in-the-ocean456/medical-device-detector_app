#!/usr/bin/env python3
from ultralytics import YOLO
import time
import sys

print("Starting test...")
sys.stdout.flush()

print("1. Imported YOLO")
sys.stdout.flush()

start = time.time()
try:
    model = YOLO('yolov8n.pt')
    print(f"2. Model loaded in {time.time() - start:.2f}s")
    sys.stdout.flush()
    
    # Test a simple predict to see if it works
    start = time.time()
    results = model.predict('https://ultralytics.com/images/bus.jpg', verbose=False)
    print(f"3. Predict test done in {time.time() - start:.2f}s")
    print(f"   Found {len(results[0].boxes)} objects")
    sys.stdout.flush()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")