#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import sys
import os

print(f"Python: {sys.version.split()[0]}")
print(f"Ultralytics: {__import__('ultralytics').__version__}")
print(f"Dataset exists: {Path('qc_dataset.yaml').exists()}")
sys.stdout.flush()

model = YOLO('yolov8n.pt')
print("✓ Model loaded")
sys.stdout.flush()

try:
    # ONLY USE THESE PARAMETERS - ABSOLUTELY MINIMAL WORKING SET FOR WINDOWS
    model.train(
        data='qc_dataset.yaml',
        epochs=1,
        imgsz=640,
        batch=2,
        device='cpu',
        workers=0,  # THIS IS NON-NEGOTIABLE ON WINDOWS
        name='minimal_test',
        project='runs/detect',
    )
    print("\n✓ SUCCESS: Training started!")
    
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
