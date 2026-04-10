#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import sys
import os

root = Path(__file__).resolve().parent
data_yaml = root / "qc_dataset.yaml"

print(f"Data yaml: {data_yaml} exists: {data_yaml.exists()}")
print(f"Current directory: {os.getcwd()}")
sys.stdout.flush()

model = YOLO('yolov8n.pt')
print("Model loaded, starting training...")
sys.stdout.flush()

try:
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=640,
        batch=2,  # Small batch for test
        device='cpu',  # Force CPU first to test
        workers=0,  # Critical for Windows!
        project=str(root / "runs" / "detect"),
        name="debug_run",
        verbose=True,
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()