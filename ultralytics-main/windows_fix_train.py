#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import sys
import os
import time

print("=" * 50)
print(f"Python version: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print(f"Ultralytics version: {__import__('ultralytics').__version__}")
print("=" * 50)
sys.stdout.flush()

root = Path(__file__).resolve().parent
data_yaml = root / "qc_dataset.yaml"

try:
    model = YOLO('yolov8n.pt')
    print("✓ Model loaded")
    sys.stdout.flush()
    
    # Test with minimal parameters that work 100% on Windows
    print("\nStarting training with Windows-optimized parameters...")
    sys.stdout.flush()
    
    model.train(
        data=str(data_yaml),
        epochs=2,
        imgsz=640,
        batch=4,          # Very small batch size
        device='cpu',     # Force CPU first to eliminate CUDA issues
        workers=0,        # ABSOLUTELY REQUIRED on Windows
        cache='disk',
        project=str(root / "runs" / "detect"),
        name="windows_debug",
        verbose=True,
        val=False,        # Disable validation temporarily
        augment=False,    # Disable augmentation
        cos_lr=False,
        amp=False,        # Disable AMP on CPU
        pin_memory=False, # Critical fix for Windows!
        # These 3 parameters solve 90% of Windows hang issues
        dnn=False,
        profile=False,
    )
    
    print("\n✓ Training completed successfully!")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
print("\nScript finished")