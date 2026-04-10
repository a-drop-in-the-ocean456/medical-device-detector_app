#!/usr/bin/env python3
"""
Train YOLOv8n on the local QC dataset.

Examples:
    python train_yolov8n.py
    python train_yolov8n.py --epochs 150 --batch 24 --device 0
    python train_yolov8n.py --freeze 10 --name yolov8n_freeze10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Train YOLOv8n for object detection.")
    parser.add_argument("--data", type=Path, default=root / "qc_dataset.yaml", help="Dataset YAML path.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model checkpoint or yaml.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=24, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or 0,1. Use cpu for CPU.")
    parser.add_argument("--workers", type=int, default=0, # 8, 
                        help="Dataloader workers.")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers. 0 means train all layers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--project", type=Path, default=root / "runs" / "detect", help="Output project dir.")
    parser.add_argument("--name", type=str, default="yolov8n_full_new", help="Run name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", action="store_true", help="Cache images to speed up training.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic augmentation in last N epochs.")
    return parser.parse_args()


def to_abs(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    data_yaml = to_abs(args.data, root)
    project_dir = to_abs(args.project, root)
    project_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    print("Training config:")
    print(f"  model   : {args.model}")
    print(f"  data    : {data_yaml}")
    print(f"  epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch}")
    print(f"  device  : {args.device}")
    print(f"  freeze  : {args.freeze}")
    print(f"  project : {project_dir}")
    print(f"  name    : {args.name}")


    model = YOLO(args.model)
    print("Model loaded. Starting training...")
    print("Note: First run will take 5-10 minutes to process 175k images")
    # Avoid start-of-training integration hooks that can crash silently on Windows/CUDA.
    model.clear_callback("on_train_start")
    # FIX: Windows中文路径BUG - 必须用绝对路径并强制转换
    model.train(
        data=str(data_yaml.absolute()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        project=str(project_dir.absolute()),
        name=args.name,
        seed=args.seed,
        cache='disk',
        close_mosaic=args.close_mosaic,
        pretrained=True,
        plots=False,
        exist_ok=True,  # 修复路径重复拼接BUG
    )


if __name__ == "__main__":
    main()
