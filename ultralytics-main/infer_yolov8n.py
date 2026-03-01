#!/usr/bin/env python3
"""
Run YOLOv8n inference on all test images and save visualized predictions.

Examples:
    python infer_yolov8n.py
    python infer_yolov8n.py --conf 0.35 --device 0 --name yolov8n_full_test_pred
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Infer on test set and save images with predicted boxes.")
    parser.add_argument("--data", type=Path, default=root / "qc_dataset.yaml", help="Dataset YAML path.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "runs" / "detect" / "yolov8n_full" / "weights" / "best.pt",
        help="Trained model weights path.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device like 0, or cpu.")
    parser.add_argument("--project", type=Path, default=root / "runs" / "detect", help="Output project dir.")
    parser.add_argument("--name", type=str, default="yolov8n_full_test_pred", help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("--save-txt", action="store_true", help="Also save txt labels for predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence with txt labels.")
    return parser.parse_args()


def to_abs(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def load_test_sources(data_yaml: Path) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid dataset yaml format: {data_yaml}")

    if "test" not in cfg:
        raise KeyError(f"`test` entry not found in dataset yaml: {data_yaml}")

    dataset_root = to_abs(Path(cfg.get("path", ".")), data_yaml.parent)
    test_value = cfg["test"]

    if isinstance(test_value, (str, Path)):
        return [to_abs(Path(test_value), dataset_root)]
    if isinstance(test_value, list):
        return [to_abs(Path(x), dataset_root) for x in test_value]

    raise TypeError(f"Unsupported `test` value type in yaml: {type(test_value)}")


def count_images(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_dir():
            total += sum(1 for p in path.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
        elif path.is_file():
            if path.suffix.lower() in IMG_SUFFIXES:
                total += 1
            elif path.suffix.lower() == ".txt":
                lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
                total += sum(1 for x in lines if x)
    return total


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    data_yaml = to_abs(args.data, root)
    weights = to_abs(args.weights, root)
    project_dir = to_abs(args.project, root)
    project_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    sources = load_test_sources(data_yaml)
    missing = [str(x) for x in sources if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Test source path(s) not found: {missing}")

    source_arg: str | list[str]
    source_arg = [str(x) for x in sources]
    if len(source_arg) == 1:
        source_arg = source_arg[0]

    print("Inference config:")
    print(f"  weights : {weights}")
    print(f"  data    : {data_yaml}")
    print(f"  source  : {sources}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  conf    : {args.conf}")
    print(f"  iou     : {args.iou}")
    print(f"  device  : {args.device}")
    print(f"  project : {project_dir}")
    print(f"  name    : {args.name}")
    print(f"  images  : {count_images(sources)}")

    model = YOLO(str(weights))
    results = model.predict(
        source=source_arg,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    save_dir = project_dir / args.name
    print(f"Done. Predicted {len(results)} image(s).")
    print(f"Saved visualized results to: {save_dir}")


if __name__ == "__main__":
    main()
