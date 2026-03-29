from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
        path = path.with_suffix(ext)
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode output image: {path}")
    encoded.tofile(str(path))


def load_classes(classes_file: Optional[Path]) -> Dict[int, str]:
    if classes_file is None or not classes_file.exists():
        return {}
    classes: Dict[int, str] = {}
    with classes_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                classes[idx] = name
    return classes


def class_color(cls_id: int) -> Tuple[int, int, int]:
    # Deterministic class color.
    hue = int((cls_id * 37) % 180)
    color = cv2.cvtColor(
        np.uint8([[[hue, 220, 255]]]),
        cv2.COLOR_HSV2BGR,
    )[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def yolo_to_xyxy(
    xc: float, yc: float, w: float, h: float, image_w: int, image_h: int
) -> Tuple[int, int, int, int]:
    bw = w * image_w
    bh = h * image_h
    x1 = int(round((xc * image_w) - bw / 2))
    y1 = int(round((yc * image_h) - bh / 2))
    x2 = int(round(x1 + bw))
    y2 = int(round(y1 + bh))
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(x1 + 1, min(x2, image_w))
    y2 = max(y1 + 1, min(y2, image_h))
    return x1, y1, x2, y2


def draw_label(
    image,
    cls_id: int,
    cls_name: str,
    box: Tuple[int, int, int, int],
    thickness: int,
    font_scale: float,
) -> None:
    x1, y1, x2, y2 = box
    color = class_color(cls_id)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    text = f"{cls_id}:{cls_name}" if cls_name else str(cls_id)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    ty1 = max(0, y1 - th - baseline - 4)
    ty2 = ty1 + th + baseline + 4
    tx2 = min(image.shape[1], x1 + tw + 8)
    cv2.rectangle(image, (x1, ty1), (tx2, ty2), color, -1)
    cv2.putText(
        image,
        text,
        (x1 + 4, ty2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def parse_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    records: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            records.append((cls_id, xc, yc, w, h))
    return records


def collect_images(images_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels for all images in a folder and save results."
    )
    parser.add_argument("--images-dir", 
                        default="一次性医疗用品—2ml注射器syringe/video/images", 
                        help="Input images folder.")
    parser.add_argument("--labels-dir", 
                        default="一次性医疗用品—2ml注射器syringe/video/labels", 
                        help="Input YOLO labels folder (.txt).")
    parser.add_argument("--output-dir", 
                        default="一次性医疗用品—2ml注射器syringe/video/images_yolo_vis", 
                        help="Output visualization folder.")
    parser.add_argument(
        "--classes-file",
        default="syringe",
        help="Optional classes.txt path. If omitted, class id will be shown.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively and keep relative output structure.",
    )
    parser.add_argument(
        "--save-no-label",
        action="store_true",
        help="Also save images even if corresponding label file does not exist.",
    )
    parser.add_argument("--thickness", type=int, default=2, help="Box line thickness.")
    parser.add_argument("--font-scale", type=float, default=0.5, help="Text font scale.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    classes_file = Path(args.classes_file) if args.classes_file else None

    if not images_dir.exists():
        raise RuntimeError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise RuntimeError(f"Labels dir not found: {labels_dir}")

    class_names = load_classes(classes_file)
    image_paths = collect_images(images_dir, recursive=bool(args.recursive))
    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    saved = 0
    skipped_no_label = 0
    skipped_bad_image = 0
    total_boxes = 0

    for image_path in image_paths:
        if args.recursive:
            rel = image_path.relative_to(images_dir)
            label_path = labels_dir / rel.with_suffix(".txt")
            output_path = output_dir / rel
        else:
            label_path = labels_dir / f"{image_path.stem}.txt"
            output_path = output_dir / image_path.name

        image = read_image_unicode(image_path)
        processed += 1
        if image is None:
            skipped_bad_image += 1
            continue

        if not label_path.exists():
            if args.save_no_label:
                write_image_unicode(output_path, image)
                saved += 1
            else:
                skipped_no_label += 1
            continue

        h, w = image.shape[:2]
        records = parse_label_file(label_path)
        for cls_id, xc, yc, bw, bh in records:
            box = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            cls_name = class_names.get(cls_id, "")
            draw_label(
                image=image,
                cls_id=cls_id,
                cls_name=cls_name,
                box=box,
                thickness=max(1, int(args.thickness)),
                font_scale=max(0.2, float(args.font_scale)),
            )
            total_boxes += 1

        write_image_unicode(output_path, image)
        saved += 1

        if processed % 100 == 0:
            print(f"[INFO] processed={processed}/{len(image_paths)} saved={saved}")

    print("[DONE] Visualization finished.")
    print(f"[INFO] images_total={len(image_paths)}")
    print(f"[INFO] processed={processed}")
    print(f"[INFO] saved={saved}")
    print(f"[INFO] total_boxes={total_boxes}")
    print(f"[INFO] skipped_no_label={skipped_no_label}")
    print(f"[INFO] skipped_bad_image={skipped_bad_image}")
    print(f"[INFO] output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
