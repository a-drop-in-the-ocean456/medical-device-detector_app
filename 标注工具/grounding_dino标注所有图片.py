from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from video_annotator_cli import (
    GroundingDinoOpenSetDetector,
    bbox_to_yolo,
    clamp_bbox,
    nms_by_class,
    normalize_detector_label,
    parse_class_names,
    write_image_file,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_root: Path) -> List[Path]:
    # 递归收集 `input_root` 下的所有原始图片。
    # 这里明确跳过 `label` 目录中的标签图，避免把分割标签/标注结果
    # 当成需要再次检测的输入图片。
    image_paths: List[Path] = []
    for path in sorted(input_root.rglob("*")):
        # `rglob("*")` 会返回文件和目录，这里只保留真实文件。
        if not path.is_file():
            continue

        # 只处理常见图片格式，其他文件（如 txt/json/zip）直接跳过。
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        # 转成相对路径后再判断，避免绝对路径中其他位置恰好包含 `label`
        # 这个字符串，导致误跳过无关文件。
        rel_parts = path.relative_to(input_root).parts

        # 如果图片位于任意层级的 `label` 文件夹下，就视为标签图片跳过。
        # 例如:
        # - train_dataset/a/label/0001.png
        # - train_dataset/b/c/label/0002.jpg
        # 这些都不会加入待标注列表。
        if "label" in rel_parts:
            continue

        # 其余图片都认为是待 Grounding DINO 标注的原图。
        image_paths.append(path)
    return image_paths


def draw_detections(frame, detections: List[Dict[str, Any]]):
    canvas = frame.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        score = float(det.get("score", 0.0))
        label = str(det.get("label", "object"))
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            canvas,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    return canvas


def write_classes_file(output_root: Path, class_names: List[str]) -> None:
    classes_path = output_root / "classes.txt"
    with classes_path.open("w", encoding="utf-8") as f:
        for class_name in class_names:
            f.write(class_name + "\n")


def write_meta_file(output_root: Path, meta: Dict[str, Any]) -> None:
    meta_path = output_root / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_meta(
    *,
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    class_names: List[str],
    requested_images: int,
    detector_device: str,
    processed: int,
    skipped_existing: int,
    empty_images: int,
    total_boxes: int,
    copied_images: int,
    saved_visualizations: int,
    unreadable_images: List[str],
    status: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_id": args.model_id,
        "device": detector_device,
        "classes": class_names,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "nms_iou": float(args.nms_iou),
        "max_per_class": int(args.max_per_class),
        "save_images": bool(args.save_images),
        "save_visualizations": bool(args.save_visualizations),
        "skip_existing": bool(args.skip_existing),
        "requested_images": requested_images,
        "processed_images": processed,
        "skipped_existing": skipped_existing,
        "empty_images": empty_images,
        "total_boxes": total_boxes,
        "copied_images": copied_images,
        "saved_visualizations": saved_visualizations,
        "unreadable_images": unreadable_images,
    }


def write_yolo_label(
    image_shape,
    detections: List[Dict[str, Any]],
    class_to_id: Dict[str, int],
    label_path: Path,
) -> int:
    image_h, image_w = image_shape[:2]
    lines: List[str] = []
    for item in detections:
        label = str(item["label"])
        if label not in class_to_id:
            continue
        x, y, w, h = item["bbox"]
        xc, yc, wn, hn = bbox_to_yolo((x, y, w, h), image_w, image_h)
        cls_id = class_to_id[label]
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
    return len(lines)


def annotate_image(
    detector: GroundingDinoOpenSetDetector,
    frame,
    class_names: List[str],
    nms_iou: float,
    max_per_class: int,
) -> List[Dict[str, Any]]:
    image_h, image_w = frame.shape[:2]
    raw_detections = detector.detect(frame, class_names)
    detections: List[Dict[str, Any]] = []
    for det in raw_detections:
        label = normalize_detector_label(str(det.get("label_text", "")), class_names)
        if label is None:
            continue
        x1, y1, x2, y2 = det["bbox_xyxy"]
        bbox = clamp_bbox(
            (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)),
            image_w,
            image_h,
        )
        detections.append(
            {
                "label": label,
                "bbox": bbox,
                "score": float(det.get("score", 0.0)),
            }
        )
    return nms_by_class(
        detections,
        nms_iou=float(nms_iou),
        max_per_class=int(max_per_class),
    )


def process_dataset(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.classes)
    if not class_names:
        raise RuntimeError("No valid classes parsed from --classes.")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    image_paths = collect_images(input_root)
    if args.limit is not None:
        image_paths = image_paths[: max(0, int(args.limit))]
    if not image_paths:
        raise RuntimeError(f"No images found under: {input_root}")

    labels_dir = output_root / "labels"
    images_dir = output_root / "images"
    vis_dir = output_root / "visualizations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 先把类别文件写出来，这样即使中途停止，输出目录里也有完整的类别定义。
    write_classes_file(output_root, class_names)

    detector = GroundingDinoOpenSetDetector(
        model_id=args.model_id,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        device=args.device,
    )

    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Images to process: {len(image_paths)}")
    print(f"[INFO] Classes: {class_names}")

    # 程序启动后立即写一版 meta，后续循环中持续刷新进度与统计信息。
    write_meta_file(
        output_root,
        build_meta(
            args=args,
            input_root=input_root,
            output_root=output_root,
            class_names=class_names,
            requested_images=len(image_paths),
            detector_device=detector.device,
            processed=0,
            skipped_existing=0,
            empty_images=0,
            total_boxes=0,
            copied_images=0,
            saved_visualizations=0,
            unreadable_images=[],
            status="running",
        ),
    )

    iterator = image_paths
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            image_paths,
            total=len(image_paths),
            desc="GroundingDINO",
            unit="img",
            dynamic_ncols=True,
        )
        iterator = progress_bar

    processed = 0
    total_boxes = 0
    empty_images = 0
    copied_images = 0
    saved_visualizations = 0
    unreadable_images: List[str] = []
    skipped_existing = 0

    for image_path in iterator:
        relative_path = image_path.relative_to(input_root)
        label_path = (labels_dir / relative_path).with_suffix(".txt")

        if args.skip_existing and label_path.exists():
            skipped_existing += 1
            write_meta_file(
                output_root,
                build_meta(
                    args=args,
                    input_root=input_root,
                    output_root=output_root,
                    class_names=class_names,
                    requested_images=len(image_paths),
                    detector_device=detector.device,
                    processed=processed,
                    skipped_existing=skipped_existing,
                    empty_images=empty_images,
                    total_boxes=total_boxes,
                    copied_images=copied_images,
                    saved_visualizations=saved_visualizations,
                    unreadable_images=unreadable_images,
                    status="running",
                ),
            )
            continue

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            unreadable_images.append(str(image_path))
            write_meta_file(
                output_root,
                build_meta(
                    args=args,
                    input_root=input_root,
                    output_root=output_root,
                    class_names=class_names,
                    requested_images=len(image_paths),
                    detector_device=detector.device,
                    processed=processed,
                    skipped_existing=skipped_existing,
                    empty_images=empty_images,
                    total_boxes=total_boxes,
                    copied_images=copied_images,
                    saved_visualizations=saved_visualizations,
                    unreadable_images=unreadable_images,
                    status="running",
                ),
            )
            continue

        detections = annotate_image(
            detector=detector,
            frame=frame,
            class_names=class_names,
            nms_iou=float(args.nms_iou),
            max_per_class=int(args.max_per_class),
        )

        line_count = write_yolo_label(frame.shape, detections, class_to_id, label_path)
        total_boxes += line_count
        if line_count == 0:
            empty_images += 1

        if args.save_images:
            image_out_path = images_dir / relative_path
            write_image_file(image_out_path, frame)
            copied_images += 1

        if args.save_visualizations:
            vis_frame = draw_detections(frame, detections)
            vis_out_path = (vis_dir / relative_path).with_suffix(".jpg")
            write_image_file(vis_out_path, vis_frame)
            saved_visualizations += 1

        processed += 1
        write_meta_file(
            output_root,
            build_meta(
                args=args,
                input_root=input_root,
                output_root=output_root,
                class_names=class_names,
                requested_images=len(image_paths),
                detector_device=detector.device,
                processed=processed,
                skipped_existing=skipped_existing,
                empty_images=empty_images,
                total_boxes=total_boxes,
                copied_images=copied_images,
                saved_visualizations=saved_visualizations,
                unreadable_images=unreadable_images,
                status="running",
            ),
        )
        if progress_bar is not None:
            progress_bar.set_postfix({"boxes": line_count, "empty": empty_images})
        elif processed % 50 == 0:
            print(
                f"[INFO] processed={processed} total_boxes={total_boxes} empty={empty_images}"
            )

    if progress_bar is not None:
        progress_bar.close()

    write_meta_file(
        output_root,
        build_meta(
            args=args,
            input_root=input_root,
            output_root=output_root,
            class_names=class_names,
            requested_images=len(image_paths),
            detector_device=detector.device,
            processed=processed,
            skipped_existing=skipped_existing,
            empty_images=empty_images,
            total_boxes=total_boxes,
            copied_images=copied_images,
            saved_visualizations=saved_visualizations,
            unreadable_images=unreadable_images,
            status="done",
        ),
    )

    print("[INFO] Done.")
    print(f"[INFO] Processed images: {processed}")
    print(f"[INFO] Total boxes: {total_boxes}")
    print(f"[INFO] Empty images: {empty_images}")
    print(f"[INFO] Skipped existing: {skipped_existing}")
    print(f"[INFO] Unreadable images: {len(unreadable_images)}")
    print(f"[INFO] Meta: {(output_root / 'meta.json')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grounding DINO annotate all images under a folder recursively."
    )
    parser.add_argument(
        "--input-root",
        default="/home/liang/grass_data_fixed/train_dataset",
        help="Root folder to scan recursively for images.",
    )
    parser.add_argument(
        "--output-dir",
        default="grounding_dino_训练集标注结果",
        help="Output folder for YOLO labels and optional image copies.",
    )
    parser.add_argument(
        "--classes",
        default="fallen leaf, fallen leaves, stone, obstacle",
        help="Comma-separated prompt classes for Grounding DINO.",
    )
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N images.")
    parser.add_argument(
        "--save-images",
        dest="save_images",
        action="store_true",
        default=True,
        help="Copy original images into output/images.",
    )
    parser.add_argument(
        "--save-visualizations",
        dest="save_visualizations",
        action="store_true",
        default=True,
        help="Save boxed preview images into output/visualizations.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output label txt already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
