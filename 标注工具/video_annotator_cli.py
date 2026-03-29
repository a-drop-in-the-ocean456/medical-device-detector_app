from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}


def collect_videos(video_args: Optional[List[str]]) -> List[Path]:
    videos: List[Path] = []
    if video_args:
        for item in video_args:
            parts = [x.strip() for x in str(item).replace("，", ",").split(",")]
            for part in parts:
                if not part:
                    continue
                p = Path(part)
                if p.exists() and p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    videos.append(p)
    dedup: List[Path] = []
    seen = set()
    for v in videos:
        key = str(v.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(v)
    dedup.sort()
    return dedup


def sanitize_name(name: str) -> str:
    banned = '<>:"/\\|?*'
    out = "".join("_" if ch in banned else ch for ch in name.strip())
    out = out.strip(" .")
    return out or "video"


def unique_subdir(base_dir: Path, base_name: str, used: set[str]) -> Path:
    stem = sanitize_name(base_name)
    candidate = stem
    idx = 2
    while candidate in used:
        candidate = f"{stem}_{idx}"
        idx += 1
    used.add(candidate)
    return base_dir / candidate


def clamp_bbox(
    bbox: Tuple[float, float, float, float], image_w: int, image_h: int
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    x = max(0.0, min(float(x), image_w - 1.0))
    y = max(0.0, min(float(y), image_h - 1.0))
    w = max(1.0, min(float(w), image_w - x))
    h = max(1.0, min(float(h), image_h - y))
    return x, y, w, h


def bbox_to_yolo(
    bbox: Tuple[float, float, float, float], image_w: int, image_h: int
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / image_w
    yc = (y + h / 2.0) / image_h
    return xc, yc, w / image_w, h / image_h


def write_image_file(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
        ext = ".jpg"
    ok, encoded = cv2.imencode(ext, frame)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    with path.open("wb") as f:
        f.write(encoded.tobytes())


class GroundingDinoOpenSetDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        def _fix_hint() -> str:
            return (
                "Fix (CPU example):\n"
                "python -m pip uninstall -y torch torchvision\n"
                "python -m pip install --upgrade --force-reinstall "
                "torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            )

        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "Grounding DINO mode requires PyTorch.\n"
                f"{_fix_hint()}\nOriginal error: {exc}"
            ) from exc

        try:
            import torchvision

            _ = torchvision.ops.nms
        except Exception as exc:
            raise RuntimeError(
                "Torchvision is missing or incompatible with torch "
                "(common error: 'operator torchvision::nms does not exist').\n"
                f"{_fix_hint()}\nOriginal error: {exc}"
            ) from exc

        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError(
                "Grounding DINO mode requires pillow (PIL).\n"
                "Install with: python -m pip install pillow\n"
                f"Original error: {exc}"
            ) from exc

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Failed to import transformers Grounding DINO modules.\n"
                "This is usually caused by incompatible torch/torchvision build.\n"
                f"{_fix_hint()}\nOriginal error: {exc}"
            ) from exc

        self.torch = torch
        self.Image = Image
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Grounding DINO model '{model_id}'.\n"
                "Check internet access, model id, and huggingface cache permissions.\n"
                f"Original error: {exc}"
            ) from exc
        self._postprocess_params = set(
            inspect.signature(
                self.processor.post_process_grounded_object_detection
            ).parameters.keys()
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.model_id = model_id

    def detect(
        self,
        frame_bgr,
        class_names: List[str],
    ) -> List[Dict[str, object]]:
        if not class_names:
            return []

        prompt = ". ".join(class_names) + "."
        image = self.Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([[image.height, image.width]])
        post_kwargs: Dict[str, Any] = {"outputs": outputs}
        if "input_ids" in self._postprocess_params and "input_ids" in inputs:
            post_kwargs["input_ids"] = inputs["input_ids"]
        if "target_sizes" in self._postprocess_params:
            post_kwargs["target_sizes"] = target_sizes
        if "box_threshold" in self._postprocess_params:
            post_kwargs["box_threshold"] = self.box_threshold
        elif "threshold" in self._postprocess_params:
            post_kwargs["threshold"] = self.box_threshold
        elif "score_threshold" in self._postprocess_params:
            post_kwargs["score_threshold"] = self.box_threshold
        if "text_threshold" in self._postprocess_params:
            post_kwargs["text_threshold"] = self.text_threshold

        results = self.processor.post_process_grounded_object_detection(**post_kwargs)[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("text_labels")
        if labels is None:
            labels = results.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        boxes_list = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else list(boxes)
        scores_list = (
            scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
        )

        normalized: List[Dict[str, object]] = []
        for idx, box in enumerate(boxes_list):
            if idx >= len(scores_list):
                break
            raw_label = labels[idx]
            if hasattr(raw_label, "item"):
                raw_label = raw_label.item()
            label_str = str(raw_label).strip()
            score = float(scores_list[idx])
            if len(box) != 4:
                continue
            normalized.append(
                {
                    "label_text": label_str,
                    "bbox_xyxy": (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    "score": score,
                }
            )
        return normalized


class ByteTrackDetections:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        xyxy = np.asarray(xyxy, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        cls = np.asarray(cls, dtype=np.float32)

        if xyxy.size == 0:
            self.xyxy = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros((0,), dtype=np.float32)
            self.cls = np.zeros((0,), dtype=np.float32)
            return

        if xyxy.ndim == 1:
            xyxy = xyxy.reshape(1, -1)
        if xyxy.shape[1] != 4:
            raise ValueError(f"Expected xyxy shape (N,4), got {xyxy.shape}")

        self.xyxy = xyxy
        self.conf = conf.reshape(-1)
        self.cls = cls.reshape(-1)

    @classmethod
    def empty(cls) -> "ByteTrackDetections":
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.float32),
        )

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, item) -> "ByteTrackDetections":
        return ByteTrackDetections(self.xyxy[item], self.conf[item], self.cls[item])

    @property
    def xywh(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1y1 = self.xyxy[:, :2]
        x2y2 = self.xyxy[:, 2:]
        wh = x2y2 - x1y1
        ctr = x1y1 + wh * 0.5
        return np.concatenate([ctr, wh], axis=1).astype(np.float32)


def load_bytetrack_class():
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker
    except Exception:
        local_ultralytics = Path(__file__).resolve().parent / "ultralytics-main"
        if local_ultralytics.exists():
            local_ultralytics_str = str(local_ultralytics)
            if local_ultralytics_str not in sys.path:
                sys.path.insert(0, local_ultralytics_str)
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            # Verify version compatibility
            try:
                import ultralytics
                ver = ultralytics.__version__
                print(f"[INFO] Using local ultralytics version: {ver}")
            except ImportError:
                print("[WARN] Could not determine ultralytics version")
            return BYTETracker
        except Exception as e:
            raise RuntimeError(
                "Failed to load BYTETracker from both system and local ultralytics. "
                "Please ensure ultralytics is installed: pip install ultralytics"
            ) from e


def normalize_detector_label(raw_text: str, class_names: List[str]) -> Optional[str]:
    text = raw_text.strip().lower()
    if not text:
        return None

    lower_classes = [c.lower() for c in class_names]
    for idx, c in enumerate(lower_classes):
        if text == c:
            return class_names[idx]

    matches: List[Tuple[int, int]] = []
    for idx, c in enumerate(lower_classes):
        if c in text or text in c:
            matches.append((idx, len(c)))
    if not matches:
        return None
    matches.sort(key=lambda x: x[1], reverse=True)
    return class_names[matches[0][0]]


def nms_by_class(
    detections: List[Dict[str, object]], nms_iou: float = 0.5, max_per_class: int = 20
) -> List[Dict[str, object]]:
    if not detections:
        return []

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for det in detections:
        label = str(det.get("label", "")).strip()
        if not label:
            continue
        grouped.setdefault(label, []).append(det)

    kept: List[Dict[str, object]] = []
    for label, group in grouped.items():
        boxes = []
        scores = []
        for det in group:
            x, y, w, h = det["bbox"]
            boxes.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
            scores.append(float(det.get("score", 0.0)))
        if not boxes:
            continue

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=0.0,
            nms_threshold=float(nms_iou),
        )
        if indices is None:
            continue

        flat_indices: List[int] = []
        for raw in indices:
            if isinstance(raw, (list, tuple, np.ndarray)):
                if len(raw) > 0:
                    flat_indices.append(int(raw[0]))
            else:
                flat_indices.append(int(raw))

        flat_indices = sorted(
            set(flat_indices),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:max_per_class]

        for idx in flat_indices:
            kept.append(group[idx])
    return kept


def build_bytetrack_results(
    detections: List[Dict[str, object]], class_to_id: Dict[str, int]
) -> ByteTrackDetections:
    if not detections:
        return ByteTrackDetections.empty()

    xyxy_list: List[List[float]] = []
    conf_list: List[float] = []
    cls_list: List[float] = []
    for det in detections:
        label = str(det.get("label", "")).strip()
        if label not in class_to_id:
            continue
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        score = float(det.get("score", 0.0))
        if w <= 0 or h <= 0:
            continue
        xyxy_list.append([x, y, x + w, y + h])
        conf_list.append(score)
        cls_list.append(float(class_to_id[label]))

    if not xyxy_list:
        return ByteTrackDetections.empty()
    return ByteTrackDetections(
        xyxy=np.asarray(xyxy_list, dtype=np.float32),
        conf=np.asarray(conf_list, dtype=np.float32),
        cls=np.asarray(cls_list, dtype=np.float32),
    )


def tracks_to_frame_boxes(
    tracks_np: np.ndarray, class_names: List[str], image_w: int, image_h: int
) -> List[Dict[str, object]]:
    frame_boxes: List[Dict[str, object]] = []
    if tracks_np is None or len(tracks_np) == 0:
        return frame_boxes

    for row in tracks_np:
        if len(row) < 7:
            continue
        x1, y1, x2, y2 = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        score = float(row[5])
        cls_id = int(round(float(row[6])))
        if not (0 <= cls_id < len(class_names)):
            continue
        bbox = clamp_bbox((x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)), image_w, image_h)
        frame_boxes.append(
            {
                "label": class_names[cls_id],
                "bbox": bbox,
                "score": score,
            }
        )
    return frame_boxes


def write_one_frame(
    frame,
    frame_idx: int,
    boxes: List[Dict[str, object]],
    class_to_id: Dict[str, int],
    labels_dir: Path,
    images_dir: Path,
    video_stem: str,
    save_images: bool,
) -> int:
    image_h, image_w = frame.shape[:2]
    name = f"{video_stem}_{frame_idx:06d}"
    label_path = labels_dir / f"{name}.txt"

    lines: List[str] = []
    for item in boxes:
        label = str(item["label"])
        if label not in class_to_id:
            continue
        x, y, w, h = item["bbox"]
        xc, yc, wn, hn = bbox_to_yolo((x, y, w, h), image_w, image_h)
        cls_id = class_to_id[label]
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    with label_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))

    if save_images:
        image_path = images_dir / f"{name}.jpg"
        write_image_file(image_path, frame)

    return len(lines)


def parse_class_names(classes_arg: str) -> List[str]:
    parts = [x.strip() for x in classes_arg.split(",")]
    cleaned: List[str] = []
    for c in parts:
        if not c:
            continue
        if c not in cleaned:
            cleaned.append(c)
    return cleaned


def run_single_video(
    args: argparse.Namespace,
    detector: GroundingDinoOpenSetDetector,
    BYTETracker,
    class_names: List[str],
    class_to_id: Dict[str, int],
    video_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 240:
        print(f"[WARN] Invalid video FPS {video_fps}, defaulting to 30")
        video_fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    start_frame = max(0, int(args.start_frame))
    if start_frame >= total_frames:
        cap.release()
        raise RuntimeError(
            f"--start-frame out of range for {video_path.name}: {start_frame}, total={total_frames}"
        )

    if int(args.end_frame) >= 0:
        end_frame = min(int(args.end_frame), total_frames - 1)
    else:
        end_frame = total_frames - 1
    if end_frame < start_frame:
        cap.release()
        raise RuntimeError(f"--end-frame must be >= --start-frame for {video_path.name}")

    detect_interval = max(1, int(args.detect_interval))
    tracker = BYTETracker(
        args=SimpleNamespace(
            track_high_thresh=float(args.track_high_thresh),
            track_low_thresh=float(args.track_low_thresh),
            new_track_thresh=float(args.new_track_thresh),
            match_thresh=float(args.match_thresh),
            track_buffer=int(args.track_buffer),
            fuse_score=bool(args.fuse_score),
        ),
        frame_rate=int(video_fps),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    video_stem = video_path.stem
    processed = 0
    saved_labels = 0
    saved_images = 0
    max_boxes = 0

    print(
        f"[INFO] Start BYTE_TRACK annotation | video={video_path} | "
        f"frames={start_frame}-{end_frame} | classes={class_names}"
    )

    total_target = end_frame - start_frame + 1
    frame_indices = range(start_frame, end_frame + 1)
    progress_bar = None
    iterator = frame_indices
    if tqdm is not None:
        progress_bar = tqdm(
            frame_indices,
            total=total_target,
            desc=f"{video_path.stem}",
            unit="frame",
            dynamic_ncols=True,
        )
        iterator = progress_bar

    for frame_idx in iterator:
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Stop early at frame {frame_idx} (read failed).")
            break
        image_h, image_w = frame.shape[:2]
        if image_w <= 0 or image_h <= 0:
            continue

        det_candidates: List[Dict[str, object]] = []
        should_detect = (frame_idx - start_frame) % detect_interval == 0
        if should_detect:
            detections = detector.detect(frame, class_names)
            for det in detections:
                label = normalize_detector_label(str(det.get("label_text", "")), class_names)
                if label is None:
                    continue
                x1, y1, x2, y2 = det["bbox_xyxy"]
                bbox = clamp_bbox(
                    (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)),
                    image_w,
                    image_h,
                )
                det_candidates.append(
                    {
                        "label": label,
                        "bbox": bbox,
                        "score": float(det.get("score", 0.0)),
                    }
                )
            det_candidates = nms_by_class(
                det_candidates,
                nms_iou=float(args.nms_iou),
                max_per_class=int(args.max_per_class),
            )

        bt_results = build_bytetrack_results(det_candidates, class_to_id)
        tracks_np = tracker.update(bt_results, frame)
        frame_boxes = tracks_to_frame_boxes(tracks_np, class_names, image_w, image_h)

        saved_labels += write_one_frame(
            frame=frame,
            frame_idx=frame_idx,
            boxes=frame_boxes,
            class_to_id=class_to_id,
            labels_dir=labels_dir,
            images_dir=images_dir,
            video_stem=video_stem,
            save_images=bool(args.save_images),
        )
        if args.save_images:
            saved_images += 1
        processed += 1
        max_boxes = max(max_boxes, len(frame_boxes))
        if progress_bar is not None:
            progress_bar.set_postfix({"boxes": len(frame_boxes), "lines": saved_labels})
        elif processed % 10 == 0 or frame_idx == end_frame:
            print(
                f"[INFO] frame={frame_idx} processed={processed} "
                f"boxes={len(frame_boxes)}"
            )

    if progress_bar is not None:
        progress_bar.close()
    cap.release()

    with (output_dir / "classes.txt").open("w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    meta = {
        "video_path": str(video_path.resolve()),
        "mode": "BYTE_TRACK",
        "model_id": args.model_id,
        "device": detector.device,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "detect_interval": detect_interval,
        "track_high_thresh": float(args.track_high_thresh),
        "track_low_thresh": float(args.track_low_thresh),
        "new_track_thresh": float(args.new_track_thresh),
        "match_thresh": float(args.match_thresh),
        "track_buffer": int(args.track_buffer),
        "fuse_score": bool(args.fuse_score),
        "nms_iou": float(args.nms_iou),
        "max_per_class": int(args.max_per_class),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "processed_frames": processed,
        "classes": class_names,
        "save_images": bool(args.save_images),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")
    print(f"[INFO] Output: {output_dir.resolve()}")
    print(f"[INFO] Processed frames: {processed}")
    print(f"[INFO] Max objects/frame: {max_boxes}")
    print(f"[INFO] Saved yolo lines: {saved_labels}")
    print(f"[INFO] Saved images: {saved_images}")

    return {
        "video": str(video_path),
        "output_dir": str(output_dir.resolve()),
        "processed_frames": processed,
        "max_boxes": max_boxes,
        "saved_labels": saved_labels,
        "saved_images": saved_images,
    }


def run(args: argparse.Namespace) -> None:
    class_names = parse_class_names(args.classes)
    if not class_names:
        raise RuntimeError("No valid classes parsed from --classes.")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    videos = collect_videos(args.video)
    if not videos:
        raise RuntimeError("No valid input videos found. Use --video with one or more video files.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(videos)} video(s).")

    # Load detection model after validating videos to avoid wasted time
    detector = GroundingDinoOpenSetDetector(
        model_id=args.model_id,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        device=args.device,
    )
    BYTETracker = load_bytetrack_class()
    summaries: List[Dict[str, Any]] = []
    used_output_names: set[str] = set()

    for idx, video_path in enumerate(videos, 1):
        print(f"[INFO] ({idx}/{len(videos)}) Processing: {video_path}")
        if len(videos) == 1 and not args.force_video_subdir:
            video_output = output_root
        else:
            video_output = unique_subdir(output_root, video_path.stem, used_output_names)
        summary = run_single_video(
            args=args,
            detector=detector,
            BYTETracker=BYTETracker,
            class_names=class_names,
            class_to_id=class_to_id,
            video_path=video_path,
            output_dir=video_output,
        )
        summaries.append(summary)

    batch_meta = {
        "mode": "BYTE_TRACK_BATCH",
        "video_count": len(summaries),
        "videos": summaries,
        "classes": class_names,
        "model_id": args.model_id,
        "device": detector.device,
    }
    with (output_root / "batch_meta.json").open("w", encoding="utf-8") as f:
        json.dump(batch_meta, f, ensure_ascii=False, indent=2)

    total_frames = sum(int(x["processed_frames"]) for x in summaries)
    print("[INFO] Batch done.")
    print(f"[INFO] Total videos: {len(summaries)}")
    print(f"[INFO] Total processed frames: {total_frames}")
    print(f"[INFO] Batch meta: {(output_root / 'batch_meta.json').resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "No-GUI video auto annotation using Grounding DINO + ByteTrack. "
            "Supports one or multiple videos."
        )
    )
    parser.add_argument(
        "--video",
        nargs="+",
        default=[
            # "医疗废物—感染性废物waste/video/1.mp4",
            #      "医疗废物—感染性废物waste/video/2.mp4",
                 "一次性医疗用品—2ml注射器syringe/video/1.mp4",
                 "一次性医疗用品—2ml注射器syringe/video/2.mp4",
                 "一次性医疗用品—2ml注射器syringe/video/3.mp4",
                #  "医疗废物—感染性废物waste/video/6.mp4",
                #  "医疗废物—感染性废物waste/video/7.mp4",
                #  "医疗废物—感染性废物waste/video/8.mp4"
                 ],
        help=(
            "Input one or multiple videos, supports space or comma separator. "
            "Example: --video a.mp4 b.mp4 or --video a.mp4,b.mp4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="一次性医疗用品—2ml注射器syringe/video/",
        help="Output root directory (per-video subfolders in batch mode).",
    )
    parser.add_argument(
        "--force-video-subdir",
        action="store_true",
        help="Always create a per-video output subfolder (even for single video).",
    )
    parser.add_argument(
        "--classes",
        default="syringe, a white paper with red text",
        help="Comma-separated class names, e.g. road_crack,baozi",
    )
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--detect-interval", type=int, default=1)

    parser.add_argument("--track-high-thresh", type=float, default=0.35)
    parser.add_argument("--track-low-thresh", type=float, default=0.05)
    parser.add_argument("--new-track-thresh", type=float, default=0.35)
    parser.add_argument("--match-thresh", type=float, default=0.80)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--fuse-score", dest="fuse_score", action="store_true", default=True)
    parser.add_argument("--no-fuse-score", dest="fuse_score", action="store_false")

    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-per-class", type=int, default=20)

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--save-images", dest="save_images", action="store_true", default=True)
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    return parser
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
