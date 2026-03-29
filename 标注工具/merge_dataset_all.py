from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SKIP_LABEL_FILES = {"classes.txt"}


@dataclass
class SourceFolder:
    path: Path
    class_id: int
    prefix: str
    class_name: str


def iter_files(folder: Path, recursive: bool):
    if recursive:
        yield from folder.rglob("*")
    else:
        yield from folder.glob("*")


def extract_english_name(folder_name: str) -> str:
    matches = re.findall(r"[A-Za-z][A-Za-z0-9 ]*", folder_name)
    if not matches:
        return ""
    return matches[-1].strip()


def sanitize_prefix(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return normalized


def build_sources(dataset_dir: Path, output_dir_name: str) -> list[SourceFolder]:
    source_dirs = [
        p for p in dataset_dir.iterdir() if p.is_dir() and p.name != output_dir_name
    ]
    source_dirs.sort(key=lambda p: p.name)

    if len(source_dirs) != 5:
        raise SystemExit(
            f"Expected exactly 5 source folders in {dataset_dir}, found {len(source_dirs)}."
        )

    used_prefixes: set[str] = set()
    sources: list[SourceFolder] = []
    for class_id, folder in enumerate(source_dirs):
        english = extract_english_name(folder.name) or f"class_{class_id}"
        prefix = sanitize_prefix(english) or f"class_{class_id}"
        if prefix in used_prefixes:
            prefix = f"{prefix}_{class_id}"
        used_prefixes.add(prefix)
        sources.append(
            SourceFolder(
                path=folder,
                class_id=class_id,
                prefix=prefix,
                class_name=english,
            )
        )
    return sources


def collect_images(images_dir: Path, recursive: bool) -> list[Path]:
    return sorted(
        [
            p
            for p in iter_files(images_dir, recursive)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def collect_labels(labels_dir: Path, recursive: bool) -> list[Path]:
    return sorted(
        [
            p
            for p in iter_files(labels_dir, recursive)
            if p.is_file()
            and p.suffix.lower() == ".txt"
            and p.name.lower() not in SKIP_LABEL_FILES
        ]
    )


def file_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def build_target_name_map(
    image_files: list[Path],
    label_files: list[Path],
    images_dir: Path,
    labels_dir: Path,
    prefix: str,
    used_target_stems: set[str],
) -> dict[str, str]:
    keys = {file_key(p, images_dir) for p in image_files}
    keys.update(file_key(p, labels_dir) for p in label_files)

    mapping: dict[str, str] = {}
    for key in sorted(keys):
        raw_stem = Path(key).name
        candidate = f"{prefix}_{raw_stem}"
        unique_name = candidate
        count = 1
        while unique_name.lower() in used_target_stems:
            unique_name = f"{candidate}_{count}"
            count += 1
        used_target_stems.add(unique_name.lower())
        mapping[key] = unique_name
    return mapping


def remap_yolo_label_text(text: str, new_class_id: int) -> str:
    new_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            new_lines.append("")
            continue
        parts = stripped.split()
        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))
    return "\n".join(new_lines) + ("\n" if new_lines else "")


def clear_folder(folder: Path) -> int:
    removed = 0
    for p in folder.glob("*"):
        if p.is_file():
            p.unlink()
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        # 默认 help 增加默认值展示，便于命令行查看参数时快速确认当前配置
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "把 dataset 下的 5 个子文件夹合并到 dataset/all；"
            "复制 images/labels，并把每个来源文件夹的 YOLO 类别重映射到 0~4。"
        )
    )

    # 数据集根目录：脚本会在该目录下查找 5 个来源子目录，并在其中创建/使用输出目录。
    # 例如：dataset/类别A, dataset/类别B, ... , dataset/all
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help=(
            "数据集根目录。脚本会在这里读取 5 个来源文件夹，并把合并结果写入该目录下的输出文件夹。"
        ),
    )

    # 每个来源目录中视频数据的中间目录名。
    # 目录结构默认是：<来源目录>/video/images 和 <来源目录>/video/labels
    parser.add_argument(
        "--video-dir-name",
        default="video",
        help="来源目录中承载 images/labels 的父目录名。",
    )

    # 图片目录名：在来源目录下用于读取图片，在输出目录下用于写入合并后的图片。
    parser.add_argument(
        "--images-dir-name",
        default="images",
        help="图片目录名（输入与输出都使用该名称）。",
    )

    # 标签目录名：在来源目录下用于读取 YOLO 标签，在输出目录下用于写入重编号后的标签。
    parser.add_argument(
        "--labels-dir-name",
        default="labels",
        help="标签目录名（输入与输出都使用该名称）。",
    )

    # 输出目录名：最终会写入 <dataset-dir>/<output-dir-name>/images 和 labels。
    # 该目录会自动从来源目录列表中排除，避免把历史合并结果再次当作输入。
    parser.add_argument(
        "--output-dir-name",
        default="all",
        help="合并结果目录名（位于 dataset 根目录下）。",
    )

    # 是否递归读取子目录文件。
    # 不加该参数时，只读取 images/labels 当前层级的文件。
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描 images/labels 子目录中的文件。",
    )

    # 预演模式：只打印将执行的操作，不实际复制文件、不写入新标签。
    # 首次使用建议先开这个参数确认无误，再正式执行。
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览操作，不实际写入任何文件。",
    )

    # 详细日志：逐文件打印图片复制和标签写入信息；不加时只输出每个来源目录的汇总。
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出逐文件日志（图片/标签每条操作都会打印）。",
    )

    # 执行前清空输出目录（仅删除输出 images/labels 下已有文件）。
    # 常用于重新合并，避免旧文件残留；与 --dry-run 同时使用时不会真的删除。
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="合并前先清空输出 images/labels 里的已有文件。",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    sources = build_sources(dataset_dir, args.output_dir_name)

    output_root = dataset_dir / args.output_dir_name
    output_images = output_root / args.images_dir_name
    output_labels = output_root / args.labels_dir_name
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    if args.clear_output and not args.dry_run:
        removed_images = clear_folder(output_images)
        removed_labels = clear_folder(output_labels)
        print(f"Cleared output images: {removed_images}")
        print(f"Cleared output labels: {removed_labels}")

    used_target_stems: set[str] = set()
    total_images = 0
    total_labels = 0

    print("Class mapping:")
    for s in sources:
        print(f"  {s.class_id}: {s.path.name} -> prefix={s.prefix}, class_name={s.class_name}")

    for source in sources:
        video_dir = source.path / args.video_dir_name
        images_dir = video_dir / args.images_dir_name
        labels_dir = video_dir / args.labels_dir_name

        if not images_dir.exists() or not labels_dir.exists():
            raise SystemExit(
                f"Missing images/labels in {source.path}. "
                f"Expected: {images_dir} and {labels_dir}"
            )

        image_files = collect_images(images_dir, args.recursive)
        label_files = collect_labels(labels_dir, args.recursive)
        name_map = build_target_name_map(
            image_files=image_files,
            label_files=label_files,
            images_dir=images_dir,
            labels_dir=labels_dir,
            prefix=source.prefix,
            used_target_stems=used_target_stems,
        )

        folder_images = 0
        folder_labels = 0

        for image_file in image_files:
            key = file_key(image_file, images_dir)
            target_name = f"{name_map[key]}{image_file.suffix.lower()}"
            target_path = output_images / target_name
            if args.verbose:
                print(f"[IMAGE] {image_file} -> {target_path}")
            if not args.dry_run:
                shutil.copy2(image_file, target_path)
            total_images += 1
            folder_images += 1

        for label_file in label_files:
            key = file_key(label_file, labels_dir)
            target_name = f"{name_map[key]}.txt"
            target_path = output_labels / target_name
            if args.verbose:
                print(
                    f"[LABEL] {label_file} -> {target_path} (class={source.class_id})"
                )
            if not args.dry_run:
                old_text = label_file.read_text(encoding="utf-8")
                new_text = remap_yolo_label_text(old_text, source.class_id)
                target_path.write_text(new_text, encoding="utf-8")
            total_labels += 1
            folder_labels += 1

        print(
            f"  Folder {source.class_id} done: "
            f"images={folder_images}, labels={folder_labels}"
        )

    if not args.dry_run:
        classes_path = output_root / "classes.txt"
        class_mapping_path = output_root / "class_mapping.txt"
        classes_path.write_text(
            "\n".join([s.class_name for s in sources]) + "\n",
            encoding="utf-8",
        )
        class_mapping_path.write_text(
            "\n".join(
                [
                    f"{s.class_id}\t{s.class_name}\t{s.prefix}\t{s.path.name}"
                    for s in sources
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    mode = "DRY-RUN" if args.dry_run else "DONE"
    print(f"\n[{mode}] images copied: {total_images}")
    print(f"[{mode}] labels copied: {total_labels}")
    print(f"Output images: {output_images}")
    print(f"Output labels: {output_labels}")


if __name__ == "__main__":
    main()
