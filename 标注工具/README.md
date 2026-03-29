# 标注工具目录说明

本目录包含用于视频自动标注、数据集处理和质量控制的各类脚本工具。

## 目录结构

```
标注工具/
├── README.md                     # 本说明文档
├── requirements.txt              # Python依赖包列表
├── video_annotator_cli.py         # 无GUI的命令行视频自动标注工具
├── cleanup_extra_files.py         # 清理多余文件的工具
├── merge_dataset_all.py          # 合并多个类别数据集的脚本
└── dataset/                      # 原始数据集目录
    ├── all/                      # 合并后的完整数据集
    │   ├── images/               # 所有类别的图片（已合并重命名）
    │   ├── labels/               # 所有类别的标注（已合并重命名）
    │   ├── classes.txt           # 类别名称列表
    │   ├── class_mapping.txt     # 类别映射文件
    │   ├── split_yolo_dataset.py # 数据集分割脚本
    │   ├── visualize_yolo_dataset.py # 标注可视化脚本
    │   ├── split_dataset/        # 分割后的数据集
    │   └── ultralytics-main/     # YOLO训练相关文件
    ├── 一次性医疗用品—2ml注射器syringe/  # 2ml注射器图片和视频
    ├── 医疗器械—持针器Needle holder/      # 持针器图片和视频
    ├── 医疗废物—感染性废物waste/          # 感染性废物图片和视频
    ├── 无菌物品—大纱large gauze/          # 大纱布图片和视频
    └── 消毒用品—碘伏Iodophor/             # 碘伏图片和视频
```

## 文件作用详解

### 核心标注工具

| 文件 | 作用 |
|------|------|
| [`video_annotator_cli.py`](标注工具/video_annotator_cli.py:1) | 无GUI的命令行视频自动标注工具。使用 Grounding DINO 进行开放集检测，配合 ByteTrack 进行目标跟踪，自动生成 YOLO 格式的标注文件。适合批量处理视频数据。 |
| [`requirements.txt`](标注工具/requirements.txt:1) | Python 依赖包清单，包含 PySide6、OpenCV、PyTorch、transformers 等必要库。安装命令：`pip install -r requirements.txt` |

### 数据集处理工具

| 文件 | 作用 |
|------|------|
| [`merge_dataset_all.py`](标注工具/merge_dataset_all.py:1) | 合并多个类别数据集的脚本。将 dataset 目录下的 5 个子文件夹（代表 5 个类别）合并到 `dataset/all` 目录，同时重映射 YOLO 类别 ID（0~4）。支持预览模式（--dry-run）和详细日志（--verbose）。 |
| [`dataset/all/split_yolo_dataset.py`](标注工具/dataset/all/split_yolo_dataset.py:1) | 将 YOLO 格式的数据集分割为训练集和测试集。支持自定义分割比例、随机种子、复制或移动模式。输出目录结构：`split_dataset/images/train|test` 和 `split_dataset/labels/train|test`。 |
| [`dataset/all/visualize_yolo_dataset.py`](标注工具/dataset/all/visualize_yolo_dataset.py:1) | 在所有图片上绘制 YOLO 标签边界框并保存可视化结果。便于人工审查标注质量，每个类别使用不同颜色区分。 |
| [`cleanup_extra_files.py`](标注工具/cleanup_extra_files.py:1) | 清理多余文件的工具。对比 `images_yolo_vis` 目录与 `images`、`labels` 目录，删除在可视化目录中不存在的图片和标签文件。 |

### 配置文件

| 文件 | 作用 |
|------|------|
| [`dataset/all/classes.txt`](标注工具/dataset/all/classes.txt:1) | YOLO 数据集的类别名称列表（英文），共 5 个类别：syringe、Needle holder、waste、large gauze、Iodophor |
| [`dataset/all/class_mapping.txt`](标注工具/dataset/all/class_mapping.txt:1) | 类别映射文件，包含：类别ID、英文名称、简称、中文完整名称。用于不同系统间的类别转换。 |

## 使用示例

### 视频自动标注（CLI）

```bash
python video_annotator_cli.py --video your_video.mp4 --classes syringe,needle --output-dir auto_labels
```

### 合并数据集

```bash
python merge_dataset_all.py --dataset-dir dataset --dry-run  # 预览
python merge_dataset_all.py --dataset-dir dataset --clear-output  # 执行合并
```

### 分割数据集

```bash
python dataset/all/split_yolo_dataset.py --images-dir dataset/all/images --labels-dir dataset/all/labels --output-dir split_dataset --train-ratio 0.9
```

### 可视化标注结果

```bash
python dataset/all/visualize_yolo_dataset.py --images-dir dataset/all/images --labels-dir dataset/all/labels --classes-file dataset/all/classes.txt --output-dir vis_images
```

---

# 视频自动标注工具（关键帧 + 开放集检测）

基于 Qt + OpenCV 的视频标注工具：

- 在关键帧上手动绘制边界框（1帧或多帧）
- 自动跟踪关键帧之间的每个段落
- 可选 ByteTrack 模式（`BYTE_TRACK`），用于更强的多目标时序一致性
- 可选 Grounding DINO 开放集检测器模式（`GROUNDING_DINO`）
- 默认跟踪器使用光流 + 模板匹配融合（`FLOW_TM`）
- 如有需要，你仍可在 UI 中切换到 OpenCV 跟踪器（CSRT/KCF/MIL）
- 支持自定义类别名称（不限于 COCO 80 类）
- 始终导出配对的 `images/` + `labels/`
- UI 内置输出标签查看器

## 安装

```bash
pip install -r requirements.txt
```

## 运行

无 GUI 的 CLI 版本（Grounding DINO + ByteTrack）：

```bash
python video_annotator_cli.py --video your_video.mp4 --classes road_crack,baozi --output-dir auto_labels_cli
```

CLI 会显示进度条（如果安装了 `tqdm` 则使用它，否则回退到周期性文本日志）。

YOLO 标签可视化（为一文件夹的图片和标签绘制所有边界框）：

```bash
python visualize_yolo_labels.py --images-dir path/to/images --labels-dir path/to/labels --output-dir path/to/vis --classes-file path/to/classes.txt --recursive
```

## 工作流程

1. 打开视频
2. 选择一帧
3. 输入类别名称
4. 启用 `Draw Box Mode` 并绘制边界框
5. 点击 `Save/Update Keyframe`
6. 移动到可能出现漂移的后续帧，调整边界框，再次点击 `Save/Update Keyframe`
7. 重复步骤 6 以添加更多校正点
8. 点击 `Run Auto Annotation`
9. 点击 `View Output Labels`（或在完成后弹出时打开）

## 输出

默认输出目录：`auto_labels/`

- `labels/*.txt`：YOLO txt 格式标签
- `images/*.jpg`：每个标签文件对应的帧图片
- `classes.txt`：你手动输入的类别列表
- `meta.json`：运行元数据

## 注意事项

- 这是跟踪，不是基于检测器的重新检测
- `GROUNDING_DINO` 模式需要在首次运行时从 Hugging Face 下载模型
- `BYTE_TRACK` 模式在后台使用 Grounding DINO 检测和 ByteTrack 关联
- 在严重遮挡/快速运动的情况下，使用更多关键帧以减少漂移
- 在最终发布数据集前，请审查自动标注结果

## 故障排除

- 如果看到 `RuntimeError: operator torchvision::nms does not exist`，说明你的 `torch` 和 `torchvision` 版本不兼容
- 重新安装匹配的一对（CPU 示例）：

```bash
python -m pip uninstall -y torch torchvision
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu