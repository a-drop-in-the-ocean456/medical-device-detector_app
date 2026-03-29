#!/usr/bin/env python3
"""
Clean up images and labels that don't exist in images_yolo_vis

This script compares files in the images and labels folders against
the images_yolo_vis folder, and deletes any files from images/labels
that don't have a corresponding file in images_yolo_vis.

Usage:
    python cleanup_extra_files.py [base_path]
    
If no base_path is provided, it defaults to the current directory.
The script expects three subdirectories: images, labels, images_yolo_vis
"""
import os
import sys
from pathlib import Path


def cleanup_extra_files(base_path="."):
    """Clean up images and labels not in images_yolo_vis"""
    
    base = Path(base_path)
    vis_dir = base / "images_yolo_vis"
    images_dir = base / "images"
    labels_dir = base / "labels"
    
    # Check if required directories exist
    if not vis_dir.exists():
        print(f"Error: images_yolo_vis folder not found at {vis_dir}")
        return False
    
    if not images_dir.exists():
        print(f"Warning: images folder not found at {images_dir}")
    
    if not labels_dir.exists():
        print(f"Warning: labels folder not found at {labels_dir}")
    
    # Get all base names from images_yolo_vis (without extension)
    vis_basenames = set()
    for f in vis_dir.iterdir():
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            vis_basenames.add(f.stem)
    
    print(f"Found {len(vis_basenames)} reference files in images_yolo_vis")
    
    # Process images
    deleted_images = 0
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                if f.stem not in vis_basenames:
                    print(f"Deleting image: {f.name}")
                    f.unlink()
                    deleted_images += 1
        
        print(f"Deleted {deleted_images} images from {images_dir}")
    else:
        print("Skipping images folder (not found)")
    
    # Process labels
    deleted_labels = 0
    if labels_dir.exists():
        for f in labels_dir.iterdir():
            if f.suffix.lower() == '.txt':
                if f.stem not in vis_basenames:
                    print(f"Deleting label: {f.name}")
                    f.unlink()
                    deleted_labels += 1
        
        print(f"Deleted {deleted_labels} labels from {labels_dir}")
    else:
        print("Skipping labels folder (not found)")
    
    print("=" * 50)
    print(f"Summary:")
    print(f"  Reference files in images_yolo_vis: {len(vis_basenames)}")
    print(f"  Images deleted: {deleted_images}")
    print(f"  Labels deleted: {deleted_labels}")
    print("Done!")
    
    return True


if __name__ == "__main__":
    # Get base path from command line argument or use current directory
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"Processing folder: {base_path}")
    print("=" * 50)
    
    cleanup_extra_files(base_path)
