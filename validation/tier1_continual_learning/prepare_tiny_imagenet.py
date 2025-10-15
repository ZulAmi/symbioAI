#!/usr/bin/env python3
"""
Prepare TinyImageNet-200 validation set
Reorganizes flat validation images into class folders for PyTorch ImageFolder
"""

import os
import shutil
from pathlib import Path

def prepare_tiny_imagenet_val(data_dir: str = "./data/tiny-imagenet-200"):
    """Reorganize TinyImageNet validation set into class folders."""
    
    val_dir = Path(data_dir) / "val"
    val_images_dir = val_dir / "images"
    val_annotations_file = val_dir / "val_annotations.txt"
    
    print("🔧 Preparing TinyImageNet-200 validation set...")
    print(f"   Location: {val_dir}")
    
    # Check if already organized
    class_folders = [d for d in val_dir.iterdir() if d.is_dir() and d.name.startswith('n')]
    if class_folders:
        print(f"✅ Validation set already organized ({len(class_folders)} class folders found)")
        return
    
    # Check if we have the raw structure
    if not val_images_dir.exists() or not val_annotations_file.exists():
        print("⚠️  Validation set appears already organized or missing files")
        return
    
    print(f"📋 Reading annotations from {val_annotations_file.name}...")
    
    # Read annotations
    image_to_class = {}
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = parts[1]
                image_to_class[img_name] = class_id
    
    print(f"   Found {len(image_to_class)} image annotations")
    print(f"   Classes: {len(set(image_to_class.values()))}")
    
    # Create class directories and move images
    print("📁 Creating class directories and organizing images...")
    
    moved_count = 0
    for img_name, class_id in image_to_class.items():
        # Create class directory
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)
        
        # Move image
        src = val_images_dir / img_name
        dst = class_dir / img_name
        
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_count += 1
    
    print(f"   Moved {moved_count} images into class folders")
    
    # Clean up empty images directory
    if val_images_dir.exists() and not list(val_images_dir.iterdir()):
        val_images_dir.rmdir()
        print("   Removed empty images directory")
    
    # Optionally remove annotations file (no longer needed)
    if val_annotations_file.exists():
        val_annotations_file.unlink()
        print("   Removed annotations file")
    
    print("✅ TinyImageNet-200 validation set prepared!")
    print(f"   Structure: val/{len(set(image_to_class.values()))} class folders")


if __name__ == "__main__":
    import sys
    
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/tiny-imagenet-200"
    
    prepare_tiny_imagenet_val(data_dir)
