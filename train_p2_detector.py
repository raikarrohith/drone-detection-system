#!/usr/bin/env python3
"""
Train YOLO11n-P2 4-Head Tiny-Object Detection Model for Long-Range Small Drone Detection.

Features:
- Stride 4 (P2) Feature Pyramid Network for tiny distant drones (4x4 to 16x16 px).
- Small-object augmentation (Mosaic, Copy-Paste, scale downsampling).
- Transfer learning from pre-trained YOLO11 backbone.
- Automated backup and installation to models/best.pt.
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def check_dataset_config(data_path):
    """Ensure dataset configuration file exists and has valid paths."""
    data_file = Path(data_path)
    if not data_file.is_file():
        print(f"[!] Warning: Dataset config '{data_path}' not found.")
        print("    Please create a data.yaml pointing to your drone dataset images and labels.")
        return False
    return True


def train_p2_detector(
    data_yaml="data.yaml",
    epochs=50,
    imgsz=640,
    batch_size=16,
    weights_init="yolo11n.pt",
    output_model="models/best_p2.pt",
    replace_active=False
):
    print("=" * 65)
    print("  YOLO11n-P2 4-HEAD TINY-OBJECT DRONE DETECTOR TRAINING")
    print("  - Head Architecture: P2/4 (Stride 4), P3/8, P4/16, P5/32")
    print("  - Target Object Scale: Micro & Distant Drones (4x4 to 16x16 px)")
    print(f"  - Device: {'NVIDIA CUDA' if torch.cuda.is_available() else 'Multi-Threaded CPU'}")
    print(f"  - Epochs: {epochs} | Image Size: {imgsz}px | Batch Size: {batch_size}")
    print("=" * 65 + "\n")

    arch_yaml = Path("models/yolo11n-p2.yaml")
    if not arch_yaml.is_file():
        raise FileNotFoundError(f"P2 Architecture file {arch_yaml} not found!")

    # 1. Initialize 4-Head P2 Model Architecture
    model = YOLO(str(arch_yaml))

    # 2. Transfer Backbone Weights from Pretrained YOLO11
    if weights_init and (Path(weights_init).is_file() or weights_init.endswith(".pt")):
        try:
            print(f"[*] Loading pre-trained transfer weights from: {weights_init}")
            model.load(weights_init)
            print("[*] Successfully transferred backbone representations.")
        except Exception as e:
            print(f"[!] Note: Training initialized with random P2 head init ({e})")

    # 3. Configure Small-Object Optimized Hyperparameters
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "workers": min(4, os.cpu_count() or 2),
        "optimizer": "auto",
        "patience": 15,
        "save": True,
        "save_period": 10,
        "cache": False,
        # Small-Object Augmentation Suite
        "mosaic": 1.0,        # 4-image mosaic shrinks objects by 50% to simulate distant targets
        "mixup": 0.10,         # Mixup for contrast robustness
        "scale": 0.4,          # Random scale jitter [0.6x, 1.4x]
        "fliplr": 0.5,         # Horizontal flip
        "copy_paste": 0.20,    # Copy-paste drone instances on diverse sky/ground backgrounds
        "degrees": 10.0,       # 3D tilt/roll simulation
        # Loss Gains tailored for small bounding boxes
        "box": 7.5,            # High CIoU weight for boundary precision
        "cls": 0.5,
        "dfl": 1.5,
        "verbose": True
    }

    print("[*] Commencing P2 Model Training...")
    results = model.train(**train_args)

    # 4. Save Trained Checkpoint
    best_weights_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best_weights_path.is_file():
        out_path = Path(output_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights_path, out_path)
        print(f"\n[+] Trained P2 Model saved to: {out_path}")

        if replace_active:
            active_path = Path("models/best.pt")
            backup_path = Path("models/best_backup.pt")
            if active_path.is_file():
                shutil.copy2(active_path, backup_path)
                print(f"[*] Backed up previous active model to: {backup_path}")
            shutil.copy2(best_weights_path, active_path)
            print(f"[+] Replaced active detector: {active_path} is now running P2 4-Head!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-P2 4-Head Tiny-Object Drone Detector")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset YAML config")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Base weights for transfer learning")
    parser.add_argument("--output", type=str, default="models/best_p2.pt", help="Output path for best weights")
    parser.add_argument("--replace-best", action="store_true", help="Automatically install as models/best.pt")
    args = parser.parse_args()

    train_p2_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        weights_init=args.weights,
        output_model=args.output,
        replace_active=args.replace_best
    )


if __name__ == "__main__":
    main()
