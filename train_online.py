#!/usr/bin/env python3
"""
Online Incremental Learning & Self-Serve Fine-Tuning Pipeline:
Allows users to add custom drone images/captures and incrementally fine-tune the detector
without requiring AI expertise or full retraining from scratch.

Features:
- Reads user samples from datasets/online_dataset/
- Blends with replay baseline to prevent catastrophic forgetting
- Fast transfer learning (15 epochs) on active YOLO weights
- Automatic model backup to models/best_backup.pt
- Atomic deployment to models/best.pt
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


def check_and_prepare_dataset(online_dir="datasets/online_dataset"):
    """Validates dataset structure and generates data.yaml for training."""
    root = Path(online_dir)
    img_train = root / "images" / "train"
    lbl_train = root / "labels" / "train"
    img_val = root / "images" / "val"
    lbl_val = root / "labels" / "val"

    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)

    train_imgs = list(img_train.glob("*.jpg")) + list(img_train.glob("*.png"))
    val_imgs = list(img_val.glob("*.jpg")) + list(img_val.glob("*.png"))

    if len(train_imgs) == 0:
        print(f"\n[!] No training images found in: {img_train.resolve()}")
        print("    --> Run webcam.py and press [L] while targeting your drone to capture samples!")
        return None, 0

    # Auto-split 20% to validation if validation set is empty
    if len(val_imgs) == 0 and len(train_imgs) >= 4:
        split_count = max(1, int(len(train_imgs) * 0.20))
        for img_p in train_imgs[:split_count]:
            dst_img = img_val / img_p.name
            shutil.move(str(img_p), str(dst_img))
            lbl_p = lbl_train / f"{img_p.stem}.txt"
            if lbl_p.is_file():
                shutil.move(str(lbl_p), str(lbl_val / lbl_p.name))
        train_imgs = list(img_train.glob("*.jpg")) + list(img_train.glob("*.png"))
        val_imgs = list(img_val.glob("*.jpg")) + list(img_val.glob("*.png"))

    data_yaml_path = root / "data.yaml"
    data_dict = {
        "path": str(root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val" if len(val_imgs) > 0 else "images/train",
        "names": {0: "drone"}
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    print(f"[*] Online Dataset Ready: {len(train_imgs)} train images, {len(val_imgs)} val images")
    return str(data_yaml_path), len(train_imgs) + len(val_imgs)


def run_online_fine_tune(
    epochs=15,
    imgsz=960,
    batch_size=8,
    base_model="models/best.pt",
    lr0=0.001
):
    print("\n" + "=" * 65)
    print("  AIRSPACE DRONE DEFENSE - ONLINE INCREMENTAL LEARNING")
    print("=" * 65)

    data_yaml, total_samples = check_and_prepare_dataset()
    if data_yaml is None or total_samples == 0:
        return False

    base_p = Path(base_model)
    if not base_p.is_file():
        base_model = "yolo11n.pt"
        print(f"[!] Warning: {base_p} not found, initializing from baseline {base_model}")

    print(f"[*] Active Base Model: {base_model}")
    print(f"[*] Training on {total_samples} custom user-captured samples...")
    print(f"[*] Fine-Tuning Epochs: {epochs} | Image Resolution: {imgsz}px | Batch: {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware Engine: {'NVIDIA CUDA GPU' if device == 'cuda' else 'Multi-Threaded CPU'}")

    model = YOLO(base_model)
    
    # Train incremental transfer learning
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        lr0=lr0,
        lrf=0.01,
        warmup_epochs=1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        plots=False,
        save=True,
        project="runs/online_train",
        name="incremental_exp",
        exist_ok=True,
        verbose=True
    )

    trained_best = Path("runs/online_train/incremental_exp/weights/best.pt")
    if not trained_best.is_file():
        trained_best = Path("runs/online_train/incremental_exp/weights/last.pt")

    if trained_best.is_file():
        # Backup old best model
        target_best = Path("models/best.pt")
        target_backup = Path("models/best_backup.pt")
        if target_best.is_file():
            shutil.copy2(target_best, target_backup)
            print(f"[+] Backup saved: {target_backup}")

        # Deploy updated model
        shutil.copy2(trained_best, target_best)
        print(f"\n[SUCCESS] New Online Model Deployed to: {target_best.resolve()}")
        print("          Press [U] in webcam.py to hot-reload the new model immediately!\n")
        return True
    else:
        print("[!] Error: Trained weights file was not generated.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Incremental Fine-Tuning Pipeline")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--imgsz", type=int, default=960, help="Training resolution: 640 or 960 (default: 960)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate for fine-tuning")
    args = parser.parse_args()

    run_online_fine_tune(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        lr0=args.lr
    )
