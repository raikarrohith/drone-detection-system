#!/usr/bin/env python3
"""Organize extracted UAV-Identification dataset into YOLO format for training."""

from pathlib import Path
import shutil

def main():
    root = Path("D:/drone/drone-detection-system")
    src_images = root / "datasets/UAV-Identification/Images"
    src_labels = root / "datasets/UAV-Identification/Annotations/Annotations/Yolo"
    
    out_dir = root / "dataset_raw/drone_dataset"
    
    for split_in, split_out in [("Train", "train"), ("Valid", "val"), ("Test", "val")]:
        img_src_dir = src_images / split_in
        lbl_src_dir = src_labels / split_in
        
        img_dst_dir = out_dir / "images" / split_out
        lbl_dst_dir = out_dir / "labels" / split_out
        
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)
        
        if not img_src_dir.exists():
            continue
            
        copied_imgs = 0
        copied_lbls = 0
        
        for img_file in img_src_dir.glob("*.*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                dst_img = img_dst_dir / img_file.name
                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)
                    copied_imgs += 1
                
                # Check matching label file
                lbl_file = lbl_src_dir / f"{img_file.stem}.txt"
                if lbl_file.exists():
                    dst_lbl = lbl_dst_dir / lbl_file.name
                    if not dst_lbl.exists():
                        shutil.copy2(lbl_file, dst_lbl)
                        copied_lbls += 1
                        
        print(f"[+] Processed '{split_in}' -> '{split_out}': {copied_imgs} images, {copied_lbls} labels")

    # Update data.yaml
    data_yaml_path = root / "data.yaml"
    data_yaml_content = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val

nc: 1
names:
  0: drone
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
        
    print(f"[+] Updated data.yaml pointing to: {out_dir.as_posix()}")

if __name__ == "__main__":
    main()
