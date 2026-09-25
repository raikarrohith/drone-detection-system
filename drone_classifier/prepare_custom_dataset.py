import os
import shutil
import random
import zipfile
from pathlib import Path
from PIL import Image

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SOURCE_DIR = Path("drone_images")
DEST_DIR = Path("drone_classifier/dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def auto_extract_zips():
    """Auto-extracts any .zip files found in drone_images."""
    for zip_path in list(SOURCE_DIR.rglob("*.zip")):
        print(f"[*] Found zip archive: {zip_path.name}. Auto-extracting...", flush=True)
        try:
            target_extract = zip_path.parent / zip_path.stem
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_extract)
            print(f"  [+] Extracted to: {target_extract}", flush=True)
        except Exception as e:
            print(f"  [!] Failed to extract {zip_path.name}: {e}", flush=True)

def crop_drone_box(img, label_file):
    """If a YOLO label file exists, crops the drone bounding box with a 10% safety margin."""
    if not label_file.exists():
        return img
    try:
        lines = label_file.read_text().strip().splitlines()
        if not lines:
            return img
        # Take the largest bounding box if multiple
        boxes = []
        w, h = img.size
        for l in lines:
            parts = l.strip().split()
            if len(parts) >= 5:
                cls_id, xc, yc, bw, bh = map(float, parts[:5])
                boxes.append((bw * bh, xc, yc, bw, bh))
        if not boxes:
            return img
        boxes.sort(key=lambda x: x[0], reverse=True)
        _, xc, yc, bw, bh = boxes[0]

        # Convert normalized coordinates to pixel coordinates
        x1 = max(0, int((xc - bw / 2.0 - bw * 0.08) * w))
        y1 = max(0, int((yc - bh / 2.0 - bh * 0.08) * h))
        x2 = min(w, int((xc + bw / 2.0 + bw * 0.08) * w))
        y2 = min(h, int((yc + bh / 2.0 + bh * 0.08) * h))

        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
            return img.crop((x1, y1, x2, y2))
    except Exception:
        pass
    return img

def prepare_custom_dataset(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    if not SOURCE_DIR.exists():
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[!] Created folder: '{SOURCE_DIR.resolve()}'")
        return

    # 1. Check and extract any zips first
    auto_extract_zips()

    print("=" * 70)
    print("  SMART DRONE DATASET IMPORTER & SPLITTER")
    print(f"  Source: {SOURCE_DIR.resolve()}")
    print(f"  Destination: {DEST_DIR.resolve()}")
    print("=" * 70)

    # Clean existing destination dataset
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR, ignore_errors=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for cdir in SOURCE_DIR.iterdir():
        if not cdir.is_dir():
            continue
        cname = cdir.name

        # Check if this class has Roboflow pre-split subfolders: train/valid/test
        sub_splits = {}
        for candidate in ["train", "valid", "val", "test"]:
            sdir = cdir / candidate
            if sdir.is_dir():
                target_split = "val" if candidate in ["valid", "val"] else candidate
                sub_splits[target_split] = sdir

        if "train" in sub_splits:
            print(f"[*] Processing pre-split Roboflow dataset for class: [{cname}]...")
            summary[cname] = {"train": 0, "val": 0, "test": 0}

            for split_key, sdir in sub_splits.items():
                img_dir = sdir / "images" if (sdir / "images").is_dir() else sdir
                lbl_dir = sdir / "labels" if (sdir / "labels").is_dir() else None

                raw_files = [f for f in img_dir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]

                dst_dir = DEST_DIR / split_key / cname
                dst_dir.mkdir(parents=True, exist_ok=True)

                saved_count = 0
                for idx, src_file in enumerate(raw_files, 1):
                    try:
                        with Image.open(src_file) as img:
                            img = img.convert("RGB")
                            # If labels exist, crop the drone
                            if lbl_dir:
                                lbl_file = lbl_dir / f"{src_file.stem}.txt"
                                img = crop_drone_box(img, lbl_file)

                            dst_file = dst_dir / f"{cname}_{idx:04d}.jpg"
                            img.save(dst_file, format="JPEG", quality=95)
                            saved_count += 1
                    except Exception:
                        pass

                summary[cname][split_key] = saved_count
        else:
            # Flat folder or un-split dataset
            raw_files = [f for f in cdir.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            if not raw_files:
                continue

            print(f"[*] Processing flat folder for class: [{cname}] ({len(raw_files)} images)...")
            valid_images = []
            for f in raw_files:
                try:
                    with Image.open(f) as img:
                        img.verify()
                    valid_images.append(f)
                except Exception:
                    pass

            if not valid_images:
                continue

            random.shuffle(valid_images)
            total = len(valid_images)
            n_train = max(1, int(total * train_ratio))
            n_val = max(1, int(total * val_ratio)) if total >= 3 else 0
            n_test = total - n_train - n_val

            splits = {
                "train": valid_images[:n_train],
                "val": valid_images[n_train:n_train + n_val],
                "test": valid_images[n_train + n_val:] if n_test > 0 else valid_images[n_train:]
            }

            summary[cname] = {"train": 0, "val": 0, "test": 0}
            for sname, sfiles in splits.items():
                sdir = DEST_DIR / sname / cname
                sdir.mkdir(parents=True, exist_ok=True)
                for idx, src_file in enumerate(sfiles, 1):
                    try:
                        with Image.open(src_file) as img:
                            img = img.convert("RGB")
                            dst_file = sdir / f"{cname}_{idx:04d}.jpg"
                            img.save(dst_file, format="JPEG", quality=95)
                            summary[cname][sname] += 1
                    except Exception:
                        pass

    if not summary:
        print("[!] No images found in any class folders.")
        return

    print("\n" + "=" * 70)
    print("  CUSTOM DATASET PROCESSED & READY FOR TRAINING")
    print("=" * 70)
    for cname, counts in summary.items():
        tot = counts["train"] + counts["val"] + counts["test"]
        print(f"  - {cname:<18}: Total={tot:<4} (Train={counts['train']}, Val={counts['val']}, Test={counts['test']})")
    print("=" * 70)
    print("\nNext step: Run 'python drone_classifier/train.py'")

if __name__ == "__main__":
    prepare_custom_dataset()
