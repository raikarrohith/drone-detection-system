import os
import shutil
import random
from pathlib import Path
from PIL import Image
from icrawler.builtin import BingImageCrawler

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = Path("drone_classifier/dataset")
TEMP_DIR = Path("drone_classifier/raw_download")

TARGETS = {
    "DJI-Mini": [
        "DJI Mini 3 Pro drone flying",
        "DJI Mini 2 drone photo",
        "DJI Mini 4 Pro quadcopter",
        "DJI Mavic Mini drone in air"
    ],
    "DJI-Neo": [
        "DJI Neo drone flying",
        "DJI Neo palm drone",
        "DJI Neo quadcopter outdoors",
        "DJI Neo mini drone photo"
    ]
}

def clean_and_validate(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w < 100 or h < 100:
                return False
        return True
    except Exception:
        return False

def collect_images_for_class(class_name, queries, target_count=100):
    class_temp = TEMP_DIR / class_name
    class_temp.mkdir(parents=True, exist_ok=True)
    
    per_query = max(25, target_count // len(queries) + 10)
    for q in queries:
        print(f"[*] Crawling for [{class_name}]: '{q}'...")
        crawler = BingImageCrawler(
            storage={'root_dir': str(class_temp)},
            downloader_threads=4
        )
        crawler.crawl(keyword=q, max_num=per_query)

    valid_images = []
    for f in class_temp.iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if clean_and_validate(f):
                valid_images.append(f)
            else:
                try:
                    f.unlink()
                except Exception:
                    pass
                    
    print(f"[+] Total valid images collected for [{class_name}]: {len(valid_images)}")
    return valid_images

def split_and_distribute(class_name, images, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(images)
    
    # Take up to 100-120 images
    images = images[:120]
    total = len(images)
    
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }
    
    for split_name, file_list in splits.items():
        dst_dir = DATA_DIR / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for idx, src_file in enumerate(file_list):
            dst_file = dst_dir / f"{class_name}_{idx+1:04d}{src_file.suffix.lower()}"
            shutil.copy2(src_file, dst_file)
            
    print(f"[+] Distributed [{class_name}]: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

def main():
    print("=" * 60)
    print("  DJI-Mini & DJI-Neo Automatic Image Collection & Dataset Split")
    print("=" * 60)
    
    for class_name, queries in TARGETS.items():
        valid_imgs = collect_images_for_class(class_name, queries, target_count=100)
        split_and_distribute(class_name, valid_imgs)
        
    print("\n[SUCCESS] Image collection and train/val/test splitting complete!")
    print(f"Target location: {DATA_DIR}")

if __name__ == "__main__":
    main()
