import os
import re
import time
import shutil
import random
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = Path("drone_classifier/dataset")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

QUERIES = {
    "DJI-Mini": [
        "DJI Mini 3 Pro drone",
        "DJI Mini 4 Pro drone flying",
        "DJI Mini 2 quadcopter in air",
        "DJI Mavic Mini drone photo",
        "DJI Mini 3 drone outdoors",
        "DJI Mini 4 Pro flying in sky"
    ],
    "DJI-Neo": [
        "DJI Neo drone flying",
        "DJI Neo palm drone",
        "DJI Neo quadcopter photo",
        "DJI Neo drone in hand",
        "DJI Neo 4k mini drone",
        "DJI Neo drone outdoors"
    ]
}

def search_bing_images(query, max_images=50):
    urls = []
    for first in range(0, max_images + 30, 30):
        url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&first={first}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                found = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
                for u in found:
                    if u not in urls:
                        urls.append(u)
            time.sleep(0.5)
        except Exception as e:
            print(f"[!] Warning fetching search page: {e}")
        if len(urls) >= max_images:
            break
    return urls

def download_and_verify(url, timeout=8):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200:
            img = Image.open(BytesIO(resp.content))
            img.verify()  # verify format integrity
            # re-open to load pixel data
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            if w >= 150 and h >= 150:
                return img
    except Exception:
        pass
    return None

def collect_images_for_class(class_name, query_list, target_count=100):
    print(f"\n[*] Collecting images for [{class_name}] (Target: {target_count})...")
    all_urls = []
    for q in query_list:
        urls = search_bing_images(q, max_images=40)
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        print(f"  - Query '{q}': found {len(urls)} image links (Total unique: {len(all_urls)})")
        time.sleep(0.5)

    random.shuffle(all_urls)
    valid_images = []
    
    print(f"[*] Downloading and verifying images for [{class_name}]...")
    for idx, u in enumerate(all_urls):
        img = download_and_verify(u)
        if img is not None:
            valid_images.append(img)
            print(f"  [+] ({len(valid_images)}/{target_count}) Downloaded: {img.size[0]}x{img.size[1]}px")
        if len(valid_images) >= target_count:
            break
            
    print(f"[+] Total successfully collected for [{class_name}]: {len(valid_images)}")
    return valid_images

def save_and_split(class_name, images, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    random.shuffle(images)
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
    
    for split_name, img_list in splits.items():
        dst_dir = DATA_DIR / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(img_list):
            save_path = dst_dir / f"{class_name}_{idx+1:04d}.jpg"
            img.save(save_path, format="JPEG", quality=92)
            
    print(f"[+] Distributed [{class_name}]: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

def main():
    print("=" * 65)
    print("  AUTOMATED IMAGE HARVESTING: DJI-Mini & DJI-Neo DATASET")
    print("=" * 65)
    
    for class_name, queries in QUERIES.items():
        images = collect_images_for_class(class_name, queries, target_count=100)
        save_and_split(class_name, images)
        
    print("\n" + "=" * 65)
    print("  [SUCCESS] All 100 images per class downloaded, verified, and split!")
    print(f"  Location: {DATA_DIR.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
