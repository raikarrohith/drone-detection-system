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
        "DJI Mini 4 Pro drone in air",
        "DJI Mini 2 quadcopter flying",
        "DJI Mavic Mini drone photo",
        "DJI Mini 3 drone outdoors",
        "DJI Mini 4 Pro quadcopter photo",
        "DJI Mini SE drone flying",
        "DJI Mini drone top view"
    ],
    "DJI-Neo": [
        "DJI Neo drone flying",
        "DJI Neo palm drone",
        "DJI Neo quadcopter in air",
        "DJI Neo drone photo",
        "DJI Neo 4k mini drone",
        "DJI Neo drone outdoors",
        "DJI Neo quadcopter flying",
        "DJI Neo drone in hand"
    ]
}

def search_bing_images(query, max_images=45):
    urls = []
    for first in range(0, max_images + 30, 30):
        url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&first={first}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                found = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
                for u in found:
                    if u not in urls:
                        urls.append(u)
            time.sleep(0.3)
        except Exception as e:
            pass
        if len(urls) >= max_images:
            break
    return urls

def download_image(url, timeout=6):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 5000:
            img = Image.open(BytesIO(resp.content))
            img.verify()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            if w >= 120 and h >= 120:
                return img
    except Exception:
        pass
    return None

def main():
    print("=" * 65, flush=True)
    print("  FAST DIRECT HARVESTER: DJI-Mini & DJI-Neo", flush=True)
    print("=" * 65, flush=True)
    
    for class_name, query_list in QUERIES.items():
        print(f"\n[*] Harvesting images for [{class_name}]...", flush=True)
        all_urls = []
        for q in query_list:
            urls = search_bing_images(q, max_images=40)
            for u in urls:
                if u not in all_urls:
                    all_urls.append(u)
            print(f"  - Query '{q}': found {len(urls)} URLs (Total: {len(all_urls)})", flush=True)

        random.shuffle(all_urls)
        valid_images = []
        
        print(f"[*] Downloading and validating {class_name} images...", flush=True)
        for idx, u in enumerate(all_urls):
            img = download_image(u)
            if img is not None:
                valid_images.append(img)
                print(f"  [{len(valid_images)}/100] Verified valid image: {img.size[0]}x{img.size[1]}px", flush=True)
            if len(valid_images) >= 100:
                break

        print(f"[+] Total valid collected for [{class_name}]: {len(valid_images)}", flush=True)
        
        # Split into train (70%), val (15%), test (15%)
        random.shuffle(valid_images)
        total = len(valid_images)
        n_train = int(total * 0.70)
        n_val = int(total * 0.15)
        
        splits = {
            "train": valid_images[:n_train],
            "val": valid_images[n_train:n_train + n_val],
            "test": valid_images[n_train + n_val:]
        }
        
        for split_name, img_list in splits.items():
            dst_dir = DATA_DIR / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(img_list):
                save_file = dst_dir / f"{class_name}_{i+1:04d}.jpg"
                img.save(save_file, format="JPEG", quality=92)
                
        print(f"[+] Saved [{class_name}] to disk: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}", flush=True)

    print("\n" + "=" * 65, flush=True)
    print("  [COMPLETE] Dataset harvested and split successfully!", flush=True)
    print(f"  Dataset path: {DATA_DIR.resolve()}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
