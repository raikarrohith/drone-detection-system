import os
import re
import time
import shutil
import random
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = Path("drone_classifier/dataset")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

QUERIES = {
    "DJI-Mini": [
        "DJI Mini 3 Pro drone",
        "DJI Mini 4 Pro drone",
        "DJI Mini 2 quadcopter flying",
        "DJI Mavic Mini drone in air",
        "DJI Mini 3 drone photo",
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

def search_bing_images(query, max_images=60):
    urls = []
    for first in range(0, max_images + 30, 30):
        url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&first={first}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                found = re.findall(r'murl&quot;:&quot;(.*?)&quot;', resp.text)
                for u in found:
                    if u not in urls:
                        urls.append(u)
            time.sleep(0.2)
        except Exception:
            pass
        if len(urls) >= max_images:
            break
    return urls

def download_and_verify_single(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=4)
        if resp.status_code == 200 and len(resp.content) > 4000:
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
    print("  HIGH-SPEED PARALLEL HARVESTER: DJI-Mini & DJI-Neo", flush=True)
    print("=" * 65, flush=True)
    
    for class_name, query_list in QUERIES.items():
        print(f"\n[*] Searching URLs for [{class_name}]...", flush=True)
        all_urls = []
        for q in query_list:
            urls = search_bing_images(q, max_images=45)
            for u in urls:
                if u not in all_urls:
                    all_urls.append(u)
        
        print(f"[+] Total {len(all_urls)} URLs found for [{class_name}]. Downloading in parallel...", flush=True)
        random.shuffle(all_urls)
        
        valid_images = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_url = {executor.submit(download_and_verify_single, url): url for url in all_urls}
            for future in as_completed(future_to_url):
                res = future.result()
                if res is not None:
                    valid_images.append(res)
                    print(f"  [{len(valid_images)}/100] Verified image for {class_name}: {res.size[0]}x{res.size[1]}px", flush=True)
                if len(valid_images) >= 100:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
        print(f"[+] Completed collection for [{class_name}]: {len(valid_images)} images.", flush=True)
        
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
    print("  [SUCCESS] All images downloaded, verified, and split!", flush=True)
    print(f"  Dataset path: {DATA_DIR.resolve()}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
