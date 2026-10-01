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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

TARGET_CLASSES = {
    "DJI-Phantom": [
        "DJI Phantom 4 Pro drone flying in air",
        "DJI Phantom 3 quadcopter flying sky",
        "DJI Phantom 4 drone photo",
        "white DJI Phantom quadcopter in flight",
        "DJI Phantom 4 RTK drone outdoors",
    ],
    "DJI-Mavic": [
        "DJI Mavic 3 Pro drone flying in air",
        "DJI Mavic 2 Pro quadcopter in flight",
        "DJI Mavic Air 2 drone flying outdoors",
        "DJI Mavic 3 Classic drone photo",
        "DJI Mavic enterprise drone flying",
    ],
    "DJI-Mini": [
        "DJI Mini 4 Pro drone flying in air",
        "DJI Mini 3 Pro quadcopter flying sky",
        "DJI Mini 2 drone in flight outdoors",
        "DJI Mavic Mini lightweight drone photo",
        "DJI Mini 3 drone flying outdoors",
    ],
    "DJI-Neo": [
        "DJI Neo drone flying in air",
        "DJI Neo palm drone hovering",
        "DJI Neo 4k mini quadcopter photo",
        "DJI Neo drone flying outdoors",
        "DJI Neo quadcopter hovering in air",
    ],
    "DJI-Avata": [
        "DJI Avata 2 FPV drone flying in air",
        "DJI Avata cinewhoop drone in flight",
        "DJI Avata FPV quadcopter photo",
        "DJI Avata 2 drone flying outdoors",
    ],
    "Parrot_Bebop": [
        "Parrot Bebop 2 drone flying in air",
        "Parrot Bebop 2 quadcopter in flight",
        "Parrot Bebop drone photo outdoors",
        "Parrot Bebop 2 red drone flying",
    ],
    "Yuneec-Typhoon": [
        "Yuneec Typhoon H hexacopter flying in air",
        "Yuneec Typhoon H Plus drone flying",
        "Yuneec Typhoon Q500 4K drone in flight",
        "Yuneec Typhoon drone photo outdoors",
    ],
    "Predator-Reaper": [
        "MQ-9 Reaper drone flying in sky",
        "General Atomics MQ-9 Reaper UAV in flight",
        "MQ-1 Predator military drone in air",
        "MQ-9 Reaper drone high altitude",
    ],
    "RQ11-Raven": [
        "RQ-11 Raven drone flying in sky",
        "AeroVironment RQ-11 Raven UAV in flight",
        "RQ-11 Raven small hand launched drone",
        "RQ-11 Raven military drone in air",
    ],
    "RQ4-GlobalHawk": [
        "RQ-4 Global Hawk drone flying in sky",
        "Northrop Grumman RQ-4 Global Hawk UAV in air",
        "RQ-4 Global Hawk high altitude surveillance drone",
        "RQ-4 Global Hawk military aircraft flying",
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
            time.sleep(0.15)
        except Exception:
            pass
        if len(urls) >= max_images:
            break
    return urls

def download_and_verify_single(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
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

def harvest_and_build_dataset(target_per_class=80):
    print("=" * 70, flush=True)
    print("  10-CLASS UNIFIED DRONE DATASET HARVESTER", flush=True)
    print(f"  Target: {target_per_class} verified images per class ({len(TARGET_CLASSES)} classes total)", flush=True)
    print("=" * 70, flush=True)

    # Clean previous dataset
    if DATA_DIR.exists():
        print(f"[*] Refreshing dataset directory: {DATA_DIR}...", flush=True)
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for idx, (class_name, queries) in enumerate(TARGET_CLASSES.items(), 1):
        print(f"\n[{idx}/{len(TARGET_CLASSES)}] Harvesting [{class_name}]...", flush=True)
        all_urls = []
        for q in queries:
            urls = search_bing_images(q, max_images=50)
            for u in urls:
                if u not in all_urls:
                    all_urls.append(u)
        
        print(f"  -> Discovered {len(all_urls)} image URLs. Downloading in parallel...", flush=True)
        random.shuffle(all_urls)

        valid_images = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(download_and_verify_single, url): url for url in all_urls}
            for future in as_completed(future_to_url):
                res = future.result()
                if res is not None:
                    valid_images.append(res)
                if len(valid_images) >= target_per_class:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        print(f"  [+] Downloaded and verified {len(valid_images)} images for [{class_name}].", flush=True)

        # Train (70%), Val (15%), Test (15%)
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

        summary[class_name] = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": total
        }

    print("\n" + "=" * 70, flush=True)
    print("  HARVEST COMPLETE SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for cname, counts in summary.items():
        print(f"  - {cname:<16}: Train={counts['train']:<3} | Val={counts['val']:<3} | Test={counts['test']:<3} | Total={counts['total']}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    harvest_and_build_dataset(target_per_class=80)
