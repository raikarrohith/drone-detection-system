#!/usr/bin/env python3
"""Ingest and organize custom civilian/military datasets into the classifier format.

Usage:
    python ingest_custom_dataset.py --source /path/to/downloaded/images --type civilian
    python ingest_custom_dataset.py --source /path/to/downloaded/images --type military
    python ingest_custom_dataset.py --source /path/to/downloaded/images --type unknown
"""

import argparse
import hashlib
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def looks_like_image(path: Path) -> bool:
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
        or header.startswith(b"BM")
    )


def choose_split(image: Path, val_ratio: float = 0.20) -> str:
    digest = hashlib.sha256(image.read_bytes()).hexdigest()[:8]
    val = int(digest, 16) / 0xFFFFFFFF
    return "val" if val < val_ratio else "train"


def main():
    parser = argparse.ArgumentParser(description="Ingest civilian/military/unknown images into classifier dataset")
    parser.add_argument("--source", type=Path, required=True, help="Path to raw image folder or zip extract")
    parser.add_argument("--type", type=str, required=True, choices=["civilian", "military", "unknown"], help="Classification target label")
    parser.add_argument("--output", type=Path, default=Path("dataset_raw/drone_type_classifier"), help="Destination dataset folder")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation split ratio (default 0.20)")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"[!] Source folder does not exist: {args.source}")
        return

    added = {"train": 0, "val": 0}
    skipped = 0

    for file_path in sorted(args.source.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not looks_like_image(file_path):
            skipped += 1
            continue

        split = choose_split(file_path, args.val_ratio)
        img_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:12]
        safe_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)[:60]
        dest_filename = f"{args.type}_{safe_stem}_{img_hash}{file_path.suffix.lower()}"
        
        dest_dir = args.output / split / args.type
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / dest_filename

        if not dest_path.exists():
            shutil.copy2(file_path, dest_path)
            added[split] += 1

    print(f"[+] Successfully ingested into '{args.type}':")
    print(f"    - Train split: {added['train']} images")
    print(f"    - Val split:   {added['val']} images")
    if skipped > 0:
        print(f"    - Skipped invalid headers: {skipped}")
    print(f"[+] Total images now in {args.output / 'train' / args.type}: {len(list((args.output / 'train' / args.type).glob('*')) if (args.output / 'train' / args.type).exists() else 0)}")


if __name__ == "__main__":
    main()
