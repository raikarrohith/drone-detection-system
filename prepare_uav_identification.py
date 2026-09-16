#!/usr/bin/env python3
"""Prepare UAV-Identification images for the drone-type classifier.

This never modifies the downloaded source dataset. It copies images into the
classifier layout and records every decision in a CSV manifest for review.
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


AIRFRAME_LABELS = {
    "DJI-Mavic": "civilian",
    "DJI-Phantom": "civilian",
    "Parrot_Bebop": "civilian",
    "Yuneec-Typhoon": "civilian",
    "RQ7-Shadow": "military",
    "RQ4-GlobalHawk": "military",
    "RQ11-Raven": "military",
    "Predator-Reaper": "military",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def looks_like_image(path):
    """Reject files with an image extension but no common image header."""
    try:
        header = path.read_bytes()[:16]
    except OSError:
        return False
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith((b"GIF87a", b"GIF89a"))
        or header.startswith(b"RIFF") and header[8:12] == b"WEBP"
        or header.startswith(b"BM")
    )


def output_name(airframe, image):
    digest = hashlib.sha256(image.read_bytes()).hexdigest()[:12]
    safe_stem = "".join(char if char.isalnum() else "_" for char in image.stem)[:70]
    return f"{airframe}_{safe_stem}_{digest}{image.suffix.lower()}"


def choose_split(image, validation_ratio):
    # Stable content-based split makes rebuilds reproducible.
    value = int(hashlib.sha256(image.read_bytes()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < validation_ratio else "train"


def main():
    parser = argparse.ArgumentParser(description="Prepare UAV-Identification classifier data")
    parser.add_argument("--source", type=Path, default=Path("datasets/UAV-Identification/extracted"))
    parser.add_argument("--output", type=Path, default=Path("dataset_raw/drone_type_classifier"))
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    args = parser.parse_args()

    if not 0.05 <= args.validation_ratio <= 0.40:
        parser.error("--validation-ratio must be between 0.05 and 0.40")
    if not args.source.is_dir():
        parser.error(f"Source folder not found: {args.source}")

    manifest_path = args.output / "uav_identification_manifest.csv"
    copied = {"civilian": 0, "military": 0}
    skipped = 0
    rows = []

    for airframe, label in AIRFRAME_LABELS.items():
        airframe_dir = args.source / airframe
        if not airframe_dir.is_dir():
            print(f"[!] Missing airframe folder: {airframe_dir}")
            continue
        for image in sorted(airframe_dir.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not looks_like_image(image):
                skipped += 1
                continue
            split = choose_split(image, args.validation_ratio)
            destination = args.output / split / label / output_name(airframe, image)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(image, destination)
                copied[label] += 1
            rows.append({
                "source": str(image), "destination": str(destination), "split": split,
                "label": label, "airframe": airframe, "review_status": "needs_review",
            })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=["source", "destination", "split", "label", "airframe", "review_status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Copied: {copied['civilian']} civilian, {copied['military']} military images")
    print(f"Skipped {skipped} files with invalid image headers")
    print(f"Manifest: {manifest_path}")
    print("All copied images are marked needs_review; remove bad candidates before training.")


if __name__ == "__main__":
    main()
