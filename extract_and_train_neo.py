#!/usr/bin/env python3
"""
DJI Neo Multi-Distance Training Pipeline:
1. Auto-detects Logitech 1080p / USB cameras.
2. Guided recording at 0.5m, 1.0m, 1.5m, and 2.0m distances (30s each), with single-distance re-recording support.
3. Automatic frame extraction with blur filtering.
4. Auto-annotation with YOLO detector + contour bounding box refinement.
5. Dataset assembly for both YOLO Detector and Airframe Classifier.
"""

import sys
import os
import cv2
import time
import shutil
import random
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def find_best_camera(preferred_idx=None, target_res="1080p"):
    """Auto-detects available cameras and picks Logitech 1080p / external webcam."""
    req_w, req_h = (1920, 1080) if target_res == "1080p" else (1280, 720)
    
    candidates = [preferred_idx] if preferred_idx is not None else [1, 0, 2, 3]
    for c_idx in candidates:
        if c_idx is None:
            continue
        cap = cv2.VideoCapture(c_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            cap.set(cv2.CAP_PROP_FPS, 30.0)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"[*] Found Active Camera Index [{c_idx}] running at {actual_w}x{actual_h}")
                return c_idx, actual_w, actual_h
            cap.release()

    return 0, 1280, 720

def extract_and_label_frames(video_path, output_dir="dataset_raw/drone_dataset", frame_step=3, model_path="models/best.pt"):
    """Extracts sharp frames and auto-annotates bounding boxes."""
    vid_file = Path(video_path)
    if not vid_file.is_file():
        return 0

    cap = cv2.VideoCapture(str(vid_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[*] Extracting frames from {vid_file.name} ({total_frames} frames)...")

    out_root = Path(output_dir)
    img_train = out_root / "images" / "train"
    lbl_train = out_root / "labels" / "train"
    img_val = out_root / "images" / "val"
    lbl_val = out_root / "labels" / "val"

    cls_neo_dir = Path("drone_images/DJI-Neo")
    cls_neo_dir.mkdir(parents=True, exist_ok=True)

    for d in [img_train, lbl_train, img_val, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path) if Path(model_path).is_file() else None

    frame_idx = 0
    saved_count = 0
    stem_prefix = vid_file.stem

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 20.0:
            continue

        h, w = frame.shape[:2]
        box_coords = None
        if model is not None:
            res = model(frame, conf=0.08, verbose=False)
            if res and len(res[0].boxes) > 0:
                best_box = max(res[0].boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                box_coords = (x1, y1, x2, y2)

        is_val = (random.random() < 0.15)
        target_img_dir = img_val if is_val else img_train
        target_lbl_dir = lbl_val if is_val else lbl_train

        base_name = f"neo_{stem_prefix}_{frame_idx:05d}"
        img_out_path = target_img_dir / f"{base_name}.jpg"
        lbl_out_path = target_lbl_dir / f"{base_name}.txt"

        cv2.imwrite(str(img_out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if box_coords:
            bx1, by1, bx2, by2 = box_coords
            bw = (bx2 - bx1) / float(w)
            bh = (by2 - by1) / float(h)
            bxc = (bx1 + bx2) / (2.0 * w)
            byc = (by1 + by2) / (2.0 * h)
            with open(lbl_out_path, "w") as f:
                f.write(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}\n")

            crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                cls_crop_path = cls_neo_dir / f"{base_name}_crop.jpg"
                cv2.imwrite(str(cls_crop_path), crop)

        saved_count += 1

    cap.release()
    return saved_count

def run_recording(target_distances, preferred_cam=None, duration=30, res="1080p"):
    """Records videos for the specified distances."""
    cam_idx, req_w, req_h = find_best_camera(preferred_idx=preferred_cam, target_res=res)
    vid_dir = Path("dataset_videos")
    vid_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[!] Error: Could not connect to camera index {cam_idx}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, 30.0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or req_w)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or req_h)
    fps = 30.0

    print("\n" + "=" * 65)
    print("  DJI NEO DATASET STUDIO (1080p)")
    print(f"  - Camera: Index [{cam_idx}] ({w}x{h} HD MJPG)")
    print(f"  - Target Distances: {target_distances}")
    print("  - [SPACE] to start / finish, or auto-advances at 0s")
    print("=" * 65)

    window_name = "DJI Neo Dataset Studio - Logitech 1080p"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    user_cancelled = False
    recorded_videos = []

    for idx, d in enumerate(target_distances, 1):
        vid_path = vid_dir / f"neo_{d:.1f}m.mp4"

        # Stage 1: Preview
        print(f"\n[*] STEP {idx}/{len(target_distances)}: Place drone at ~{d:.1f}m distance.")
        print("    Press [SPACE] when ready to record...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            disp = frame.copy()
            cv2.rectangle(disp, (0, 0), (w, 95), (25, 25, 25), -1)
            cv2.putText(disp, f"TARGET DISTANCE = {d:.1f} METERS ({w}x{h})", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(disp, f"Position drone at ~{d:.1f}m. Press [SPACE] to start recording.", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in [32, 13]:  # Space or Enter
                break
            elif key in [27, ord('q'), ord('Q')]:
                user_cancelled = True
                break

        if user_cancelled:
            print("[!] Cancelled by user.")
            break

        # Stage 2: Record
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))

        start_time = time.time()
        print(f"[*] Recording {d:.1f}m video (30s timer started)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            remaining = max(0.0, duration - elapsed)

            out.write(frame)

            disp = frame.copy()
            cv2.rectangle(disp, (0, 0), (w, 95), (0, 0, 160), -1)
            cv2.circle(disp, (35, 48), 14, (0, 0, 255), -1)
            cv2.putText(disp, f"RECORDING ({d:.1f}m) -- {remaining:.1f}s REMAINING", (65, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(disp, "Rotate drone 360 deg | Press [SPACE] to finish early & save", (65, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)

            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in [32, 13, ord('n'), ord('N')] and elapsed > 2.0:
                print(f"[+] Recording for {d:.1f}m finished by user.")
                break
            elif elapsed >= duration:
                print(f"[+] Recording for {d:.1f}m (30s) completed automatically.")
                break
            elif key in [27, ord('q'), ord('Q')]:
                user_cancelled = True
                break

        out.release()
        recorded_videos.append(vid_path)

        if user_cancelled:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Re-extract all available videos in dataset_videos/ to produce the complete dataset
    all_videos = sorted(list(vid_dir.glob("neo_*.mp4")))
    if all_videos:
        print(f"\n[*] Processing & auto-annotating all {len(all_videos)} video files ({', '.join(v.name for v in all_videos)})...")
        total_saved = 0
        for vid in all_videos:
            count = extract_and_label_frames(vid, frame_step=3)
            total_saved += count

        print("\n" + "=" * 65)
        print("  DATASET CAPTURE & AUTO-LABELING COMPLETE!")
        print(f"  - Total Sharp Frames in Dataset: {total_saved}")
        print(f"  - Detector Dataset: dataset_raw/drone_dataset/")
        print(f"  - Classifier Dataset: drone_images/DJI-Neo/")
        print("=" * 65)
        print("\nNext step: Run training to update your detector:")
        print("  python train_p2_detector.py --epochs 30 --batch-size 16\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DJI Neo Multi-Distance Dataset Studio")
    parser.add_argument("--distance", "--redo", "-d_redo", type=float, default=None, help="Re-record ONLY a specific distance (e.g. --distance 1.5)")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Camera index (auto-detects Logi webcam by default)")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Duration per distance in seconds (default: 30s)")
    parser.add_argument("--res", type=str, default="1080p", choices=["1080p", "720p"], help="Capture resolution (default: 1080p)")
    parser.add_argument("--video", "-v", type=str, default=None, help="Process a single existing video file")
    parser.add_argument("--extract-all", action="store_true", help="Re-extract and auto-label all existing videos in dataset_videos/")
    args = parser.parse_args()

    if args.extract_all:
        vid_dir = Path("dataset_videos")
        for vid in sorted(vid_dir.glob("neo_*.mp4")):
            extract_and_label_frames(vid, frame_step=3)
    elif args.video:
        extract_and_label_frames(args.video, frame_step=3)
    elif args.distance is not None:
        run_recording(target_distances=[args.distance], preferred_cam=args.camera, duration=args.duration, res=args.res)
    else:
        run_recording(target_distances=[0.5, 1.0, 1.5, 2.0], preferred_cam=args.camera, duration=args.duration, res=args.res)
