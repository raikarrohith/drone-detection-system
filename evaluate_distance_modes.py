#!/usr/bin/env python3
"""
Drone Distance Estimation Benchmark & Error Analysis Tool:
Tests and compares 3 Monocular Estimation Modes:
1. Width-Only Estimator (W)
2. Height-Only Estimator (H)
3. Combined W+H Estimator (Fusion / Diagonal)

Calculates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Bias Error (ME)
- Maximum Error (%)
"""

import sys
import os
import cv2
import time
import math
import argparse
from pathlib import Path
import numpy as np

def run_distance_benchmark(cam_idx=1, drone_width=0.16, drone_height=0.05, focal_length_px=1350.0):
    """Interactive ground-truth benchmark across 0.5m, 1.0m, 1.5m, 2.0m."""
    from ultralytics import YOLO
    model = YOLO("models/best.pt")

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    test_distances = [0.5, 1.0, 1.5, 2.0]
    results_data = {
        "width": [],
        "height": [],
        "fusion": [],
        "ground_truth": []
    }

    print("=" * 70)
    print("  DRONE DISTANCE ESTIMATION BENCHMARK STUDIO")
    print(f"  - Physical Dimensions: Width = {drone_width*100:.1f}cm, Height = {drone_height*100:.1f}cm")
    print(f"  - Focal Length: F = {focal_length_px:.1f}px")
    print(f"  - Test Distances: {test_distances} meters")
    print("=" * 70)

    window_name = "Distance Benchmark Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for step, gt_d in enumerate(test_distances, 1):
        print(f"\n[*] STEP {step}/{len(test_distances)}: Place drone at EXACTLY {gt_d:.2f}m.")
        print("    Press [SPACE] to capture 50 benchmark samples...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            res = model(frame, conf=0.15, verbose=False)
            
            box_found = False
            cur_dw, cur_dh, cur_dwh = 0, 0, 0

            if res and len(res[0].boxes) > 0:
                best_box = max(res[0].boxes, key=lambda b: float(b.conf[0]))
                bx1, by1, bx2, by2 = map(int, best_box.xyxy[0])
                pw = max(2, bx2 - bx1)
                ph = max(2, by2 - by1)
                cur_dw = (drone_width * focal_length_px) / float(pw)
                cur_dh = (drone_height * focal_length_px) / float(ph)
                p_diag = math.sqrt(pw**2 + ph**2)
                t_diag = math.sqrt(drone_width**2 + drone_height**2)
                cur_dwh = (t_diag * focal_length_px) / float(p_diag)
                box_found = True

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            disp = frame.copy()
            cv2.rectangle(disp, (0, 0), (w, 100), (25, 25, 25), -1)
            cv2.putText(disp, f"BENCHMARK STEP {step}/{len(test_distances)}: TARGET = {gt_d:.2f} METERS", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            
            if box_found:
                cv2.putText(disp, f"LIVE -> Width: {cur_dw:.2f}m | Height: {cur_dh:.2f}m | W+H: {cur_dwh:.2f}m | Press [SPACE] to sample",
                            (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 255, 200), 2)
            else:
                cv2.putText(disp, "Aim camera at drone. Waiting for detection...",
                            (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 1)

            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in [32, 13] and box_found:
                break
            elif key in [27, ord('q'), ord('Q')]:
                cap.release()
                cv2.destroyAllWindows()
                return

        # Sample 50 consecutive frames at this ground truth distance
        samples_w = []
        samples_h = []
        samples_wh = []

        print(f"[*] Sampling {gt_d:.2f}m measurements...")
        while len(samples_w) < 40:
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame, conf=0.15, verbose=False)
            if res and len(res[0].boxes) > 0:
                best_box = max(res[0].boxes, key=lambda b: float(b.conf[0]))
                bx1, by1, bx2, by2 = map(int, best_box.xyxy[0])
                pw = max(2, bx2 - bx1)
                ph = max(2, by2 - by1)
                dw = (drone_width * focal_length_px) / float(pw)
                dh = (drone_height * focal_length_px) / float(ph)
                p_diag = math.sqrt(pw**2 + ph**2)
                t_diag = math.sqrt(drone_width**2 + drone_height**2)
                dwh = (t_diag * focal_length_px) / float(p_diag)

                samples_w.append(dw)
                samples_h.append(dh)
                samples_wh.append(dwh)

        results_data["width"].append(np.mean(samples_w))
        results_data["height"].append(np.mean(samples_h))
        results_data["fusion"].append(np.mean(samples_wh))
        results_data["ground_truth"].append(gt_d)
        print(f"  [+] Ground Truth: {gt_d:.2f}m -> Avg Width: {np.mean(samples_w):.2f}m | Avg Height: {np.mean(samples_h):.2f}m | Avg W+H: {np.mean(samples_wh):.2f}m")

    cap.release()
    cv2.destroyAllWindows()

    # Compute Statistical Accuracy Metrics
    gt = np.array(results_data["ground_truth"])
    w_est = np.array(results_data["width"])
    h_est = np.array(results_data["height"])
    wh_est = np.array(results_data["fusion"])

    mae_w = np.mean(np.abs(w_est - gt))
    mae_h = np.mean(np.abs(h_est - gt))
    mae_wh = np.mean(np.abs(wh_est - gt))

    rmse_w = np.sqrt(np.mean((w_est - gt)**2))
    rmse_h = np.sqrt(np.mean((h_est - gt)**2))
    rmse_wh = np.sqrt(np.mean((wh_est - gt)**2))

    print("\n" + "=" * 75)
    print("                      ACCURACY BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Distance (GT)':<15}{'Width Only':<18}{'Height Only':<18}{'Combined (W+H)':<18}")
    print("-" * 75)
    for g, w_v, h_v, wh_v in zip(gt, w_est, h_est, wh_est):
        print(f"{g:.2f}m           {w_v:.2f}m ({abs(w_v-g)*100:.1f}cm)    {h_v:.2f}m ({abs(h_v-g)*100:.1f}cm)    {wh_v:.2f}m ({abs(wh_v-g)*100:.1f}cm)")
    print("-" * 75)
    print(f"{'MAE (Mean Error)':<15}{mae_w*100:.1f} cm{'':<10}{mae_h*100:.1f} cm{'':<10}{mae_wh*100:.1f} cm")
    print(f"{'RMSE':<15}{rmse_w*100:.1f} cm{'':<10}{rmse_h*100:.1f} cm{'':<10}{rmse_wh*100:.1f} cm")
    print("=" * 75)

    best_mode = "Width Only"
    best_mae = mae_w
    if mae_wh < best_mae:
        best_mode = "Combined W+H (Fusion)"
        best_mae = mae_wh
    if mae_h < best_mae:
        best_mode = "Height Only"
        best_mae = mae_h

    print(f"\n[WINNER]: >> {best_mode.upper()} << achieved highest accuracy (MAE: {best_mae*100:.1f}cm)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Distance Estimation Benchmark")
    parser.add_argument("--camera", "-c", type=int, default=1, help="Camera index")
    parser.add_argument("--width", "-w", type=float, default=0.16, help="Physical drone wingspan in meters (default 0.16m for Neo)")
    parser.add_argument("--height", "-ht", type=float, default=0.05, help="Physical drone height in meters (default 0.05m for Neo)")
    parser.add_argument("--focal", "-f", type=float, default=1350.0, help="Camera focal length in pixels (default 1350px for 1080p)")
    args = parser.parse_args()

    run_distance_benchmark(
        cam_idx=args.camera,
        drone_width=args.width,
        drone_height=args.height,
        focal_length_px=args.focal
    )
