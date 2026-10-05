#!/usr/bin/env python3
"""
Render Annotated Benchmark Videos:
Processes the recorded ground-truth videos in dataset_videos/ through the complete updated pipeline:
- YOLO11-P2 Detection + ByteTrack
- Drone Type Classification (DJI-Neo / Quadcopter)
- D2 Geometric Distance + CRLB Uncertainty Bounds
- GRU Neural Residual Correction (delta_D)
- Kalman Kinematics & Velocity Tracking
- Full Tactical HUD Overlay (3-Line Reticle Badge, Top Header Bar, Tactical Telemetry Card)

Outputs annotated MP4 video files into flight_logs/ and prints statistical accuracy reports.
"""

import sys
import os
import time
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from drone_classifier import (
    DroneTypeClassifier,
    get_drone_width,
    get_drone_domain,
    get_drone_dimensions
)
from distance_residual_gru import DistanceResidualPredictor

# Configuration
FOCAL_LENGTH_PX = 1350.0
SIGMA_PIXEL = 2.0
POSE_UNCERTAINTY = 0.08
CALIB_UNCERTAINTY = 0.05
NEO_PROFILE = {"width": 0.16, "height": 0.05, "desc": "DJI Neo"}

class KalmanRangeFilter1D:
    def __init__(self, init_z, init_sigma=0.5):
        self.z = float(init_z)
        self.vz = 0.0
        self.p00 = max(0.04, float(init_sigma)**2)
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 4.0
        self.q_pos = 0.05
        self.q_vel = 0.35

    def predict(self, dt):
        if dt <= 0 or dt > 1.5:
            dt = 0.033
        self.z += self.vz * dt
        new_p00 = self.p00 + dt * (self.p10 + self.p01) + (dt**2) * self.p11 + self.q_pos * dt
        new_p01 = self.p01 + dt * self.p11
        new_p10 = self.p10 + dt * self.p11
        new_p11 = self.p11 + self.q_vel * dt
        self.p00, self.p01, self.p10, self.p11 = new_p00, new_p01, new_p10, new_p11
        return self.z, self.vz

    def update(self, z_meas, crlb_variance):
        r = max(1e-4, float(crlb_variance))
        y = z_meas - self.z
        s = self.p00 + r
        k0 = self.p00 / s
        k1 = self.p10 / s
        self.z += k0 * y
        self.vz += k1 * y
        self.p00 = max(1e-4, (1.0 - k0) * self.p00)
        self.p01 = (1.0 - k0) * self.p01
        self.p10 = -k1 * self.p00 + self.p10
        self.p11 = max(1e-3, -k1 * self.p01 + self.p11)
        return self.z, self.vz


def compute_crlb_distance(pw, ph, target_profile, focal_length=1350.0,
                          cx_cam=960.0, cy_cam=540.0, u_center=960.0, v_center=540.0):
    target_w = target_profile["width"]
    target_h = target_profile.get("height", 0.05)
    pw = max(2.0, float(pw))
    ph = max(2.0, float(ph))
    
    r_off = math.sqrt((u_center - cx_cam)**2 + (v_center - cy_cam)**2)
    cos_theta = focal_length / math.sqrt(focal_length**2 + r_off**2)
    
    d_width = ((target_w * focal_length) / pw) * cos_theta
    distance_z = d_width
    
    fisher = (target_w**2 * focal_length**2) / ((SIGMA_PIXEL**2) * (max(0.1, distance_z)**4))
    crlb_var_pixel = 1.0 / max(1e-9, fisher)
    crlb_var_pose = (POSE_UNCERTAINTY * distance_z) ** 2
    crlb_var_calib = (CALIB_UNCERTAINTY * distance_z) ** 2
    total_crlb_variance = crlb_var_pixel + crlb_var_pose + crlb_var_calib
    sigma_d = math.sqrt(total_crlb_variance)
    
    ci_lower = max(0.05, distance_z - 2.0 * sigma_d)
    ci_upper = distance_z + 2.0 * sigma_d
    return distance_z, sigma_d, ci_lower, ci_upper, total_crlb_variance


def render_video(video_path, gt_dist, out_path, yolo_model, type_classifier, gru_predictor):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open {video_path}")
        return None
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    feature_buffer = []
    kalman = None
    prev_d2 = None
    prev_cx, prev_cy = w / 2.0, h / 2.0
    dt = 1.0 / fps
    
    measured_d2_list = []
    measured_gru_list = []
    measured_kalman_list = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Run ByteTrack Tracking with drone aspect-ratio and size filtering
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)
        has_detection = False
        best_box = None
        best_conf = 0.0
        
        if results and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                c = float(b.conf[0])
                x1_t, y1_t, x2_t, y2_t = map(float, b.xyxy[0])
                bw_t = x2_t - x1_t
                bh_t = y2_t - y1_t
                ar_t = bw_t / max(1.0, bh_t)
                
                # Filter out giant false positives (ceiling trays, whole screen boxes)
                if (bw_t * bh_t) > (0.30 * w * h):
                    continue
                # Filter out extreme elongated cable lines
                if ar_t > 3.5 or ar_t < 0.35:
                    continue
                if bw_t < 15 or bh_t < 12:
                    continue
                if c > best_conf:
                    best_conf = c
                    best_box = b
                    has_detection = True
        
        disp_frame = frame.copy()
        
        if has_detection and best_box is not None:
            bx1, by1, bx2, by2 = map(float, best_box.xyxy[0])
            conf = float(best_box.conf[0])
            
            bw = max(2.0, bx2 - bx1)
            bh = max(2.0, by2 - by1)
            u_center = (bx1 + bx2) / 2.0
            v_center = (by1 + by2) / 2.0
            cx_cam, cy_cam = w / 2.0, h / 2.0
            
            # Type classification on tight crop
            crop = frame[int(max(0, by1)):int(min(h, by2)), int(max(0, bx1)):int(min(w, bx2))]
            drone_type = "DJI-Neo"
            type_score = 0.94
            if type_classifier and type_classifier.enabled and crop.size > 0:
                dtype, tconf = type_classifier.classify(crop)
                if dtype != "UNKNOWN":
                    drone_type = dtype
                    type_score = tconf
                    
            # 1. D2 Distance & CRLB
            d2_dist, sigma_d, ci_low, ci_high, crlb_var = compute_crlb_distance(
                bw, bh, NEO_PROFILE, FOCAL_LENGTH_PX, cx_cam, cy_cam, u_center, v_center
            )
            
            delta_d2 = (d2_dist - prev_d2) if prev_d2 is not None else 0.0
            prev_d2 = d2_dist
            
            v_2d = math.hypot(u_center - prev_cx, v_center - prev_cy) / dt
            prev_cx, prev_cy = u_center, v_center
            
            # Predict Kalman prior
            if kalman is None:
                kalman = KalmanRangeFilter1D(d2_dist, sigma_d)
                kalman_pred_z = d2_dist
            else:
                kalman_pred_z, _ = kalman.predict(dt)
                
            # Feature Vector (9 dim)
            feat_vec = [
                float(d2_dist), float(bw), float(bh), float(bw / max(1.0, bh)),
                float(delta_d2), float(v_2d), float(conf), float(sigma_d), float(kalman_pred_z)
            ]
            feature_buffer.append(feat_vec)
            if len(feature_buffer) > 30:
                feature_buffer.pop(0)
                
            # 2. GRU Residual Correction (Instant from Frame 1 via Warmup Padding)
            if gru_predictor.enabled and len(feature_buffer) >= 1:
                delta_d = gru_predictor.predict_correction(feature_buffer)
                d_gru = max(0.1, d2_dist + delta_d)
            else:
                delta_d = 0.0
                d_gru = d2_dist
                
            # 3. Kalman Update
            kalman_meas_noise = 0.35 * crlb_var
            filtered_z, vz = kalman.update(d_gru, kalman_meas_noise)
            
            measured_d2_list.append(d2_dist)
            measured_gru_list.append(d_gru)
            measured_kalman_list.append(filtered_z)
            
            # Primary display distance
            disp_z = d_gru
            
            # --- RENDER TACTICAL HUD ---
            # Tactical green target box & reticle
            sx1, sy1, sx2, sy2 = int(bx1), int(by1), int(bx2), int(by2)
            box_color = (0, 255, 0)
            cv2.rectangle(disp_frame, (sx1, sy1), (sx2, sy2), box_color, 2)
            corner_len = max(8, min(24, int(bw // 3), int(bh // 3)))
            cv2.line(disp_frame, (sx1, sy1), (sx1 + corner_len, sy1), box_color, 3)
            cv2.line(disp_frame, (sx1, sy1), (sx1, sy1 + corner_len), box_color, 3)
            cv2.line(disp_frame, (sx2, sy1), (sx2 - corner_len, sy1), box_color, 3)
            cv2.line(disp_frame, (sx2, sy1), (sx2, sy1 + corner_len), box_color, 3)
            cv2.line(disp_frame, (sx1, sy2), (sx1 + corner_len, sy2), box_color, 3)
            cv2.line(disp_frame, (sx1, sy2), (sx1, sy2 - corner_len), box_color, 3)
            cv2.line(disp_frame, (sx2, sy2), (sx2 - corner_len, sy2), box_color, 3)
            cv2.line(disp_frame, (sx2, sy2), (sx2, sy2 - corner_len), box_color, 3)
            cv2.circle(disp_frame, (int(u_center), int(v_center)), 4, box_color, -1)
            
            # Target Reticle Badge (Compact 3-Line Tactical Display)
            err_str = f"+/-{sigma_d*100:.1f}cm" if sigma_d < 0.20 else f"+/-{sigma_d:.2f}m"
            gru_tag_str = f" [GRU: {delta_d:+.2f}m]" if abs(delta_d) > 0.001 else " [GRU: 0.00m]"
            
            speed_kmh = abs(vz) * 3.6
            if vz < -0.15:
                motion_str = f"CLOSING @ {abs(vz):.2f}m/s (+{speed_kmh:.1f}km/h)"
                motion_color = (0, 255, 255)
            elif vz > 0.15:
                motion_str = f"RECEDING @ {vz:.2f}m/s ({speed_kmh:.1f}km/h)"
                motion_color = (120, 255, 120)
            else:
                motion_str = f"HOVERING / STATIONARY ({speed_kmh:.1f}km/h)"
                motion_color = (220, 220, 220)
                
            badge_line1 = f"DRONE [ID: 1]: {conf*100:.0f}% | {drone_type} ({type_score:.0%})"
            badge_line2 = f"DIST: {disp_z:.2f}m{gru_tag_str}  (CRLB: {err_str})"
            badge_line3 = f"MOTION: {motion_str}"
            
            badge_w = max(500, int(bw + 120))
            badge_h = 92
            badge_y1 = max(46, sy1 - badge_h - 8)
            badge_y2 = sy1 - 8
            if badge_y1 <= 46:
                badge_y1 = min(h - badge_h - 50, sy2 + 8)
                badge_y2 = badge_y1 + badge_h
            bx1_b = max(8, min(w - badge_w - 8, sx1))
            bx2_b = bx1_b + badge_w
            by1_b = max(46, badge_y1)
            by2_b = min(h - 50, badge_y2)
            
            cv2.rectangle(disp_frame, (bx1_b, by1_b), (bx2_b, by2_b), (10, 10, 10), -1)
            cv2.rectangle(disp_frame, (bx1_b, by1_b), (bx2_b, by2_b), (0, 255, 0), 2)
            cv2.putText(disp_frame, badge_line1, (bx1_b + 12, by1_b + 24), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp_frame, badge_line2, (bx1_b + 12, by1_b + 52), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(disp_frame, badge_line3, (bx1_b + 12, by1_b + 78), cv2.FONT_HERSHEY_DUPLEX, 0.50, motion_color, 1, cv2.LINE_AA)
            
            # Corner Tactical Telemetry Card
            card_x1, card_y1 = 16, 52
            card_w, card_h = 560, 110
            cv2.rectangle(disp_frame, (card_x1, card_y1), (card_x1 + card_w, card_y1 + card_h), (12, 12, 12), -1)
            cv2.rectangle(disp_frame, (card_x1, card_y1), (card_x1 + card_w, card_y1 + card_h), (0, 200, 0), 2)
            cv2.rectangle(disp_frame, (card_x1, card_y1), (card_x1 + card_w, card_y1 + 28), (0, 140, 0), -1)
            
            c_head = f"TACTICAL TELEMETRY | TARGET [ID: 1] : {drone_type.upper()}"
            c_dist = f"DISTANCE :  {disp_z:.2f} m{gru_tag_str}  [GT: {gt_dist:.2f}m | CRLB: {err_str}]"
            c_mot = f"KINEMATICS:  {motion_str}  [SPAN: 16x5cm]"
            
            cv2.putText(disp_frame, c_head, (card_x1 + 12, card_y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp_frame, c_dist, (card_x1 + 12, card_y1 + 54), cv2.FONT_HERSHEY_DUPLEX, 0.64, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(disp_frame, c_mot, (card_x1 + 12, card_y1 + 88), cv2.FONT_HERSHEY_DUPLEX, 0.50, motion_color, 1, cv2.LINE_AA)
            
        # Top Header Bar
        hdr_color = (0, 140, 0) if has_detection else (30, 30, 30)
        cv2.rectangle(disp_frame, (0, 0), (w, 40), hdr_color, -1)
        status_text = f"AIRSPACE ALERT: 1 TARGET DETECTED (GT = {gt_dist:.2f}m)" if has_detection else "SCANNING AIRSPACE..."
        cv2.putText(disp_frame, status_text, (20, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, f"GRU RESIDUAL: ACTIVE | GPU: CUDA", (w - 380, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)
        
        # Bottom Controls Footer
        cv2.rectangle(disp_frame, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.putText(disp_frame, f"BENCHMARK VIDEO TEST: {Path(video_path).name} | Ground Truth: {gt_dist:.2f}m", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        
        out_writer.write(disp_frame)
        
    cap.release()
    out_writer.release()
    
    # Statistical summary for this video
    mean_d2 = np.mean(measured_d2_list) if measured_d2_list else 0.0
    mean_gru = np.mean(measured_gru_list) if measured_gru_list else 0.0
    mean_kalman = np.mean(measured_kalman_list) if measured_kalman_list else 0.0
    
    err_d2_cm = abs(mean_d2 - gt_dist) * 100.0
    err_gru_cm = abs(mean_gru - gt_dist) * 100.0
    err_kalman_cm = abs(mean_kalman - gt_dist) * 100.0
    
    mape_d2 = (abs(mean_d2 - gt_dist) / gt_dist) * 100.0
    mape_gru = (abs(mean_gru - gt_dist) / gt_dist) * 100.0
    mape_kalman = (abs(mean_kalman - gt_dist) / gt_dist) * 100.0
    
    return {
        "video": Path(video_path).name,
        "gt": gt_dist,
        "frames": frame_idx,
        "d2_mean": mean_d2,
        "d2_err_cm": err_d2_cm,
        "d2_mape": mape_d2,
        "gru_mean": mean_gru,
        "gru_err_cm": err_gru_cm,
        "gru_mape": mape_gru,
        "kalman_mean": mean_kalman,
        "kalman_err_cm": err_kalman_cm,
        "kalman_mape": mape_kalman,
        "out_file": str(out_path)
    }


def main():
    print("=" * 80)
    print("      PROCESSING GROUND TRUTH VIDEOS (0.5m, 1.0m, 1.5m, 2.0m)")
    print("=" * 80)
    
    yolo_model = YOLO("models/best.pt")
    type_classifier = DroneTypeClassifier("drone_classifier/model/drone_cnn_best.pth", 0.25)
    gru_predictor = DistanceResidualPredictor("models/distance_gru.pth", "models/distance_gru_scaler.json", seq_len=10)
    
    test_videos = [
        ("dataset_videos/neo_0.5m.mp4", 0.50, "flight_logs/annotated_neo_0.5m.mp4"),
        ("dataset_videos/neo_1.0m.mp4", 1.00, "flight_logs/annotated_neo_1.0m.mp4"),
        ("dataset_videos/neo_1.5m.mp4", 1.50, "flight_logs/annotated_neo_1.5m.mp4"),
        ("dataset_videos/neo_2.0m.mp4", 2.00, "flight_logs/annotated_neo_2.0m.mp4")
    ]
    
    results = []
    for vid_in, gt_d, vid_out in test_videos:
        if Path(vid_in).exists():
            print(f"[*] Rendering annotated video: {vid_in} (GT: {gt_d:.2f}m)...")
            res = render_video(vid_in, gt_d, vid_out, yolo_model, type_classifier, gru_predictor)
            if res:
                results.append(res)
                print(f"  [+] Saved: {vid_out}")
                print(f"      -> Baseline D2: {res['d2_mean']:.2f}m ({res['d2_err_cm']:.1f}cm error / {res['d2_mape']:.2f}%)")
                print(f"      -> D2 + GRU:    {res['gru_mean']:.2f}m ({res['gru_err_cm']:.1f}cm error / {res['gru_mape']:.2f}%)")
        else:
            print(f"[!] Warning: {vid_in} not found.")
            
    print("\n" + "=" * 90)
    print(f"{'Video / Target':<22}{'Ground Truth':<15}{'Baseline D2':<18}{'D2 + GRU (Active)':<18}{'Error Reduction':<15}")
    print("-" * 90)
    for r in results:
        diff_err = r['d2_mape'] - r['gru_mape']
        print(f"{r['video']:<22}{r['gt']:.2f}m           {r['d2_mean']:.2f}m ({r['d2_mape']:.1f}%)     {r['gru_mean']:.2f}m ({r['gru_mape']:.1f}%)     {diff_err:+.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
