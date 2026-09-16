import sys
import os
import time
import math
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. PARSE ARGUMENTS & LOAD YOLO MODEL
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. PARSE ARGUMENTS & LOAD YOLO MODEL WITH ACCELERATION
# ---------------------------------------------------------
import torch

parser = argparse.ArgumentParser(description="Drone Defense & CRLB Distance Estimation System")
parser.add_argument("--video", "-v", type=str, default=None, help="Path to test video file (e.g. drone_test.mp4)")
parser.add_argument("--camera", "-c", type=int, default=None, help="Camera index (e.g. 0, 1, 2)")
parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution: 960 (balanced long-range) or 1280/640")
parser.add_argument("video_pos", nargs="?", default=None, help="Positional video file path")
args, _ = parser.parse_known_args()

video_source = args.video if args.video else args.video_pos
current_imgsz = args.imgsz


# Auto-detect best hardware device (NVIDIA CUDA, Apple Silicon MPS, or multi-threaded CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("[*] Hardware Acceleration: NVIDIA CUDA GPU (High Performance)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("[*] Hardware Acceleration: Apple Silicon Metal GPU (High Performance)")
else:
    DEVICE = "cpu"
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"[*] Hardware Acceleration: Multi-Threaded CPU ({torch.get_num_threads()} threads)")

MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

DRONE_CLASS_ID = 0
for cls_id, name in model.names.items():
    if "drone" in name.lower():
        DRONE_CLASS_ID = cls_id
        break

# ---------------------------------------------------------
# 2. SOURCE SETUP (Zero-Latency Live Camera & Video Stream)
# ---------------------------------------------------------
cap = None
is_video_file = False
video_filename = ""
is_paused = False

if video_source and os.path.isfile(video_source):
    cap = cv2.VideoCapture(video_source)
    if cap.isOpened():
        is_video_file = True
        video_filename = os.path.basename(video_source)
        print(f"[*] Testing with Video File: {video_source}")
    else:
        print(f"ERROR: Could not open video file {video_source}")

if cap is None or not cap.isOpened():
    is_windows = sys.platform.startswith("win")
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY

    target_cams = [args.camera] if args.camera is not None else [1, 2, 0, 3]
    for cam_idx in target_cams:
        temp_cap = cv2.VideoCapture(cam_idx, backend)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            # Hardware acceleration: Zero-latency buffer & fast MJPEG codec
            try:
                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                temp_cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
                print(f"[*] Successfully connected to Live Camera (index {cam_idx}) at Zero-Latency Buffer")
                break
            else:
                temp_cap.release()

if cap is None or not cap.isOpened():
    print("ERROR: Could not open camera or video file. Please check connections or video path.")
    exit(1)

WINDOW_NAME = "Drone Defense System - CRLB Range & Kinematics Engine"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ---------------------------------------------------------
# 3. CRAMER-RAO LOWER BOUND (CRLB) & PHYSICAL TARGET MODELS
# ---------------------------------------------------------
# Camera Optical Model: Focal length in pixels (calibrated for standard ~80 deg HFOV at 1080p/720p)
FOCAL_LENGTH_PX = 850.0    # Updated dynamically based on frame width
SIGMA_PIXEL = 2.5          # Realistic YOLO bounding box edge regression noise std-dev (pixels)
POSE_ASPECT_RATIO_UNCERTAINTY = 0.045  # 4.5% standard error due to 3D drone yaw/pitch rotation


# Drone Physical Size Profiles (Wingspan in meters)
DRONE_PROFILES = {
    1: {"name": "Micro/Mini", "width": 0.24, "desc": "DJI Mini / Avata (24cm)"},
    2: {"name": "Standard Quad", "width": 0.38, "desc": "Mavic / Phantom / FPV (38cm) [DEFAULT]"},
    3: {"name": "Heavy Lift", "width": 0.75, "desc": "Matrice / Hexacopter (75cm)"},
    4: {"name": "Tactical Wing", "width": 1.40, "desc": "Fixed-Wing / Loitering (140cm)"}
}
active_profile_id = 2
target_nominal_width = DRONE_PROFILES[active_profile_id]["width"]

def compute_crlb_distance(pixel_w, pixel_h, target_w, focal_length, sigma_w):
    """
    Computes Deterministic Monocular Distance and the theoretical
    Cramer-Rao Lower Bound (CRLB) / Fisher Information bounds.
    
    Measurement Model: w = (W * F) / D + epsilon, where epsilon ~ N(0, sigma_w^2)
    Sensitivity Derivative: dw/dD = -(W * F) / D^2
    Fisher Information I(D) = (1 / sigma_w^2) * (dw/dD)^2 = (W^2 * F^2) / (sigma_w^2 * D^4)
    CRLB(D) = 1 / I(D) = (sigma_w^2 * D^4) / (W^2 * F^2)
    Minimum Standard Error: sigma_D = sqrt(CRLB_pixel + sigma_pose^2)
    """
    pixel_w = max(2.0, float(pixel_w))
    # Estimated Distance (Z in meters)
    distance_z = (target_w * focal_length) / pixel_w
    
    # Fisher Information & CRLB Variance from Pixel Localization Noise
    fisher_info_pixel = (target_w**2 * focal_length**2) / ((sigma_w**2) * (distance_z**4))
    crlb_var_pixel = 1.0 / max(1e-9, fisher_info_pixel)
    
    # Combined variance with 3D pose/aspect angle uncertainty
    crlb_var_pose = (POSE_ASPECT_RATIO_UNCERTAINTY * distance_z) ** 2
    total_crlb_variance = crlb_var_pixel + crlb_var_pose
    sigma_d = math.sqrt(total_crlb_variance)
    
    # 95% Confidence Interval (2-sigma theoretical bound)
    ci_lower = max(0.1, distance_z - 2.0 * sigma_d)
    ci_upper = distance_z + 2.0 * sigma_d
    
    return distance_z, sigma_d, ci_lower, ci_upper, total_crlb_variance

# ---------------------------------------------------------
# 4. KINEMATIC TRACKER MEMORY (3D State, Velocity, History)
# ---------------------------------------------------------
class TargetKinematics:
    def __init__(self, track_id):
        self.track_id = track_id
        self.history = []  # [(timestamp, x, y, z, dist, sigma_d)]
        self.hits = 0
        self.last_frame = 0
        self.smoothed_z = None
        self.velocity_3d = (0.0, 0.0, 0.0)  # vx, vy, vz in m/s
        self.speed_kmh = 0.0
        self.approach_rate = 0.0  # m/s (+ approaching, - receding)
        self.eta_seconds = None

    def update(self, frame_num, timestamp, x_m, y_m, z_m, sigma_d):
        self.hits += 1
        self.last_frame = frame_num

        # Exponential Moving Average for jitter reduction
        if self.smoothed_z is None:
            self.smoothed_z = z_m
        else:
            alpha = 0.35  # Smoothing factor
            self.smoothed_z = alpha * z_m + (1.0 - alpha) * self.smoothed_z

        self.history.append((timestamp, x_m, y_m, self.smoothed_z, sigma_d))
        if len(self.history) > 30:
            self.history.pop(0)

        # Calculate 3D Velocity & Approach Rate if we have enough temporal baseline (>= 0.15s)
        if len(self.history) >= 4:
            t_old, x_old, y_old, z_old, _ = self.history[-4]
            dt = timestamp - t_old
            if dt > 0.05:
                vx = (x_m - x_old) / dt
                vy = (y_m - y_old) / dt
                vz = (self.smoothed_z - z_old) / dt
                self.velocity_3d = (vx, vy, vz)
                self.speed_kmh = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6
                
                # Approach rate (closing speed along z-axis)
                self.approach_rate = -vz  # Positive when closing in
                if self.approach_rate > 0.8:
                    self.eta_seconds = max(0.1, self.smoothed_z / self.approach_rate)
                else:
                    self.eta_seconds = None

tracks_db = {}
MIN_CONSECUTIVE_FRAMES = 1  # Instant response for small mini drones
MAX_MISSED_FRAMES = 25      # Generous track retention for hand-held & distant tests

# UI State (Default 15% sensitivity for tiny hand-held / distant mini drones)
conf_percent = 15
brightness_boost = 0
clahe_enabled = False
dragging_slider = False
SLIDER_X1, SLIDER_X2, SLIDER_Y1, SLIDER_Y2 = 145, 520, 0, 0

def handle_mouse(event, x, y, flags, param):
    global conf_percent, dragging_slider, SLIDER_X1, SLIDER_X2, SLIDER_Y1, SLIDER_Y2
    if event == cv2.EVENT_LBUTTONDOWN:
        if SLIDER_X1 - 15 <= x <= SLIDER_X2 + 15 and SLIDER_Y1 - 10 <= y <= SLIDER_Y2 + 10:
            dragging_slider = True
            norm_val = (x - SLIDER_X1) / max(1, (SLIDER_X2 - SLIDER_X1))
            conf_percent = int(max(5, min(95, 5 + norm_val * 90)))
    elif event == cv2.EVENT_MOUSEMOVE and dragging_slider:
        norm_val = (x - SLIDER_X1) / max(1, (SLIDER_X2 - SLIDER_X1))
        conf_percent = int(max(5, min(95, 5 + norm_val * 90)))
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_slider = False

cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

prev_time = time.time()
frame_count = 0

print("\n" + "=" * 65)
print("  AIRSPACE DRONE TRACKING & CRLB ESTIMATION ENGINE ONLINE")
print("  - CRLB: Active (Analytical Fisher Information & Bounds)")
print("  - Target Profiles: Press [1, 2, 3, 4] to change drone size baseline")
print("  - Sensitivity: [ / ] or click bottom bar | Light: B | Quit: Q")
print("=" * 65 + "\n")

# ---------------------------------------------------------
# 5. MAIN REAL-TIME ESTIMATION & TRACKING LOOP
# ---------------------------------------------------------
current_frame = None

while True:
    if not is_paused or current_frame is None:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                # Auto-loop video file when it reaches the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                print("End of video or could not read frame.")
                break
        current_frame = frame.copy()
    else:
        frame = current_frame.copy()

    frame_count += 1
    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0
    prev_time = current_time

    if brightness_boost != 0:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_boost)

    # Optional CLAHE Contrast & Edge Enhancement (Key 'C')
    if clahe_enabled:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    h, w, _ = frame.shape
    cx_cam = w / 2.0
    cy_cam = h / 2.0
    FOCAL_LENGTH_PX = (w / 1280.0) * 850.0  # Dynamic focal length scaling

    conf_threshold = conf_percent / 100.0

    # Hardware-Accelerated YOLO Inference with High-Sensitivity Multi-Stage Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=max(0.05, conf_threshold * 0.70),
        iou=0.45,
        imgsz=current_imgsz,
        device=DEVICE,
        verbose=False
    )

    confirmed_drone_count = 0
    telemetry_records = []  # Digital Twin telemetry stream container

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id != DRONE_CLASS_ID or confidence < max(0.06, conf_threshold * 0.75):
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = (x2 - x1), (y2 - y1)
            aspect_ratio = float(box_w) / max(1.0, float(box_h))
            char_dim = max(box_w, box_h)

            # 1. Eyeglasses / Spectacles & Desktop Clutter Rejection Filter:
            # - Multi-rotor drones (Profiles 1-3) have compact symmetric footprints (aspect ratio 0.65 - 1.75).
            # - Eyeglasses/Spectacles have wide, elongated horizontal profiles (aspect ratio 1.85 - 3.8).
            # - Filter out elongated clutter (specs, pens, keyboards) unless confidence is exceptionally high (> 0.50).
            if active_profile_id != 4:  # For Multirotors (Mini 24cm, Standard 38cm, Heavy 75cm)
                if (aspect_ratio > 1.80 or aspect_ratio < 0.45) and confidence < 0.48:
                    continue
                if aspect_ratio > 2.2:  # Strictly reject extreme wide shapes (glasses, bars)
                    continue
            else:  # Tactical Fixed-Wing Profile
                if (aspect_ratio > 3.0 or aspect_ratio < 0.30) and confidence < 0.45:
                    continue

            # 2. Filter massive screen-filling objects (laptops, walls taking > 45% of screen)
            if (box_w * box_h) > (0.45 * w * h):
                continue

            # Anti-glitch persistence
            if track_id is None:
                continue

            if track_id not in tracks_db:
                tracks_db[track_id] = TargetKinematics(track_id)
            
            target_kin = tracks_db[track_id]

            # Compute Monocular Distance & CRLB (using rotation-invariant dimension)
            z_est, sigma_d, ci_low, ci_high, crlb_var = compute_crlb_distance(
                char_dim, box_h, target_nominal_width, FOCAL_LENGTH_PX, SIGMA_PIXEL
            )

            # 3D Coordinates relative to camera optical axis
            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0
            x_3d = ((u_center - cx_cam) * z_est) / FOCAL_LENGTH_PX
            y_3d = ((v_center - cy_cam) * z_est) / FOCAL_LENGTH_PX

            target_kin.update(frame_count, current_time, x_3d, y_3d, z_est, sigma_d)

            # Require persistent confirmation (1-2 frames or confidence >= 0.20)
            if target_kin.hits >= MIN_CONSECUTIVE_FRAMES or confidence >= 0.20:
                confirmed_drone_count += 1
                disp_z = target_kin.smoothed_z if target_kin.smoothed_z is not None else z_est

                # Record Telemetry for Digital Twin
                telemetry_records.append({
                    "track_id": track_id,
                    "confidence": confidence,
                    "position_3d": [x_3d, y_3d, disp_z],
                    "crlb_sigma": sigma_d,
                    "velocity_3d": target_kin.velocity_3d,
                    "speed_kmh": target_kin.speed_kmh,
                    "approach_rate": target_kin.approach_rate,
                    "eta_s": target_kin.eta_seconds,
                    "bbox": [x1, y1, x2, y2]
                })

                # Threat Zone Color Coding by Distance
                if disp_z < 10.0:
                    box_color = (0, 0, 255)       # Red: Critical Proximity
                    zone_str = "CRITICAL"
                elif disp_z < 25.0:
                    box_color = (0, 165, 255)     # Orange: Tactical Range
                    zone_str = "CAUTION"
                else:
                    box_color = (0, 255, 0)       # Green: Long Range
                    zone_str = "TRACKING"

                # Draw Target Box & Corner Reticles
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                corner_len = max(8, min(24, box_w // 3, box_h // 3))
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1 - corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)

                # Center Reticle Target Point
                cv2.circle(frame, (int(u_center), int(v_center)), 4, (0, 255, 255), -1)

                # --- MULTI-LINE TACTICAL HUD BADGE ---
                line1 = f"DRONE [ID:{track_id}] {confidence*100:.0f}% | {zone_str}"
                
                # Adaptive error formatting (cm for close range, m for long range)
                if sigma_d < 0.20:
                    err_str = f"+/-{sigma_d*100:.1f}cm"
                    ci_str = f"{ci_low:.2f}-{ci_high:.2f}m"
                else:
                    err_str = f"+/-{sigma_d:.2f}m"
                    ci_str = f"{ci_low:.1f}-{ci_high:.1f}m"
                    
                line2 = f"DIST: {disp_z:.2f}m [CRLB: {err_str} (95% CI: {ci_str})]"
                
                # Approach vector text
                if target_kin.approach_rate > 0.8:
                    eta_str = f"ETA: {target_kin.eta_seconds:.1f}s" if target_kin.eta_seconds else ""
                    line3 = f"VEL: {target_kin.speed_kmh:.1f}km/h (CLOSING @ +{target_kin.approach_rate:.1f}m/s {eta_str})"
                    line3_color = (0, 255, 255)
                elif target_kin.approach_rate < -0.8:
                    line3 = f"VEL: {target_kin.speed_kmh:.1f}km/h (RECEDING @ {target_kin.approach_rate:.1f}m/s)"
                    line3_color = (180, 255, 180)
                else:
                    line3 = f"VEL: {target_kin.speed_kmh:.1f}km/h (HOVER / LATERAL)"
                    line3_color = (220, 220, 220)

                badge_w = max(340, int(box_w + 80))
                badge_h = 58
                badge_y1 = max(0, y1 - badge_h - 6)
                badge_y2 = y1 - 6

                # Semi-transparent HUD overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, badge_y1), (x1 + badge_w, badge_y2), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + badge_w, badge_y2), box_color, 1)

                cv2.putText(frame, line1, (x1 + 6, badge_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, line2, (x1 + 6, badge_y1 + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, line3, (x1 + 6, badge_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, line3_color, 1, cv2.LINE_AA)

    # Stale track cleanup
    stale_ids = [tid for tid, obj in tracks_db.items() if frame_count - obj.last_frame > MAX_MISSED_FRAMES]
    for tid in stale_ids:
        del tracks_db[tid]

    # ---------------------------------------------------------
    # 6. TOP HEADER TELEMETRY BAR
    # ---------------------------------------------------------
    header_color = (0, 0, 180) if confirmed_drone_count > 0 else (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 42), header_color, -1)
    
    status_msg = f"AIRSPACE ALERT: {confirmed_drone_count} ACTIVE TARGET(S)" if confirmed_drone_count > 0 else "AIRSPACE SURVEILLANCE: SCANNING (CRLB ACTIVE)"
    cv2.putText(frame, status_msg, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255) if confirmed_drone_count > 0 else (0, 255, 0), 2, cv2.LINE_AA)

    profile_name = DRONE_PROFILES[active_profile_id]["name"]
    source_tag = f"VID: {video_filename}" if is_video_file else f"Ref: {profile_name} ({target_nominal_width*100:.0f}cm)"
    pause_tag = " [PAUSED]" if is_paused else ""
    header_right = f"{source_tag}{pause_tag} | FPS: {fps:.1f}"
    (rw, _), _ = cv2.getTextSize(header_right, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(frame, header_right, (w - rw - 15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

    # ---------------------------------------------------------
    # 7. BOTTOM TACTICAL FOOTER HUD (Sensitivity & CRLB Controls)
    # ---------------------------------------------------------
    footer_h = 46
    cv2.rectangle(frame, (0, h - footer_h), (w, h), (20, 20, 20), -1)
    cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (60, 60, 60), 1)

    cv2.putText(frame, "SENSITIVITY:", (15, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    SLIDER_X1 = 145
    SLIDER_X2 = min(w - 480, 520)
    SLIDER_Y1 = h - 30
    SLIDER_Y2 = h - 16

    cv2.rectangle(frame, (SLIDER_X1, SLIDER_Y1), (SLIDER_X2, SLIDER_Y2), (50, 50, 50), -1)
    fill_ratio = (conf_percent - 10) / 85.0
    fill_x = int(SLIDER_X1 + fill_ratio * (SLIDER_X2 - SLIDER_X1))
    fill_color = (0, 165, 255) if conf_percent > 45 else (0, 220, 100)
    cv2.rectangle(frame, (SLIDER_X1, SLIDER_Y1), (fill_x, SLIDER_Y2), fill_color, -1)
    cv2.circle(frame, (fill_x, (SLIDER_Y1 + SLIDER_Y2) // 2), 7, (255, 255, 255), -1)
    cv2.circle(frame, (fill_x, (SLIDER_Y1 + SLIDER_Y2) // 2), 8, (0, 140, 255), 2)

    cv2.putText(frame, f"{conf_percent}%", (SLIDER_X2 + 12, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Key helpers & Profile selector prompt
    if is_video_file:
        controls_msg = "Profiles [1-4] | [SPACE]: Pause/Play | [R]: Replay | Mode: [T] | Quit: Q"
    else:
        clahe_tag = "[ON]" if clahe_enabled else ""
        controls_msg = f"Profiles [1-4] | Mode: [T] | Contrast: [C]{clahe_tag} | Sens: [ / ] | Quit: Q"
        
    (cw, _), _ = cv2.getTextSize(controls_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, controls_msg, (w - cw - 15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    # ---------------------------------------------------------
    # 8. KEYBOARD COMMAND DISPATCHER
    # ---------------------------------------------------------
    wait_delay = 30 if is_video_file and not is_paused else 1
    key = cv2.waitKey(wait_delay) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break
    elif key in (ord("t"), ord("T")):  # T: Toggle Turbo 60FPS (640) <-> Ultra-Range (1280)
        resolutions = [640, 960, 1280]
        curr_idx = resolutions.index(current_imgsz) if current_imgsz in resolutions else 0
        current_imgsz = resolutions[(curr_idx + 1) % len(resolutions)]
        print(f"[*] Switched Inference Resolution Mode: {current_imgsz}px")
    elif key in (ord("c"), ord("C")):  # C: Toggle CLAHE Contrast Enhancement
        clahe_enabled = not clahe_enabled
        print(f"[*] CLAHE Edge & Contrast Boost: {'ENABLED' if clahe_enabled else 'DISABLED'}")
    elif key == ord(" "):  # Space: Pause/Resume video
        is_paused = not is_paused
    elif key in (ord("r"), ord("R")):  # R: Replay video
        if is_video_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            is_paused = False
            tracks_db.clear()
    elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
        active_profile_id = int(chr(key))
        target_nominal_width = DRONE_PROFILES[active_profile_id]["width"]
        print(f"[*] Switched Drone Size Profile: {DRONE_PROFILES[active_profile_id]['desc']}")
    elif key in (ord("+"), ord("="), ord("]"), 0, 82):
        conf_percent = min(95, conf_percent + 5)
    elif key in (ord("-"), ord("_"), ord("["), 1, 84):
        conf_percent = max(5, conf_percent - 5)
    elif key in (ord("b"),):
        brightness_boost = (brightness_boost + 15) if brightness_boost < 60 else -30
    elif key in (ord("B"),):
        brightness_boost = max(-50, brightness_boost - 15)

cap.release()
cv2.destroyAllWindows()
