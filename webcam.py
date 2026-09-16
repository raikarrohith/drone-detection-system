import sys
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. LOAD FINE-TUNED DRONE DETECTOR MODEL
# ---------------------------------------------------------
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

DRONE_CLASS_ID = 0
for cls_id, name in model.names.items():
    if "drone" in name.lower():
        DRONE_CLASS_ID = cls_id
        break

# ---------------------------------------------------------
# 2. CAMERA SETUP & SENSOR CALIBRATION PARAMETERS
# ---------------------------------------------------------
cap = None
is_windows = sys.platform.startswith("win")
backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY

for cam_idx in [1, 2, 0, 3]:
    temp_cap = cv2.VideoCapture(cam_idx, backend)
    if not temp_cap.isOpened():
        temp_cap = cv2.VideoCapture(cam_idx)
    if temp_cap.isOpened():
        ret, test_frame = temp_cap.read()
        if ret and test_frame is not None:
            cap = temp_cap
            print(f"[*] Successfully connected to camera (index {cam_idx})")
            break
        else:
            temp_cap.release()

if cap is None or not cap.isOpened():
    print("ERROR: Could not open camera. Please check camera connections.")
    exit(1)

# Set high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

WINDOW_NAME = "Drone Defense System - CRLB Range & Kinematics Engine"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# ---------------------------------------------------------
# 3. CRAMER-RAO LOWER BOUND (CRLB) & PHYSICAL TARGET MODELS
# ---------------------------------------------------------
# Camera Optical Model: Focal length in pixels (calibrated for standard ~80 deg HFOV at 1080p/720p)
FOCAL_LENGTH_PX = 1150.0   # Updated dynamically if resolution changes
SIGMA_PIXEL = 1.5          # Bounding box edge localization noise std-dev (pixels)

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
    Minimum Standard Error: sigma_D = sqrt(CRLB) = (sigma_w * D^2) / (W * F)
    """
    pixel_w = max(2.0, float(pixel_w))
    # Estimated Distance (Z in meters)
    distance_z = (target_w * focal_length) / pixel_w
    
    # Fisher Information & CRLB Variance
    fisher_info = (target_w**2 * focal_length**2) / ((sigma_w**2) * (distance_z**4))
    crlb_variance = 1.0 / max(1e-9, fisher_info)
    sigma_d = math.sqrt(crlb_variance)
    
    # 95% Confidence Interval (2-sigma theoretical bound)
    ci_lower = max(0.2, distance_z - 2.0 * sigma_d)
    ci_upper = distance_z + 2.0 * sigma_d
    
    return distance_z, sigma_d, ci_lower, ci_upper, crlb_variance

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
MIN_CONSECUTIVE_FRAMES = 3
MAX_MISSED_FRAMES = 15

# UI State
conf_percent = 35
brightness_boost = 0
dragging_slider = False
SLIDER_X1, SLIDER_X2, SLIDER_Y1, SLIDER_Y2 = 145, 520, 0, 0

def handle_mouse(event, x, y, flags, param):
    global conf_percent, dragging_slider, SLIDER_X1, SLIDER_X2, SLIDER_Y1, SLIDER_Y2
    if event == cv2.EVENT_LBUTTONDOWN:
        if SLIDER_X1 - 15 <= x <= SLIDER_X2 + 15 and SLIDER_Y1 - 10 <= y <= SLIDER_Y2 + 10:
            dragging_slider = True
            norm_val = (x - SLIDER_X1) / max(1, (SLIDER_X2 - SLIDER_X1))
            conf_percent = int(max(10, min(95, 10 + norm_val * 85)))
    elif event == cv2.EVENT_MOUSEMOVE and dragging_slider:
        norm_val = (x - SLIDER_X1) / max(1, (SLIDER_X2 - SLIDER_X1))
        conf_percent = int(max(10, min(95, 10 + norm_val * 85)))
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
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from camera.")
        break

    frame_count += 1
    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0
    prev_time = current_time

    if brightness_boost != 0:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_boost)

    h, w, _ = frame.shape
    cx_cam = w / 2.0
    cy_cam = h / 2.0
    FOCAL_LENGTH_PX = (w / 1920.0) * 1150.0  # Dynamic focal length scaling

    conf_threshold = conf_percent / 100.0

    # High-Resolution YOLO Inference
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf_threshold,
        iou=0.45,
        imgsz=1280,
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

            if cls_id != DRONE_CLASS_ID or confidence < conf_threshold:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = (x2 - x1), (y2 - y1)

            # Filter massive screen-filling objects (laptops, humans right against lens)
            if (box_w * box_h) > (0.40 * w * h):
                continue

            # Anti-glitch persistence
            if track_id is None:
                continue

            if track_id not in tracks_db:
                tracks_db[track_id] = TargetKinematics(track_id)
            
            target_kin = tracks_db[track_id]

            # Compute Monocular Distance & CRLB
            z_est, sigma_d, ci_low, ci_high, crlb_var = compute_crlb_distance(
                box_w, box_h, target_nominal_width, FOCAL_LENGTH_PX, SIGMA_PIXEL
            )

            # 3D Coordinates relative to camera optical axis
            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0
            x_3d = ((u_center - cx_cam) * z_est) / FOCAL_LENGTH_PX
            y_3d = ((v_center - cy_cam) * z_est) / FOCAL_LENGTH_PX

            target_kin.update(frame_count, current_time, x_3d, y_3d, z_est, sigma_d)

            # Require persistent confirmation (3 frames)
            if target_kin.hits >= MIN_CONSECUTIVE_FRAMES or (target_kin.hits >= 2 and confidence >= 0.65):
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
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)

                # Center Reticle Target Point
                cv2.circle(frame, (int(u_center), int(v_center)), 4, (0, 255, 255), -1)

                # --- MULTI-LINE TACTICAL HUD BADGE ---
                line1 = f"DRONE [ID:{track_id}] {confidence*100:.0f}% | {zone_str}"
                line2 = f"DIST: {disp_z:.1f}m [CRLB: +/-{sigma_d:.2f}m (2sigma: {ci_low:.1f}-{ci_high:.1f}m)]"
                
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

                badge_w = max(320, int(box_w + 80))
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
    header_right = f"Target Ref: {profile_name} ({target_nominal_width*100:.0f}cm) | FPS: {fps:.1f}"
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
    controls_msg = "Profiles [1:Mini | 2:Std | 3:Hvy | 4:Wing] | Sens: [ / ] | Light: B | Quit: Q"
    (cw, _), _ = cv2.getTextSize(controls_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, controls_msg, (w - cw - 15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)

    # ---------------------------------------------------------
    # 8. KEYBOARD COMMAND DISPATCHER
    # ---------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break
    elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
        active_profile_id = int(chr(key))
        target_nominal_width = DRONE_PROFILES[active_profile_id]["width"]
        print(f"[*] Switched Drone Size Profile: {DRONE_PROFILES[active_profile_id]['desc']}")
    elif key in (ord("+"), ord("="), ord("]"), 0, 82):
        conf_percent = min(95, conf_percent + 5)
    elif key in (ord("-"), ord("_"), ord("["), 1, 84):
        conf_percent = max(10, conf_percent - 5)
    elif key in (ord("b"),):
        brightness_boost = (brightness_boost + 15) if brightness_boost < 60 else -30
    elif key in (ord("B"),):
        brightness_boost = max(-50, brightness_boost - 15)

cap.release()
cv2.destroyAllWindows()




