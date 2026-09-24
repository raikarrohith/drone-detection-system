import sys
import os
import time
import math
import argparse
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from drone_classifier import (
    DroneTypeClassifier,
    get_drone_width,
    get_drone_domain,
    get_drone_dimensions,
    DRONE_WIDTHS,
    AIRFRAME_DOMAINS
)

# ---------------------------------------------------------
# 1. PARSE ARGUMENTS & LOAD YOLO MODEL WITH ACCELERATION
# ---------------------------------------------------------
import torch

parser = argparse.ArgumentParser(description="Drone Defense & CRLB Distance Estimation System")
parser.add_argument("--video", "-v", type=str, default=None, help="Path to test video file (e.g. drone_test.mp4)")
parser.add_argument("--camera", "-c", type=int, default=None, help="Camera index (e.g. 0, 1, 2)")
parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution: 960 (balanced long-range) or 1280/640")
parser.add_argument("--type-model", type=str, default="models/drone_type_classifier.pt", help="Optional civilian/military classification model (.pt or .pth)")
parser.add_argument("--type-confidence", type=float, default=0.45, help="Minimum drone-type classifier confidence (0-1)")
parser.add_argument("--drone-width", type=float, default=None, help="Measured rotor-tip-to-tip span in metres; overrides the selected profile width")
parser.add_argument("--focal-length-px", type=float, default=None, help="Calibrated focal length in pixels for the active camera resolution")
parser.add_argument("--calibration-distance", type=float, default=None, help="Known target distance in metres; press K while the drone is detected to calibrate focal length")
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

# Detect if 4-Head P2 Tiny-Object Architecture is active
is_p2_model = False
try:
    if hasattr(model.model, "model"):
        detect_head = model.model.model[-1]
        if hasattr(detect_head, "nl") and detect_head.nl == 4:
            is_p2_model = True
except Exception:
    pass

if is_p2_model:
    print("[*] Detector Architecture: 4-Head P2 Tiny-Object Detection Engine (Stride 4 ACTIVE)")
else:
    print("[*] Detector Architecture: Standard 3-Head YOLO (Stride 8-32)")

type_classifier = DroneTypeClassifier(args.type_model, args.type_confidence)
if type_classifier.enabled:
    print(f"[*] Drone Type Classifier: {args.type_model} (threshold {args.type_confidence:.0%})")
else:
    print("[*] Drone Type Classifier: unavailable; detections will be labeled UNKNOWN")

DRONE_CLASS_ID = 0
for cls_id, name in model.names.items():
    if "drone" in name.lower():
        DRONE_CLASS_ID = cls_id
        break

# ---------------------------------------------------------
# PERSISTENT MULTI-CAMERA CALIBRATION
# ---------------------------------------------------------
CALIBRATION_FILE = "models/camera_calibration.json"

def get_device_profile_key(cam_idx=None, is_vid=False, vid_name=""):
    if is_vid:
        stem = Path(vid_name).stem if vid_name else "default_video"
        return f"video_{stem}"
    elif cam_idx is not None:
        return f"camera_{cam_idx}"
    return "camera_default"

def load_camera_calibration(device_key):
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
                cameras_dict = data.get("cameras", {})
                if device_key in cameras_dict:
                    cam_entry = cameras_dict[device_key]
                    f_val = float(cam_entry.get("focal_length_px", 850.0))
                    print(f"[*] Loaded Calibration for [{device_key}]: F = {f_val:.1f}px")
                    return f_val
                elif "focal_length_px" in data:
                    f_val = float(data["focal_length_px"])
                    print(f"[*] Loaded Default Calibration: F = {f_val:.1f}px")
                    return f_val
        except Exception as e:
            print(f"[!] Warning reading camera calibration: {e}")
    return None

def save_camera_calibration(device_key, focal_len, calib_dist=None, target_width=None, resolution="1920x1080"):
    try:
        os.makedirs(os.path.dirname(CALIBRATION_FILE) or ".", exist_ok=True)
        data = {"cameras": {}}
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, "r") as f:
                    existing = json.load(f)
                    if "cameras" in existing:
                        data["cameras"] = existing["cameras"]
            except Exception:
                pass
                
        data["cameras"][device_key] = {
            "focal_length_px": round(float(focal_len), 2),
            "resolution": resolution,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "calibration_distance_m": calib_dist,
            "target_width_m": target_width
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[*] Saved Calibration for [{device_key}] to {CALIBRATION_FILE}: F = {focal_len:.1f}px")
        return True
    except Exception as e:
        print(f"[!] Error saving calibration: {e}")
        return False

# ---------------------------------------------------------
# 2. SOURCE SETUP (Zero-Latency Live Camera & Video Stream)
# ---------------------------------------------------------
cap = None
is_video_file = False
video_filename = ""
active_cam_idx = None
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
            try:
                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                temp_cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
                active_cam_idx = cam_idx
                print(f"[*] Successfully connected to Live Camera (index {cam_idx}) at Zero-Latency Buffer")
                break
            else:
                temp_cap.release()

if cap is None or not cap.isOpened():
    print("ERROR: Could not open camera or video file. Please check connections or video path.")
    exit(1)

WINDOW_NAME = "Drone Defense System - CRLB Range & Kinematics Engine"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Determine Active Device Calibration Key
ACTIVE_DEVICE_KEY = get_device_profile_key(active_cam_idx, is_video_file, video_filename)

# ---------------------------------------------------------
# 3. CRAMER-RAO LOWER BOUND (CRLB) & PHYSICAL TARGET MODELS
# ---------------------------------------------------------
# Camera Optical Model: Focal length in pixels
saved_focal_len = load_camera_calibration(ACTIVE_DEVICE_KEY)
calibrated_focal_length_px = args.focal_length_px if args.focal_length_px is not None else saved_focal_len
FOCAL_LENGTH_PX = calibrated_focal_length_px if calibrated_focal_length_px is not None else 850.0
SIGMA_PIXEL = 2.5          # Realistic YOLO bounding box edge regression noise std-dev (pixels)
POSE_ASPECT_RATIO_UNCERTAINTY = 0.045  # 4.5% standard error due to 3D drone yaw/pitch rotation
CALIBRATION_RELATIVE_UNCERTAINTY = 0.05 if calibrated_focal_length_px is not None else 0.10

# Drone Physical Size Profiles (Wingspan, Height, and Diagonal in meters)
DRONE_PROFILES = {
    1: {"name": "Micro/Mini", "width": 0.24, "height": 0.08, "desc": "DJI Mini / Avata (24x8cm)"},
    2: {"name": "Standard Quad", "width": 0.38, "height": 0.14, "desc": "Mavic / Phantom / FPV (38x14cm) [DEFAULT]"},
    3: {"name": "Heavy Lift", "width": 0.75, "height": 0.30, "desc": "Matrice / Hexacopter (75x30cm)"},
    4: {"name": "Tactical Wing", "width": 1.40, "height": 0.35, "desc": "Fixed-Wing / Loitering (140x35cm)"}
}
active_profile_id = 2
target_nominal_profile = DRONE_PROFILES[active_profile_id]
if args.drone_width is not None:
    if args.drone_width <= 0:
        parser.error("--drone-width must be greater than zero")
    profile_ratio = target_nominal_profile["height"] / target_nominal_profile["width"]
    target_nominal_profile = {
        **target_nominal_profile,
        "name": "Measured Drone",
        "width": args.drone_width,
        "height": args.drone_width * profile_ratio,
        "desc": f"Measured width ({args.drone_width * 100:.1f}cm)",
    }
target_nominal_width = target_nominal_profile["width"]

def compute_crlb_distance(pixel_w, pixel_h, target_profile, focal_length, sigma_w, u_center=None, v_center=None, cx_cam=None, cy_cam=None):
    """
    Computes High-Precision Monocular Orthogonal Depth (Z) and Cramer-Rao Lower Bound (CRLB).
    
    In projective pinhole geometry:
    - Orthogonal depth Z along the optical axis is directly: Z = (W_physical * F) / pw
    - Rotor-to-rotor Width (W) is the most robust geometric invariant (immune to pitch/gimbal tilt).
    - Sub-pixel boundary compensation eliminates YOLO edge regression slack.
    """
    target_w = target_profile["width"]
    target_h = target_profile.get("height", target_w * 0.35)
    target_diag = math.sqrt(target_w**2 + target_h**2)
    
    # Sub-pixel boundary compensation (YOLO bounding box regression padding)
    pw = max(2.0, float(pixel_w) - 1.5)
    ph = max(2.0, float(pixel_h) - 1.5)
    pdiag = math.sqrt(pw**2 + ph**2)
    
    # 1. Multi-Cue Distance Estimators
    d_w = (target_w * focal_length) / pw
    d_diag = (target_diag * focal_length) / pdiag
    d_h = (target_h * focal_length) / ph
    
    # 2. Optimal Width-Dominant BLUE Fusion
    # Rotor width is invariant to 3D pitch/tilt; height is susceptible to landing gear and rotor blur
    fisher_w = (target_w**2 * focal_length**2) / ((sigma_w**2) * (d_w**4))
    fisher_diag = (target_diag**2 * focal_length**2) / ((sigma_w**2) * (d_diag**4)) * 0.25
    fisher_h = (target_h**2 * focal_length**2) / ((sigma_w**2) * (d_h**4)) * 0.05
    
    fisher_total = max(1e-9, fisher_w + fisher_diag + fisher_h)
    w_w = fisher_w / fisher_total
    w_diag = fisher_diag / fisher_total
    w_h = fisher_h / fisher_total
    
    distance_z = (w_w * d_w) + (w_diag * d_diag) + (w_h * d_h)
    
    # 3. Variance & CRLB Lower Bounds
    crlb_var_pixel = 1.0 / fisher_total
    crlb_var_pose = (POSE_ASPECT_RATIO_UNCERTAINTY * distance_z) ** 2
    crlb_var_calibration = (CALIBRATION_RELATIVE_UNCERTAINTY * distance_z) ** 2
    total_crlb_variance = crlb_var_pixel + crlb_var_pose + crlb_var_calibration
    sigma_d = math.sqrt(total_crlb_variance)
    
    # 95% Confidence Interval (2-sigma bound)
    ci_lower = max(0.1, distance_z - 2.0 * sigma_d)
    ci_upper = distance_z + 2.0 * sigma_d
    
    return distance_z, sigma_d, ci_lower, ci_upper, total_crlb_variance

def is_hollow_eyeglasses(crop_bgr):
    """
    Discriminates hollow eyeglasses frames from real drones with physical motors/fuselage.
    Eyeglasses have two large transparent empty lens regions (< 2.8% internal edge density),
    whereas real drones have motors, propellers, duct struts, camera chassis, and wiring.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    ch, cw, _ = crop_bgr.shape
    if ch < 18 or cw < 25:
        return False
        
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    # Left and Right optical lens centers (inner 50% height, 18-38% width and 62-82% width)
    y1, y2 = int(0.22 * ch), int(0.78 * ch)
    left_lens = edges[y1:y2, int(0.18 * cw):int(0.38 * cw)]
    right_lens = edges[y1:y2, int(0.62 * cw):int(0.82 * cw)]
    
    l_density = np.count_nonzero(left_lens) / max(1, left_lens.size)
    r_density = np.count_nonzero(right_lens) / max(1, right_lens.size)
    
    # If both lens zones are hollow glass windows, reject as eyeglasses
    return bool(l_density < 0.028 and r_density < 0.028)

def is_human_or_face_false_positive(crop_bgr, aspect_ratio, confidence):
    """
    Discriminates humans, faces, hands, and moving bodies from airborne drones.
    - Dual HSV + YCrCb chrominance detects human facial/body skin across complexions and lighting.
    - Rejects moving heads, faces, hands, and nearby humans while preserving drones.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
        
    # 1. Reject vertically elongated shapes (faces, heads, standing/sitting humans, necks)
    if aspect_ratio < 1.05 and confidence < 0.65:
        return True
        
    # 2. Dual HSV + YCrCb Chromaticity Skin Analysis
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([28, 255, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([168, 20, 40]), np.array([180, 255, 255]))
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 128, 70]), np.array([255, 182, 138]))
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    skin_ratio = np.count_nonzero(skin_mask) / max(1, skin_mask.size)
    
    # Face / head / hand signature: contains noticeable skin tone (> 5%) on near-square or vertical shape
    if aspect_ratio < 1.45 and skin_ratio > 0.05 and confidence < 0.70:
        return True
        
    # Moving face / hands / body signature: contains > 10% skin tone regardless of aspect ratio
    if skin_ratio > 0.10 and confidence < 0.65:
        return True
        
    return False

# ---------------------------------------------------------
# 4. KINEMATIC TRACKER MEMORY (3D State, Dynamic CRLB Kalman Filter)
# ---------------------------------------------------------
class KalmanRangeFilter:
    """
    Constant-Velocity Kinematic Kalman Filter with dynamic CRLB measurement noise covariance (Derivation 3B).
    State vector: x = [z (distance in m), vz (approach velocity in m/s)]^T
    Dynamically adjusts Kalman Gain K_k = P_k^- H^T (H P_k^- H^T + R_k)^-1 where R_k = CRLB_variance.
    """
    def __init__(self, init_z, init_sigma):
        self.z = float(init_z)
        self.vz = 0.0
        self.p00 = max(0.04, float(init_sigma)**2)
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 4.0  # Initial velocity variance (2 m/s std)
        self.q_pos = 0.05
        self.q_vel = 0.35

    def predict(self, dt):
        if dt <= 0 or dt > 1.5:
            dt = 0.033  # Safe fallback for frame gaps
        self.z += self.vz * dt
        # P = F * P * F^T + Q
        new_p00 = self.p00 + dt * (self.p10 + self.p01) + (dt**2) * self.p11 + self.q_pos * dt
        new_p01 = self.p01 + dt * self.p11
        new_p10 = self.p10 + dt * self.p11
        new_p11 = self.p11 + self.q_vel * dt
        self.p00, self.p01, self.p10, self.p11 = new_p00, new_p01, new_p10, new_p11
        return self.z, self.vz

    def update(self, z_meas, crlb_variance):
        r = max(1e-4, float(crlb_variance))
        y = z_meas - self.z  # Residual
        s = self.p00 + r      # Innovation covariance
        k0 = self.p00 / s    # Optimal Kalman Gain for distance (Derivation 3B)
        k1 = self.p10 / s    # Optimal Kalman Gain for velocity
        
        self.z += k0 * y
        self.vz += k1 * y
        
        # P = (I - K * H) * P
        self.p00 = max(1e-4, (1.0 - k0) * self.p00)
        self.p01 = (1.0 - k0) * self.p01
        self.p10 = -k1 * self.p00 + self.p10
        self.p11 = max(1e-3, -k1 * self.p01 + self.p11)
        return self.z, self.vz

class BayesianDroneClassifier:
    """
    Physics-Informed Bayesian Multi-Cue Classification Engine (Derivation Multi-Modal).
    Fuses:
    1. Visual Evidence: P(Military | Visual Crop) from neural classifier.
    2. Kinematic Evidence: Speed & Hover State (Fixed-wing cannot hover, multirotor hovers).
    3. Geometric Evidence: Aspect ratio (w/h) and CRLB estimated physical span.
    4. Temporal Persistence: Recursive Log-Odds Bayes Filter.
    """
    def __init__(self, prior_military_prob=0.30):
        self.log_odds = math.log(prior_military_prob / (1.0 - prior_military_prob))
        self.military_prob = prior_military_prob

    def update(self, visual_label, visual_conf, speed_kmh, aspect_ratio, span_m=0.38):
        # 1. Visual Evidence Likelihood
        if visual_label and str(visual_label).upper() != "UNKNOWN":
            domain = get_drone_domain(visual_label)
            if domain == "MILITARY":
                p_vis_mil = min(0.95, max(0.55, float(visual_conf)))
            elif domain == "CIVILIAN":
                p_vis_mil = max(0.05, min(0.45, 1.0 - float(visual_conf)))
            else:
                p_vis_mil = 0.50
            l_visual = math.log(p_vis_mil / max(1e-4, 1.0 - p_vis_mil))
        else:
            l_visual = 0.0

        # 2. Kinematic Velocity & Hover Likelihood
        # Multirotors (Civilian) hover (0-20 km/h) or fly 0-60 km/h.
        # Fixed-Wing (Military) cannot hover (stall speed ~70-90 km/h), cruise 100-300 km/h.
        if speed_kmh is not None and speed_kmh >= 0.0:
            if speed_kmh < 22.0:
                l_kin = -1.2  # Strong multirotor / civilian hovering evidence
            elif speed_kmh > 85.0:
                l_kin = 1.4   # Strong fixed-wing / tactical military speed
            elif speed_kmh > 55.0:
                l_kin = 0.4
            else:
                l_kin = -0.3
        else:
            l_kin = 0.0

        # 3. Geometric Aspect Ratio Likelihood
        # Multirotors: 1.2 <= AR <= 2.2. Fixed-wing tactical UAVs: AR >= 2.6 up to 5.0
        if aspect_ratio >= 2.7:
            l_geom = 1.5   # High aspect ratio = Fixed-wing Tactical
        elif aspect_ratio >= 2.3:
            l_geom = 0.7
        elif aspect_ratio <= 1.8:
            l_geom = -1.0  # Multirotor signature
        else:
            l_geom = 0.0

        # 4. Physical Span Likelihood (CRLB estimated)
        if span_m > 1.20:
            l_span = 0.8
        elif span_m < 0.50:
            l_span = -0.6
        else:
            l_span = 0.0

        # 5. Recursive Bayesian Fusion (weighted update with temporal smoothing)
        w_vis = 1.0 if l_visual != 0.0 else 0.0
        w_kin = 0.7 if speed_kmh is not None and speed_kmh > 0.5 else 0.0
        w_geom = 1.0
        w_span = 0.5

        delta_log_odds = (w_vis * l_visual) + (w_kin * l_kin) + (w_geom * l_geom) + (w_span * l_span)
        self.log_odds = 0.85 * self.log_odds + delta_log_odds

        # Bound log odds to prevent saturation [-5.0, 5.0]
        self.log_odds = max(-5.0, min(5.0, self.log_odds))
        self.military_prob = 1.0 / (1.0 + math.exp(-self.log_odds))

        # Decision
        if self.military_prob >= 0.52:
            domain = "MILITARY"
            conf = self.military_prob
            if visual_label and str(visual_label).lower() not in ["civilian", "military", "unknown"]:
                type_name = str(visual_label)
            else:
                type_name = "Tactical-Wing"
        else:
            domain = "CIVILIAN"
            conf = 1.0 - self.military_prob
            if visual_label and str(visual_label).lower() not in ["civilian", "military", "unknown"]:
                type_name = str(visual_label)
            else:
                if aspect_ratio >= 1.4:
                    type_name = "Quadcopter-UAV"
                else:
                    type_name = "Micro-Mini"

        return domain, type_name, conf


class TargetKinematics:
    def __init__(self, track_id):
        self.track_id = track_id
        self.history = []  # [(timestamp, x, y, z, dist, sigma_d)]
        self.hits = 0
        self.last_frame = 0
        self.last_timestamp = None
        self.kalman_filter = None
        self.smoothed_z = None
        self.velocity_3d = (0.0, 0.0, 0.0)  # vx, vy, vz in m/s
        self.speed_kmh = 0.0
        self.approach_rate = 0.0  # m/s (+ approaching, - receding)
        self.eta_seconds = None
        self.bayes_classifier = BayesianDroneClassifier()
        self.cached_type = "UNKNOWN"
        self.cached_domain = "UNKNOWN"
        self.cached_type_score = 0.0
        self.auto_profile = None
        self.last_classified_frame = -10

    def update_type(self, visual_label, visual_conf, aspect_ratio=2.0):
        """Physics-Informed Bayesian Multi-Cue Fusion: Visual + Kinematics + Geometry."""
        current_span = self.auto_profile["width"] if self.auto_profile else 0.38
        domain, type_name, conf = self.bayes_classifier.update(
            visual_label=visual_label,
            visual_conf=visual_conf,
            speed_kmh=self.speed_kmh,
            aspect_ratio=aspect_ratio,
            span_m=current_span
        )
        self.cached_domain = domain
        self.cached_type = type_name
        self.cached_type_score = min(0.99, max(0.50, conf))
        self.auto_profile = get_drone_dimensions(self.cached_type, aspect_ratio=aspect_ratio)
        return self.cached_type, self.cached_type_score, self.auto_profile

    def update(self, frame_num, timestamp, x_m, y_m, z_m, sigma_d, crlb_variance):
        self.hits += 1
        self.last_frame = frame_num

        dt = (timestamp - self.last_timestamp) if (self.last_timestamp is not None) else 0.033
        self.last_timestamp = timestamp

        # Adaptive CRLB Kalman State Estimation
        if self.kalman_filter is None:
            self.kalman_filter = KalmanRangeFilter(z_m, sigma_d)
            self.smoothed_z = z_m
        else:
            self.kalman_filter.predict(dt)
            filtered_z, filtered_vz = self.kalman_filter.update(z_m, crlb_variance)
            self.smoothed_z = max(0.1, filtered_z)

        self.history.append((timestamp, x_m, y_m, self.smoothed_z, sigma_d))
        if len(self.history) > 30:
            self.history.pop(0)

        # Calculate 3D Velocity & Approach Rate if we have temporal baseline
        if len(self.history) >= 4:
            t_old, x_old, y_old, z_old, _ = self.history[-4]
            dt_base = timestamp - t_old
            if dt_base > 0.05:
                vx = (x_m - x_old) / dt_base
                vy = (y_m - y_old) / dt_base
                vz = (self.smoothed_z - z_old) / dt_base
                self.velocity_3d = (vx, vy, vz)
                self.speed_kmh = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6
                
                # Approach rate (closing speed along z-axis)
                self.approach_rate = -vz  # Positive when closing in
                if self.approach_rate > 0.8:
                    self.eta_seconds = max(0.1, self.smoothed_z / self.approach_rate)
                else:
                    self.eta_seconds = None
                
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
print("  - Calibration: --calibration-distance <m>, then press [K] while tracking")
print("  - Sensitivity: [ / ] or click bottom bar | Light: B | Quit: Q")
print("=" * 65 + "\n")

# ---------------------------------------------------------
# 5. MAIN REAL-TIME ESTIMATION & TRACKING LOOP
# ---------------------------------------------------------
current_frame = None
latest_box_width = None
latest_target_width = target_nominal_profile["width"]
calib_status_text = f"Loaded F={FOCAL_LENGTH_PX:.1f}px from file" if saved_focal_len else ""
calib_status_expiry = time.time() + 3.0 if saved_focal_len else 0

# Setup Telemetry Logging Directory
telemetry_dir = Path("flight_logs")
telemetry_dir.mkdir(parents=True, exist_ok=True)
telemetry_csv_path = telemetry_dir / "telemetry.csv"
if not telemetry_csv_path.exists():
    with open(telemetry_csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "timestamp", "frame", "track_id", "confidence", "airframe_type",
            "domain", "type_conf", "span_m", "x_m", "y_m", "distance_m",
            "crlb_sigma_m", "speed_kmh", "approach_rate_mps", "eta_s"
        ])

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
    if calibrated_focal_length_px is not None:
        FOCAL_LENGTH_PX = calibrated_focal_length_px
    else:
        FOCAL_LENGTH_PX = (w / 1280.0) * 850.0

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
    telemetry_records = []
    classified_in_this_frame = 0

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id != DRONE_CLASS_ID or confidence < max(0.12, conf_threshold * 0.75):
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = (x2 - x1), (y2 - y1)
            aspect_ratio = float(box_w) / max(1.0, float(box_h))
            latest_box_width = max(2, box_w)
            char_dim = max(box_w, box_h)

            # 1. Clutter Rejection Filter (extreme slivers)
            if aspect_ratio > 4.5 and confidence < 0.50:
                continue
            if (box_w * box_h) > (0.65 * w * h):
                continue

            # 2. Optical Hollow-Lens & Tall Human Rejection Filter:
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if is_hollow_eyeglasses(crop):
                continue
            if is_human_or_face_false_positive(crop, aspect_ratio, confidence):
                continue

            # Fallback track ID when ByteTrack doesn't immediately assign one
            if track_id is None:
                track_id = 1

            if track_id not in tracks_db:
                tracks_db[track_id] = TargetKinematics(track_id)
            target_kin = tracks_db[track_id]
            
            # 3D Center coordinates relative to camera optical axis
            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0

            # 3. Dynamic Airframe Classification & Auto-Profile Resolution (Zero-Lag Throttled: max 1 per frame)
            needs_classification = type_classifier.enabled and (target_kin.cached_type == "UNKNOWN" or (frame_count - target_kin.last_classified_frame >= 45))
            if needs_classification and classified_in_this_frame < 1 and crop.size > 0:
                frame_type, frame_type_conf = type_classifier.classify(crop, device=DEVICE)
                classified_in_this_frame += 1
                target_kin.last_classified_frame = frame_count
                drone_type, drone_type_score, auto_prof = target_kin.update_type(frame_type, frame_type_conf, aspect_ratio=aspect_ratio)
            else:
                drone_type, drone_type_score, auto_prof = target_kin.cached_type, target_kin.cached_type_score, target_kin.auto_profile

            if auto_prof is None:
                auto_prof = get_drone_dimensions(drone_type, aspect_ratio=aspect_ratio)

            # Auto-calibrated physical dimension profile for this specific target
            distance_profile = auto_prof
            latest_target_width = distance_profile["width"]

            # Compute Multi-Cue Fused Monocular Distance & CRLB (Derivations 1, 2, 3A)
            z_est, sigma_d, ci_low, ci_high, crlb_var = compute_crlb_distance(
                box_w, box_h, distance_profile, FOCAL_LENGTH_PX, SIGMA_PIXEL,
                u_center, v_center, cx_cam, cy_cam
            )

            x_3d = ((u_center - cx_cam) * z_est) / FOCAL_LENGTH_PX
            y_3d = ((v_center - cy_cam) * z_est) / FOCAL_LENGTH_PX

            # Update Kinematics
            target_kin.update(frame_count, current_time, x_3d, y_3d, z_est, sigma_d, crlb_var)

            # Anti-Glitch Confirmation:
            # - Immediate confirmation for confident detections (>= 0.35)
            # - 2-hit confirmation for low-confidence detections (>= 0.22) to eliminate momentary body/head movements
            is_confirmed = (confidence >= 0.35) or (target_kin.hits >= 2 and confidence >= 0.22)
            if is_confirmed:
                confirmed_drone_count += 1
                disp_z = target_kin.smoothed_z if target_kin.smoothed_z is not None else z_est

                # Record Telemetry for Digital Twin & Flight Log
                telemetry_records.append({
                    "track_id": track_id,
                    "confidence": confidence,
                    "position_3d": [x_3d, y_3d, disp_z],
                    "crlb_sigma": sigma_d,
                    "velocity_3d": target_kin.velocity_3d,
                    "speed_kmh": target_kin.speed_kmh,
                    "approach_rate": target_kin.approach_rate,
                    "eta_s": target_kin.eta_seconds,
                    "drone_type": drone_type,
                    "domain": target_kin.cached_domain,
                    "drone_type_confidence": drone_type_score,
                    "span_m": distance_profile["width"],
                    "bbox": [x1, y1, x2, y2]
                })

                # Threat Zone & Domain Classification Styling
                domain = target_kin.cached_domain
                if domain == "MILITARY":
                    box_color = (0, 0, 255)       # Red: Military Drone
                    domain_label = "MILITARY"
                    domain_text_color = (0, 70, 255)
                elif domain == "CIVILIAN":
                    box_color = (0, 255, 120)     # Green/Cyan: Civilian Drone
                    domain_label = "CIVILIAN"
                    domain_text_color = (0, 255, 120)
                else:
                    box_color = (0, 215, 255)     # Amber/Yellow: Unknown Drone
                    domain_label = "UNKNOWN"
                    domain_text_color = (0, 215, 255)

                # Draw Target Box & Corner Reticles
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                corner_len = max(6, min(18, box_w // 3, box_h // 3))
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 2)
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 2)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 2)

                # Center Reticle Target Point
                cv2.circle(frame, (int(u_center), int(v_center)), 3, (0, 255, 255), -1)

                # --- RESOLUTION-SCALED CRISP TACTICAL HUD BADGE ---
                scale_res = max(0.9, w / 1280.0)
                if sigma_d < 0.20:
                    err_str = f"+/-{sigma_d*100:.0f}cm"
                else:
                    err_str = f"+/-{sigma_d:.1f}m"

                type_conf_str = f" {drone_type_score:.0%}" if drone_type != "UNKNOWN" else ""
                
                line1 = f"[{domain_label}] #{track_id} ({confidence*100:.0f}%) | {disp_z:.2f}m ({err_str})"
                line2 = f"TYPE: {drone_type}{type_conf_str} | SPAN: {distance_profile['width']*100:.0f}cm"

                badge_w = int(max(250, min(370, box_w + 35)) * scale_res)
                badge_h = int(42 * scale_res)
                
                # Smart badge vertical positioning: if not enough room above or near top header, put below
                if y1 - badge_h - 4 < 44:
                    badge_y1 = min(h - badge_h - 2, y2 + 4)
                    badge_y2 = badge_y1 + badge_h
                else:
                    badge_y1 = max(44, y1 - badge_h - 4)
                    badge_y2 = badge_y1 + badge_h

                badge_x1 = max(2, min(w - badge_w - 2, x1))
                badge_x2 = badge_x1 + badge_w

                # Semi-transparent HUD overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 18, 18), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, 1)

                cv2.putText(frame, line1, (badge_x1 + int(7 * scale_res), badge_y1 + int(18 * scale_res)), cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale_res, domain_text_color, 1, cv2.LINE_AA)
                cv2.putText(frame, line2, (badge_x1 + int(7 * scale_res), badge_y1 + int(34 * scale_res)), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale_res, (230, 230, 230), 1, cv2.LINE_AA)

    # Periodic Telemetry CSV Append
    if telemetry_records and frame_count % 3 == 0:
        with open(telemetry_csv_path, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            for rec in telemetry_records:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), frame_count, rec["track_id"],
                    f"{rec['confidence']:.2f}", rec["drone_type"], rec["domain"],
                    f"{rec['drone_type_confidence']:.2f}", f"{rec['span_m']:.2f}",
                    f"{rec['position_3d'][0]:.2f}", f"{rec['position_3d'][1]:.2f}",
                    f"{rec['position_3d'][2]:.2f}", f"{rec['crlb_sigma']:.2f}",
                    f"{rec['speed_kmh']:.1f}", f"{rec['approach_rate']:.1f}",
                    f"{rec['eta_s']:.1f}" if rec["eta_s"] else ""
                ])

    # Stale track cleanup
    stale_ids = [tid for tid, obj in tracks_db.items() if frame_count - obj.last_frame > MAX_MISSED_FRAMES]
    for tid in stale_ids:
        del tracks_db[tid]

    # ---------------------------------------------------------
    # 6. TOP HEADER TELEMETRY BAR
    # ---------------------------------------------------------
    header_color = (0, 0, 180) if confirmed_drone_count > 0 else (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 42), header_color, -1)
    
    status_msg = f"AIRSPACE ALERT: {confirmed_drone_count} ACTIVE TARGET(S)" if confirmed_drone_count > 0 else "AIRSPACE SURVEILLANCE: SCANNING (CRLB AUTO-PROFILE ACTIVE)"
    cv2.putText(frame, status_msg, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 255) if confirmed_drone_count > 0 else (0, 255, 0), 2, cv2.LINE_AA)

    calib_tag = f"CALIB: F={FOCAL_LENGTH_PX:.0f}px"
    source_tag = f"VID: {video_filename}" if is_video_file else f"CAM (1080p)"
    pause_tag = " [PAUSED]" if is_paused else ""
    p2_tag = " | P2-HEAD" if is_p2_model else ""
    header_right = f"{source_tag} | {calib_tag}{p2_tag}{pause_tag} | FPS: {fps:.1f}"
    (rw, _), _ = cv2.getTextSize(header_right, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(frame, header_right, (w - rw - 15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # ---------------------------------------------------------
    # 7. BOTTOM TACTICAL FOOTER HUD (Sensitivity & Calibration)
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

    # Key helpers prompt
    if is_video_file:
        controls_msg = "[K]: Calib Focal | [SPACE]: Pause | [R]: Replay | Mode: [T] | Quit: Q"
    else:
        clahe_tag = "[ON]" if clahe_enabled else ""
        controls_msg = f"[K]: Calib Focal | Contrast: [C]{clahe_tag} | Mode: [T] | Sens: [ / ] | Quit: Q"
        
    (cw, _), _ = cv2.getTextSize(controls_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, controls_msg, (w - cw - 15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Calibration Status Banner Overlay
    if calib_status_text and time.time() < calib_status_expiry:
        banner_w = 480
        banner_h = 32
        bx1 = (w - banner_w) // 2
        by1 = h - footer_h - banner_h - 10
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 100, 0), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 255, 0), 2)
        cv2.putText(frame, calib_status_text, (bx1 + 15, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

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
        target_nominal_profile = DRONE_PROFILES[active_profile_id]
        target_nominal_width = target_nominal_profile["width"]
        print(f"[*] Switched Drone Size Override Profile: {DRONE_PROFILES[active_profile_id]['desc']}")
    elif key in (ord("k"), ord("K")):
        # Auto-Calibrate Focal Length and Save Persistently for Active Camera Profile
        calib_dist = args.calibration_distance if args.calibration_distance is not None else 1.0
        if latest_box_width is None:
            print("[!] No drone bounding box visible. Keep target in frame and press [K].")
            calib_status_text = "ERROR: No Target Box Visible to Calibrate"
            calib_status_expiry = time.time() + 3.0
        else:
            calibrated_focal_length_px = (latest_box_width * calib_dist) / max(0.05, float(latest_target_width))
            FOCAL_LENGTH_PX = calibrated_focal_length_px
            res_str = f"{w}x{h}"
            save_camera_calibration(ACTIVE_DEVICE_KEY, calibrated_focal_length_px, calib_dist, latest_target_width, resolution=res_str)
            calib_status_text = f"SAVED [{ACTIVE_DEVICE_KEY}]: F = {calibrated_focal_length_px:.1f}px ({res_str})"
            calib_status_expiry = time.time() + 4.0
            print(f"[*] Camera Focal Length Calibrated & Saved for [{ACTIVE_DEVICE_KEY}]: {calibrated_focal_length_px:.1f}px")
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

