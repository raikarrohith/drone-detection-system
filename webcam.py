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
    get_drone_dimensions
)
import torch
from distance_residual_gru import DistanceResidualPredictor

parser = argparse.ArgumentParser(description="Drone Defense & CRLB Distance Estimation System")
parser.add_argument("--video", "-v", type=str, default=None, help="Path to test video file (e.g. drone_test.mp4)")
parser.add_argument("--camera", "-c", type=int, default=None, help="Camera index (e.g. 0, 1, 2)")
parser.add_argument("--imgsz", type=int, default=None, help="Inference resolution: 640 (fastest/realtime CPU) or 960/1280 (GPU)")
parser.add_argument("--type-model", type=str, default="drone_classifier/model/drone_cnn_best.pth", help="Optional civilian/military classification model")
parser.add_argument("--type-confidence", type=float, default=0.25, help="Minimum drone-type classifier confidence (0-1)")
parser.add_argument("--drone-width", type=float, default=None, help="Measured rotor-tip-to-tip span in metres; overrides the selected profile width")
parser.add_argument("--focal-length-px", type=float, default=None, help="Calibrated focal length in pixels for the active camera resolution")
parser.add_argument("--calibration-distance", type=float, default=None, help="Known target distance in metres; press K while the drone is detected to calibrate focal length")
parser.add_argument("--res", type=str, default="1080p", choices=["1080p", "720p", "480p"], help="Camera capture resolution: 1080p (Logitech HD default) or 720p/480p")
parser.add_argument("--gru", action="store_true", default=True, help="Enable GRU residual distance correction (default: True)")
parser.add_argument("--no-gru", dest="gru", action="store_false", help="Disable GRU residual distance correction")
parser.add_argument("--gru-model", type=str, default="models/distance_gru.pth", help="Path to trained GRU model")
parser.add_argument("--gru-scaler", type=str, default="models/distance_gru_scaler.json", help="Path to GRU scaler JSON")
parser.add_argument("video_pos", nargs="?", default=None, help="Positional video file path")
args, _ = parser.parse_known_args()

video_source = args.video if args.video else args.video_pos


# Auto-detect best hardware device (NVIDIA CUDA, Apple Silicon MPS, or multi-threaded CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
    current_imgsz = args.imgsz if args.imgsz is not None else 1280
    print("[*] Hardware Acceleration: NVIDIA CUDA GPU (Ultra-Range 1280px Active)")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    current_imgsz = args.imgsz if args.imgsz is not None else 960
    print("[*] Hardware Acceleration: Apple Silicon Metal GPU (High Performance)")
else:
    DEVICE = "cpu"
    current_imgsz = args.imgsz if args.imgsz is not None else 640
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f"[*] Hardware Acceleration: Multi-Threaded CPU ({torch.get_num_threads()} threads, imgsz={current_imgsz})")

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

# Distance Residual GRU Predictor Initialization
USE_GRU_CORRECTION = bool(args.gru)
gru_predictor = DistanceResidualPredictor(model_path=args.gru_model, scaler_path=args.gru_scaler, seq_len=10)
if gru_predictor.enabled:
    print(f"[*] Distance Residual GRU Engine: ONLINE (Active: {USE_GRU_CORRECTION}, Model: {args.gru_model})")
else:
    print(f"[*] Distance Residual GRU Engine: UNAVAILABLE (Using standard D2 + Kalman baseline)")

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

    res_map = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "480p": (640, 480)
    }
    target_w, target_h = res_map.get(args.res, (1280, 720))

    target_cams = [args.camera] if args.camera is not None else [1, 2, 0, 3]
    for cam_idx in target_cams:
        temp_cap = cv2.VideoCapture(cam_idx, backend)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            try:
                # Force MJPG codec first to eliminate USB bandwidth bottlenecks & frame queue lag
                temp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
                temp_cap.set(cv2.CAP_PROP_FPS, 30)
                temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
                active_cam_idx = cam_idx
                actual_w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or target_w)
                actual_h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or target_h)
                print(f"[*] Successfully connected to Live Camera (index {cam_idx}) at {actual_w}x{actual_h} (MJPG Zero-Latency)")
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

# Bounding Box Stabilization & Anti-Flicker Parameters
BBOX_TIGHTNESS = 0.98      # Tight airframe fitting factor (minimizes loose margin)
EMA_SMOOTH_MIN = 0.65      # Strong temporal smoothing factor for stationary/hovering targets (kills jitter)
EMA_SMOOTH_MAX = 0.90      # Dynamic smoothing factor for high-speed maneuvering targets
MAX_COAST_FRAMES = 6       # 6-frame coasting buffer to eliminate flicker on sideways profile rotations

# Drone Physical Size Profiles (Wingspan, Height, and Diagonal in meters)
DRONE_PROFILES = {
    0: {"name": "Palm/Neo", "width": 0.16, "height": 0.05, "desc": "DJI Neo / BetaFPV / Palm (16x5cm) [DEFAULT]"},
    1: {"name": "Micro/Mini", "width": 0.24, "height": 0.08, "desc": "DJI Mini / Avata (24x8cm)"},
    2: {"name": "Standard Quad", "width": 0.38, "height": 0.14, "desc": "Mavic / Phantom / FPV (38x14cm)"},
    3: {"name": "Heavy Lift", "width": 0.75, "height": 0.30, "desc": "Matrice / Hexacopter (75x30cm)"},
    4: {"name": "Tactical Wing", "width": 1.40, "height": 0.35, "desc": "Fixed-Wing / Loitering (140x35cm)"}
}
manual_profile_override = False
active_profile_id = 0  # Default to DJI Neo (16x5cm)
active_distance_mode = "width"  # "width", "height", or "fusion" (combined W+H)
if args.drone_width is not None:
    if args.drone_width <= 0:
        parser.error("--drone-width must be greater than zero")
    manual_profile_override = True
    profile_ratio = DRONE_PROFILES[2]["height"] / DRONE_PROFILES[2]["width"]
    target_nominal_profile = {
        "name": "Measured Drone",
        "width": args.drone_width,
        "height": args.drone_width * profile_ratio,
        "desc": f"Measured width ({args.drone_width * 100:.1f}cm)",
        "auto": False
    }
else:
    target_nominal_profile = DRONE_PROFILES[active_profile_id]
target_nominal_width = target_nominal_profile["width"]

def compute_crlb_distance(pixel_w, pixel_h, target_profile, focal_length, sigma_w, u_center=None, v_center=None, cx_cam=None, cy_cam=None, est_mode="width"):
    """
    Computes Deterministic Monocular Distance with 3 selectable estimators:
    1. Width-Only:    D_w   = (W_physical * F) / pixel_w * cos(theta)
    2. Height-Only:   D_h   = (H_physical * F) / pixel_h * cos(theta)
    3. Combined W+H:  D_wh  = (Diag_physical * F) / Diag_pixel * cos(theta)
    """
    target_w = target_profile["width"]
    target_h = target_profile.get("height", target_w * 0.35)
    
    pw = max(2.0, float(pixel_w))
    ph = max(2.0, float(pixel_h))
    
    # 1. Perspective ray angle correction
    if u_center is not None and v_center is not None and cx_cam is not None and cy_cam is not None:
        r_off = math.sqrt((u_center - cx_cam)**2 + (v_center - cy_cam)**2)
        cos_theta = focal_length / math.sqrt(focal_length**2 + r_off**2)
    else:
        cos_theta = 1.0
        
    # Compute all 3 distance estimators
    d_width = ((target_w * focal_length) / pw) * cos_theta
    d_height = ((target_h * focal_length) / ph) * cos_theta
    
    target_diag = math.sqrt(target_w**2 + target_h**2)
    pixel_diag = math.sqrt(pw**2 + ph**2)
    d_fusion = ((target_diag * focal_length) / pixel_diag) * cos_theta
    
    if est_mode == "height":
        distance_z = d_height
        target_span = target_h
    elif est_mode in ("fusion", "combined", "both", "wh"):
        distance_z = d_fusion
        target_span = target_diag
    else:  # "width"
        distance_z = d_width
        target_span = target_w
        
    fisher = (target_span**2 * focal_length**2) / ((sigma_w**2) * (max(0.1, distance_z)**4))
    crlb_var_pixel = 1.0 / max(1e-9, fisher)
    crlb_var_pose = (POSE_ASPECT_RATIO_UNCERTAINTY * distance_z) ** 2
    crlb_var_calib = (CALIBRATION_RELATIVE_UNCERTAINTY * distance_z) ** 2
    total_crlb_variance = crlb_var_pixel + crlb_var_pose + crlb_var_calib
    sigma_d = math.sqrt(total_crlb_variance)
    
    ci_lower = max(0.05, distance_z - 2.0 * sigma_d)
    ci_upper = distance_z + 2.0 * sigma_d
    
    return distance_z, sigma_d, ci_lower, ci_upper, total_crlb_variance, d_width, d_height, d_fusion

def refine_bbox_contours(frame, bbox):
    """
    Refines loose YOLO bounding boxes around real physical drones.
    Trims off empty background margins and mirror tabletop reflections by analyzing
    horizontal & vertical edge energy and structural intensity profiles.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(w_img - 1, int(x1)))
    y1 = max(0, min(h_img - 1, int(y1)))
    x2 = max(x1 + 15, min(w_img, int(x2)))
    y2 = max(y1 + 15, min(h_img, int(y2)))
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return bbox
        
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.uint8(np.clip(mag, 0, 255))
    
    # Canny edges with clean threshold
    canny = cv2.Canny(gray, 40, 120)
    struct_mask = cv2.bitwise_or(canny, (mag > 45).astype(np.uint8) * 255)
    
    # Horizontal projection (to find left and right airframe boundaries)
    h_proj = np.sum(struct_mask > 0, axis=0)
    # Vertical projection (to find top and bottom airframe boundaries)
    v_proj = np.sum(struct_mask > 0, axis=1)
    
    min_col_pixels = max(4, int(0.08 * ch))
    min_row_pixels = max(4, int(0.08 * cw))
    
    cols_active = np.where(h_proj >= min_col_pixels)[0]
    rows_active = np.where(v_proj >= min_row_pixels)[0]
    
    if len(cols_active) > 4 and len(rows_active) > 4:
        trim_x1 = cols_active[0]
        trim_x2 = cols_active[-1]
        trim_y1 = rows_active[0]
        trim_y2 = rows_active[-1]
        
        # When on a reflective desk, bottom edge is often near 70-80% height of the raw box
        # If the lower 25% has lower edge density than upper part, trim the reflection
        mid_y = (trim_y1 + trim_y2) // 2
        upper_density = np.mean(v_proj[trim_y1:mid_y]) if mid_y > trim_y1 else 1.0
        lower_quarter = int(trim_y1 + 0.75 * (trim_y2 - trim_y1))
        if lower_quarter < trim_y2:
            lower_density = np.mean(v_proj[lower_quarter:trim_y2])
            if lower_density < 0.35 * upper_density:
                trim_y2 = lower_quarter
        
        new_w = (trim_x2 - trim_x1)
        new_h = (trim_y2 - trim_y1)
        
        # Only accept refinement if it represents a valid drone body
        if new_w >= 0.30 * cw and new_h >= 0.20 * ch:
            final_x1 = max(0, x1 + trim_x1)
            final_y1 = max(0, y1 + trim_y1)
            final_x2 = min(w_img - 1, x1 + trim_x2)
            final_y2 = min(h_img - 1, y1 + trim_y2)
            return (final_x1, final_y1, final_x2, final_y2)
            
    return bbox

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
    Discriminates humans, faces, and moving heads from airborne and desk drones.
    - Dual HSV + YCrCb chrominance detects human facial skin across complexions and lighting.
    - Rejects moving heads, faces, and nearby humans while preserving flying and desk drones (including sideways/tilted orientations).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    # 1. Dual HSV + YCrCb Chromaticity Skin Analysis
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([28, 255, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([168, 20, 40]), np.array([180, 255, 255]))
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 128, 70]), np.array([255, 182, 138]))
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    skin_ratio = np.count_nonzero(skin_mask) / max(1, skin_mask.size)
    
    # Human face/head signature: requires actual skin tone presence (> 14%) on near-square/vertical crops
    if aspect_ratio < 1.30 and skin_ratio > 0.14 and confidence < 0.70:
        return True
        
    # High skin-tone coverage (> 22% regardless of aspect ratio) indicates hand/arm/body close-up
    if skin_ratio > 0.22 and confidence < 0.65:
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
        self.feature_sequence = []
        self.last_d2_raw = 0.0
        self.last_d2_corrected = 0.0
        self.last_gru_correction = 0.0
        self.prev_raw_z = None
        self.hits = 0
        self.missed_frames = 0
        self.last_frame = 0
        self.last_timestamp = None
        self.kalman_filter = None
        self.smoothed_z = None
        self.smoothed_bbox = None  # [x1, y1, x2, y2]
        self.velocity_2d_px = (0.0, 0.0)  # vx_px, vy_px (for coasting)
        self.velocity_3d = (0.0, 0.0, 0.0)  # vx, vy, vz in m/s
        self.speed_kmh = 0.0
        self.approach_rate = 0.0  # m/s (+ approaching, - receding)
        self.eta_seconds = None
        self.bayes_classifier = BayesianDroneClassifier()
        self.type_votes = {}
        self.cached_type = "UNKNOWN"
        self.cached_domain = "UNKNOWN"
        self.cached_type_score = 0.0
        self.auto_profile = None
        self.last_classified_frame = -10
        self.last_conf = 0.0
        self.last_sigma_d = 0.5
        self.last_ci_low = 0.1
        self.last_ci_high = 10.0
        self.last_crlb_var = 0.25
        self.last_u_center = 0.0
        self.last_v_center = 0.0
        self.last_distance_profile = None

    def update_type(self, label, confidence, aspect_ratio=2.0):
        """Smooth classifications over a track and dynamically auto-calibrate dimensions."""
        for k in list(self.type_votes.keys()):
            self.type_votes[k] *= 0.88
            
        if label and label != "UNKNOWN":
            self.type_votes[label] = self.type_votes.get(label, 0.0) + confidence

        if self.type_votes:
            best_label, score = max(self.type_votes.items(), key=lambda item: item[1])
            if score >= 0.25:
                self.cached_type = best_label
                self.cached_domain = get_drone_domain(best_label)
                self.cached_type_score = min(0.99, score / (score + 0.15))
            else:
                self.cached_type = "UNKNOWN"
                self.cached_domain = "UNKNOWN"
                self.cached_type_score = 0.0
        else:
            self.cached_type = "UNKNOWN"
            self.cached_domain = "UNKNOWN"
            self.cached_type_score = 0.0

        self.auto_profile = get_drone_dimensions(self.cached_type, aspect_ratio=aspect_ratio)
        return self.cached_type, self.cached_type_score, self.auto_profile

    def update(self, frame_num, timestamp, raw_bbox, confidence, distance_profile, focal_len, sigma_pixel, cx_cam, cy_cam, tightness=0.98):
        self.hits += 1
        self.missed_frames = 0
        self.last_frame = frame_num
        self.last_conf = (0.65 * self.last_conf + 0.35 * confidence) if self.hits > 1 else confidence
        self.last_distance_profile = distance_profile

        dt = (timestamp - self.last_timestamp) if (self.last_timestamp is not None) else 0.033
        if dt <= 0 or dt > 1.5:
            dt = 0.033
        self.last_timestamp = timestamp

        # 1. Apply Bounding Box Tightness & Center Extraction
        rx1, ry1, rx2, ry2 = raw_bbox
        rcx = (rx1 + rx2) / 2.0
        rcy = (ry1 + ry2) / 2.0
        rw = (rx2 - rx1) * tightness
        rh = (ry2 - ry1) * tightness
        tx1 = rcx - rw / 2.0
        ty1 = rcy - rh / 2.0
        tx2 = rcx + rw / 2.0
        ty2 = rcy + rh / 2.0

        # 2. Adaptive Exponential Moving Average (EMA) Bounding Box Smoothing
        if self.smoothed_bbox is None:
            self.smoothed_bbox = [float(tx1), float(ty1), float(tx2), float(ty2)]
        else:
            prev_cx = (self.smoothed_bbox[0] + self.smoothed_bbox[2]) / 2.0
            prev_cy = (self.smoothed_bbox[1] + self.smoothed_bbox[3]) / 2.0
            prev_w = self.smoothed_bbox[2] - self.smoothed_bbox[0]
            prev_h = self.smoothed_bbox[3] - self.smoothed_bbox[1]
            pos_jump = math.hypot(rcx - prev_cx, rcy - prev_cy)
            size_jump = max(abs(rw - prev_w), abs(rh - prev_h)) * 1.5
            total_jump = pos_jump + size_jump
            
            # Compute 2D pixel velocity for coasting dead reckoning
            if dt > 0.01:
                vx_p = (rcx - prev_cx) / dt
                vy_p = (rcy - prev_cy) / dt
                self.velocity_2d_px = (0.7 * self.velocity_2d_px[0] + 0.3 * vx_p,
                                       0.7 * self.velocity_2d_px[1] + 0.3 * vy_p)

            # Adaptive alpha: instant response when shifting distance, stable when hovering
            alpha = min(0.90, max(0.35, 0.35 + (total_jump / 12.0) * 0.55))
            self.smoothed_bbox[0] = alpha * tx1 + (1.0 - alpha) * self.smoothed_bbox[0]
            self.smoothed_bbox[1] = alpha * ty1 + (1.0 - alpha) * self.smoothed_bbox[1]
            self.smoothed_bbox[2] = alpha * tx2 + (1.0 - alpha) * self.smoothed_bbox[2]
            self.smoothed_bbox[3] = alpha * ty2 + (1.0 - alpha) * self.smoothed_bbox[3]

        sx1, sy1, sx2, sy2 = self.smoothed_bbox
        box_w = max(2.0, sx2 - sx1)
        box_h = max(2.0, sy2 - sy1)
        u_center = (sx1 + sx2) / 2.0
        v_center = (sy1 + sy2) / 2.0
        self.last_u_center = u_center
        self.last_v_center = v_center

        # 3. Compute CRLB Distance with Multi-Estimator Support (W, H, W+H)
        z_est, sigma_d, ci_low, ci_high, crlb_var, dw, dh, dwh = compute_crlb_distance(
            box_w, box_h, distance_profile, focal_len, sigma_pixel,
            u_center, v_center, cx_cam, cy_cam, est_mode=active_distance_mode
        )
        self.last_d2_raw = z_est
        self.last_sigma_d = sigma_d
        self.last_ci_low = ci_low
        self.last_ci_high = ci_high
        self.last_crlb_var = crlb_var

        # 4. Neural GRU Residual Correction (delta_D)
        delta_d2 = (z_est - self.prev_raw_z) if self.prev_raw_z is not None else 0.0
        self.prev_raw_z = z_est
        v_2d_mag = math.hypot(self.velocity_2d_px[0], self.velocity_2d_px[1])

        if self.kalman_filter is not None:
            kalman_pred_z, _ = self.kalman_filter.predict(dt)
        else:
            kalman_pred_z = z_est

        feat_vec = [
            float(z_est),
            float(box_w),
            float(box_h),
            float(box_w / max(1.0, box_h)),
            float(delta_d2),
            float(v_2d_mag),
            float(confidence),
            float(sigma_d),
            float(kalman_pred_z)
        ]
        self.feature_sequence.append(feat_vec)
        if len(self.feature_sequence) > 30:
            self.feature_sequence.pop(0)

        if USE_GRU_CORRECTION and gru_predictor.enabled and len(self.feature_sequence) >= 10:
            delta_d = gru_predictor.predict_correction(self.feature_sequence)
            self.last_gru_correction = delta_d
            z_corrected = max(0.1, z_est + delta_d)
            self.last_d2_corrected = z_corrected
            z_meas_for_kalman = z_corrected
        else:
            self.last_gru_correction = 0.0
            self.last_d2_corrected = z_est
            z_meas_for_kalman = z_est

        # Dynamic distance smoothing: fast response on distance shift, steady lock on hover
        d_alpha = min(0.90, max(0.40, 0.40 + (total_jump / 12.0) * 0.50)) if self.hits > 1 else 1.0
        if hasattr(self, "last_d_w") and self.last_d_w is not None and self.hits > 1:
            self.last_d_w = (1.0 - d_alpha) * self.last_d_w + d_alpha * dw
            self.last_d_h = (1.0 - d_alpha) * self.last_d_h + d_alpha * dh
            self.last_d_wh = (1.0 - d_alpha) * self.last_d_wh + d_alpha * dwh
        else:
            self.last_d_w = dw
            self.last_d_h = dh
            self.last_d_wh = dwh

        x_m = ((u_center - cx_cam) * z_meas_for_kalman) / focal_len
        y_m = ((v_center - cy_cam) * z_meas_for_kalman) / focal_len

        # 5. Adaptive CRLB Kalman State Estimation (1D Range & Kinematics)
        kalman_meas_noise = (0.35 * crlb_var) if (USE_GRU_CORRECTION and gru_predictor.enabled) else crlb_var
        if self.kalman_filter is None:
            self.kalman_filter = KalmanRangeFilter(z_meas_for_kalman, sigma_d)
            self.smoothed_z = z_meas_for_kalman
        else:
            filtered_z, filtered_vz = self.kalman_filter.update(z_meas_for_kalman, kalman_meas_noise)
            self.smoothed_z = max(0.1, filtered_z)

        self.history.append((timestamp, x_m, y_m, self.smoothed_z, sigma_d))
        if len(self.history) > 30:
            self.history.pop(0)

        # 5. Calculate 3D Velocity & Approach Rate if baseline exists
        if len(self.history) >= 4:
            t_old, x_old, y_old, z_old, _ = self.history[-4]
            dt_base = timestamp - t_old
            if dt_base > 0.05:
                vx = (x_m - x_old) / dt_base
                vy = (y_m - y_old) / dt_base
                vz = (self.smoothed_z - z_old) / dt_base
                self.velocity_3d = (vx, vy, vz)
                self.speed_kmh = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6
                self.approach_rate = -vz  # Positive when closing in
                if self.approach_rate > 0.8:
                    self.eta_seconds = max(0.1, self.smoothed_z / self.approach_rate)
                else:
                    self.eta_seconds = None

        return z_est, sigma_d, ci_low, ci_high, crlb_var, x_m, y_m

    def coast(self, dt):
        """Dead-reckoning coasting forward during 1-4 frame detection dropouts to eliminate flicker."""
        self.missed_frames += 1
        self.last_conf *= 0.90  # Gradual confidence decay
        if self.smoothed_bbox is not None and dt > 0:
            vx_p, vy_p = self.velocity_2d_px
            # Shift bounding box slightly with estimated 2D pixel velocity
            dx = vx_p * dt * 0.5  # Damped coast
            dy = vy_p * dt * 0.5
            self.smoothed_bbox[0] += dx
            self.smoothed_bbox[1] += dy
            self.smoothed_bbox[2] += dx
            self.smoothed_bbox[3] += dy
            self.last_u_center += dx
            self.last_v_center += dy
            if self.kalman_filter is not None:
                self.kalman_filter.predict(dt)
                self.smoothed_z = max(0.1, self.kalman_filter.z)

tracks_db = {}
MIN_CONSECUTIVE_FRAMES = 1  # Instant response for small mini drones
MAX_MISSED_FRAMES = 25      # Generous track retention for hand-held & distant tests

# UI State (Default 35% sensitivity: clean noise-free drone tracking)
conf_percent = 35
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
            "timestamp", "frame", "track_id", "confidence", "airframe_model",
            "model_confidence", "span_m", "x_m", "y_m", "distance_m",
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
        FOCAL_LENGTH_PX = calibrated_focal_length_px * (float(w) / 1920.0)
    else:
        FOCAL_LENGTH_PX = (float(w) / 1920.0) * 1350.0

    conf_threshold = conf_percent / 100.0

    # Hardware-Accelerated YOLO Inference with High-Sensitivity Multi-Stage Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=max(0.18, conf_threshold),
        iou=0.45,
        imgsz=current_imgsz,
        device=DEVICE,
        verbose=False
    )

    confirmed_drone_count = 0
    telemetry_records = []
    classified_this_frame = 0
    active_frame_track_ids = set()

    # Phase 1: Process YOLO Detections & Update Kinematic Trackers
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id != DRONE_CLASS_ID or confidence < max(0.18, conf_threshold):
                continue

            track_id = int(box.id[0]) if box.id is not None else 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = (x2 - x1), (y2 - y1)
            aspect_ratio = float(box_w) / max(1.0, float(box_h))

            # 1. Clutter & Flat Edge Rejection Filter (rejects desk lines, mousepads, camera pan motion blur)
            if box_h < 14 or box_w < 14:
                continue
            if aspect_ratio > 3.5 and confidence < 0.50:
                continue
            if aspect_ratio > 4.5:
                continue
            if (box_w * box_h) > (0.65 * w * h):
                continue

            # 2. Optical Hollow-Lens & Tall Human Rejection Filter:
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if is_hollow_eyeglasses(crop):
                continue
            if is_human_or_face_false_positive(crop, aspect_ratio, confidence):
                continue

            if track_id not in tracks_db:
                tracks_db[track_id] = TargetKinematics(track_id)
            target_kin = tracks_db[track_id]
            active_frame_track_ids.add(track_id)

            # 3. Dynamic Airframe Classification (Rate-limited for zero lag)
            should_classify = False
            if type_classifier.enabled and classified_this_frame < 1:
                if target_kin.cached_type == "UNKNOWN":
                    should_classify = (frame_count - target_kin.last_classified_frame >= 4)
                else:
                    should_classify = (frame_count - target_kin.last_classified_frame >= 25)

            if should_classify and crop.size > 0:
                frame_type, frame_type_conf = type_classifier.classify(crop, device=DEVICE)
                target_kin.last_classified_frame = frame_count
                classified_this_frame += 1
                drone_type, drone_type_score, auto_prof = target_kin.update_type(frame_type, frame_type_conf, aspect_ratio=aspect_ratio)
            else:
                drone_type, drone_type_score, auto_prof = target_kin.cached_type, target_kin.cached_type_score, target_kin.auto_profile

            if auto_prof is None:
                auto_prof = get_drone_dimensions(drone_type, aspect_ratio=aspect_ratio)

            # Physical dimension profile for this specific target (respect manual override)
            if manual_profile_override or args.drone_width is not None:
                distance_profile = target_nominal_profile
            else:
                distance_profile = auto_prof

            # Refine loose raw YOLO box by trimming background/reflections around drone
            refined_bbox = refine_bbox_contours(frame, (x1, y1, x2, y2))

            # Update target state with tighter box & adaptive EMA temporal smoothing
            target_kin.update(
                frame_num=frame_count,
                timestamp=current_time,
                raw_bbox=refined_bbox,
                confidence=confidence,
                distance_profile=distance_profile,
                focal_len=FOCAL_LENGTH_PX,
                sigma_pixel=SIGMA_PIXEL,
                cx_cam=cx_cam,
                cy_cam=cy_cam,
                tightness=1.0
            )

    # Phase 2: Coast / Dead-Reckon tracks missed in this frame (with Spatial Duplicate Suppression)
    for tid, target_kin in list(tracks_db.items()):
        if tid not in active_frame_track_ids:
            # Check if this coasted track is a duplicate ghost of any active detection (e.g. after rapid motion)
            is_duplicate = False
            if active_frame_track_ids and target_kin.smoothed_bbox is not None:
                for active_id in active_frame_track_ids:
                    active_kin = tracks_db.get(active_id)
                    if active_kin is not None and active_kin.smoothed_bbox is not None:
                        dx = target_kin.last_u_center - active_kin.last_u_center
                        dy = target_kin.last_v_center - active_kin.last_v_center
                        dist = math.hypot(dx, dy)
                        ax1, ay1, ax2, ay2 = active_kin.smoothed_bbox
                        active_span = max(ax2 - ax1, ay2 - ay1)
                        # If nearby (within 1.5x active box span or 120px), suppress the old ghost track
                        if dist < max(120.0, 1.5 * active_span):
                            is_duplicate = True
                            break

            if is_duplicate:
                target_kin.missed_frames = MAX_COAST_FRAMES + 1  # Expire immediately
            elif target_kin.hits >= 2 and target_kin.missed_frames < MAX_COAST_FRAMES:
                target_kin.coast(dt)
            else:
                target_kin.missed_frames += 1

    # Phase 2: Maintain target lock during brief frame drops (Locks on target without drift)
    for tid, target_kin in list(tracks_db.items()):
        if tid not in active_frame_track_ids:
            target_kin.missed_frames += 1
            if target_kin.kalman_filter is not None:
                target_kin.kalman_filter.predict(dt)

    # Phase 3: Render all locked drone targets (Strictly Green, Solid Rock Lock, Zero Flicker)
    for tid, target_kin in list(tracks_db.items()):
        if target_kin.smoothed_bbox is None:
            continue
            
        # Hold firm target lock across 1-4 missed frames so the box never flickers or drops
        if target_kin.missed_frames > 4:
            continue

        conf = target_kin.last_conf
        is_confirmed = (conf >= 0.18) or (target_kin.hits >= 3 and target_kin.missed_frames <= 3)
        if not is_confirmed:
            continue

        confirmed_drone_count += 1
        distance_profile = target_kin.last_distance_profile or target_nominal_profile
        drone_type = target_kin.cached_type
        drone_type_score = target_kin.cached_type_score
        # Display direct high-accuracy GRU distance (3.24% error) when active, otherwise Kalman smoothed
        if USE_GRU_CORRECTION and gru_predictor.enabled and target_kin.last_d2_corrected > 0.05:
            disp_z = target_kin.last_d2_corrected
        else:
            disp_z = target_kin.smoothed_z if target_kin.smoothed_z is not None else 1.0
        sigma_d = target_kin.last_sigma_d
        ci_low = target_kin.last_ci_low
        ci_high = target_kin.last_ci_high
        u_center = target_kin.last_u_center
        v_center = target_kin.last_v_center
        x_3d = ((u_center - cx_cam) * disp_z) / FOCAL_LENGTH_PX
        y_3d = ((v_center - cy_cam) * disp_z) / FOCAL_LENGTH_PX

        sx1, sy1, sx2, sy2 = map(int, target_kin.smoothed_bbox)
        # Clamp to frame boundary
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w - 1, sx2), min(h - 1, sy2)
        sbox_w = max(2, sx2 - sx1)
        sbox_h = max(2, sy2 - sy1)

        latest_box_width = sbox_w
        latest_target_width = distance_profile["width"]

        # Record Telemetry for Digital Twin & Flight Log
        telemetry_records.append({
            "track_id": tid,
            "confidence": conf,
            "position_3d": [x_3d, y_3d, disp_z],
            "crlb_sigma": sigma_d,
            "velocity_3d": target_kin.velocity_3d,
            "speed_kmh": target_kin.speed_kmh,
            "approach_rate": target_kin.approach_rate,
            "eta_s": target_kin.eta_seconds,
            "drone_type": drone_type,
            "drone_type_confidence": drone_type_score,
            "span_m": distance_profile["width"],
            "bbox": [sx1, sy1, sx2, sy2]
        })

        # Target Reticle Color: Strictly Tactical Green
        box_color = (0, 255, 0)
        status_tag = drone_type if drone_type != "UNKNOWN" else "DRONE"

        # Draw Target Box & Corner Reticles
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), box_color, 2)
        corner_len = max(8, min(24, sbox_w // 3, sbox_h // 3))
        cv2.line(frame, (sx1, sy1), (sx1 + corner_len, sy1), box_color, 3)
        cv2.line(frame, (sx1, sy1), (sx1, sy1 + corner_len), box_color, 3)
        cv2.line(frame, (sx2, sy1), (sx2 - corner_len, sy1), box_color, 3)
        cv2.line(frame, (sx2, sy1), (sx2, sy1 + corner_len), box_color, 3)
        cv2.line(frame, (sx1, sy2), (sx1 + corner_len, sy2), box_color, 3)
        cv2.line(frame, (sx1, sy2), (sx1, sy2 - corner_len), box_color, 3)
        cv2.line(frame, (sx2, sy2), (sx2 - corner_len, sy2), box_color, 3)
        cv2.line(frame, (sx2, sy2), (sx2, sy2 - corner_len), box_color, 3)

        # Center Reticle Target Point
        cv2.circle(frame, (int(u_center), int(v_center)), 4, box_color, -1)

        # --- HIGH-LEGIBILITY MULTI-LINE TARGET BADGE ---
        dw_val = target_kin.last_d_w if hasattr(target_kin, "last_d_w") else disp_z
        dh_val = target_kin.last_d_h if hasattr(target_kin, "last_d_h") else disp_z
        dwh_val = target_kin.last_d_wh if hasattr(target_kin, "last_d_wh") else disp_z

        # Adaptive error formatting
        if sigma_d < 0.20:
            err_str = f"+/-{sigma_d*100:.1f}cm"
            ci_str = f"{ci_low:.2f}-{ci_high:.2f}m"
        else:
            err_str = f"+/-{sigma_d:.2f}m"
            ci_str = f"{ci_low:.1f}-{ci_high:.1f}m"
            
        type_confidence_text = f" ({drone_type_score:.0%})" if drone_type != "UNKNOWN" else ""
        target_h_cm = distance_profile.get('height', distance_profile['width'] * 0.31) * 100.0

        # Motion status string and color
        if target_kin.approach_rate > 0.15:
            eta_str = f" | ETA: {target_kin.eta_seconds:.1f}s" if target_kin.eta_seconds else ""
            motion_str = f"CLOSING @ +{target_kin.approach_rate:.2f}m/s (+{target_kin.speed_kmh:.1f}km/h{eta_str})"
            motion_color = (0, 255, 255)
        elif target_kin.approach_rate < -0.15:
            motion_str = f"RECEDING @ {target_kin.approach_rate:.2f}m/s ({target_kin.speed_kmh:.1f}km/h)"
            motion_color = (120, 255, 120)
        else:
            motion_str = f"HOVERING / STATIONARY ({target_kin.speed_kmh:.1f}km/h)"
            motion_color = (220, 220, 220)

        gru_tag_str = f" [GRU: {target_kin.last_gru_correction:+.2f}m]" if (USE_GRU_CORRECTION and gru_predictor.enabled and abs(target_kin.last_gru_correction) > 0.001) else ""
        badge_line1 = f"DRONE [{tid}]: {conf*100:.0f}% | {status_tag}{type_confidence_text}"
        badge_line2 = f"DIST: {disp_z:.2f}m{gru_tag_str}  (CRLB: {err_str})"
        badge_line3 = f"MOTION: {motion_str}"

        # Target Bounding Box Badge Dimensions (Compact 3-Line Tactical Badge)
        badge_w = max(500, int(sbox_w + 120))
        badge_h = 92
        badge_y1 = max(46, sy1 - badge_h - 8)
        badge_y2 = sy1 - 8
        if badge_y1 <= 46:
            badge_y1 = min(h - badge_h - 50, sy2 + 8)
            badge_y2 = badge_y1 + badge_h

        bx1 = max(8, min(w - badge_w - 8, sx1))
        bx2 = bx1 + badge_w
        by1 = max(46, badge_y1)
        by2 = min(h - 50, badge_y2)

        # High-Contrast Solid Matte Opaque Target Badge
        if by2 > by1 and bx2 > bx1:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (10, 10, 10), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            cv2.putText(frame, badge_line1, (bx1 + 12, by1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, badge_line2, (bx1 + 12, by1 + 52), cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, badge_line3, (bx1 + 12, by1 + 78), cv2.FONT_HERSHEY_DUPLEX, 0.50, motion_color, 1, cv2.LINE_AA)

        # --- CORNER TACTICAL TELEMETRY CARD (Fixed Top-Left HUD Card) ---
        if confirmed_drone_count == 1:
            card_x1, card_y1 = 16, 52
            card_w, card_h = 560, 110
            card_x2, card_y2 = card_x1 + card_w, card_y1 + card_h
            
            # Matte Black Panel with Neon Accent
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (12, 12, 12), -1)
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (0, 200, 0), 2)
            cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y1 + 28), (0, 140, 0), -1)
            
            c_header = f"TACTICAL TELEMETRY | TARGET [ID:{tid}] : {drone_type.upper()}"
            cv2.putText(frame, c_header, (card_x1 + 12, card_y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
            
            c_dist = f"DISTANCE :  {disp_z:.2f} m{gru_tag_str}   [CRLB: {err_str} | 95% CI: {ci_str}]"
            cv2.putText(frame, c_dist, (card_x1 + 12, card_y1 + 54), cv2.FONT_HERSHEY_DUPLEX, 0.66, (0, 255, 0), 2, cv2.LINE_AA)
            
            c_mot = f"KINEMATICS:  {motion_str}  [SPAN: {distance_profile['width']*100:.0f}x{target_h_cm:.0f}cm]"
            cv2.putText(frame, c_mot, (card_x1 + 12, card_y1 + 88), cv2.FONT_HERSHEY_DUPLEX, 0.50, motion_color, 1, cv2.LINE_AA)


    # Periodic Telemetry CSV Append
    if telemetry_records and frame_count % 3 == 0:
        with open(telemetry_csv_path, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            for rec in telemetry_records:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), frame_count, rec["track_id"],
                    f"{rec['confidence']:.2f}", rec["drone_type"],
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
    clahe_tag = "[ON]" if clahe_enabled else ""
    gru_hud_tag = "[ON]" if USE_GRU_CORRECTION else "[OFF]"
    if is_video_file:
        controls_msg = f"[S]: Shot | [G]: GRU{gru_hud_tag} | [K]: Calib | [SPACE]: Pause | [R]: Replay | Sens: [ / ] | Q: Quit"
    else:
        controls_msg = f"[S]: Shot | [G]: GRU{gru_hud_tag} | [K]: Calib | [C]: Contrast{clahe_tag} | Sens: [ / ] | Mode: [T] | Q: Quit"
        
    (cw, _), _ = cv2.getTextSize(controls_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(frame, controls_msg, (w - cw - 15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Calibration / Screenshot Status Banner Overlay
    if calib_status_text and time.time() < calib_status_expiry:
        banner_w = 540
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
    elif key in (ord("s"), ord("S")):  # S: Take High-Res Screenshot of Full HUD & Detection Screen
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        shot_path = screenshots_dir / f"capture_{ts_str}.jpg"
        cv2.imwrite(str(shot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        calib_status_text = f"SCREENSHOT SAVED: {shot_path.name}"
        calib_status_expiry = time.time() + 3.0
        print(f"\n[+] Full HUD Screenshot Saved: {shot_path.resolve()}\n")
    elif key in (ord("m"), ord("M")):  # M: Cycle Distance Estimation Mode (Width -> Height -> Fusion)
        modes = ["width", "height", "fusion"]
        curr_m_idx = modes.index(active_distance_mode) if active_distance_mode in modes else 0
        active_distance_mode = modes[(curr_m_idx + 1) % len(modes)]
        calib_status_text = f"DISTANCE ESTIMATOR: {active_distance_mode.upper()} (Key [M] to cycle)"
        calib_status_expiry = time.time() + 3.0
        print(f"[*] Switched Distance Estimation Mode: {active_distance_mode.upper()}")
    elif key in (ord("g"), ord("G")):  # G: Toggle GRU Residual Distance Correction
        USE_GRU_CORRECTION = not USE_GRU_CORRECTION
        calib_status_text = f"GRU RESIDUAL CORRECTION: {'ENABLED' if USE_GRU_CORRECTION else 'DISABLED'} (Key [G])"
        calib_status_expiry = time.time() + 3.0
        print(f"[*] GRU Residual Correction: {'ENABLED' if USE_GRU_CORRECTION else 'DISABLED'}")
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
    elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4")):
        active_profile_id = int(chr(key))
        manual_profile_override = True
        target_nominal_profile = DRONE_PROFILES[active_profile_id]
        target_nominal_width = target_nominal_profile["width"]
        print(f"[*] Switched Drone Size Override Profile: {DRONE_PROFILES[active_profile_id]['desc']}")
    elif key in (ord("a"), ord("A")):
        manual_profile_override = False
        print("[*] Reverted to Dynamic AI Auto-Classification Profiles")
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

