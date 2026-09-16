import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load fine-tuned drone detector model
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

# Drone class index lookup (default 0)
DRONE_CLASS_ID = 0
for cls_id, name in model.names.items():
    if "drone" in name.lower():
        DRONE_CLASS_ID = cls_id
        break

# 2. External & Built-in Camera Setup (DirectShow for Windows, AVFoundation/Any for macOS)
cap = None
is_windows = sys.platform.startswith("win")
backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY

# Prioritize external cameras (index 1, 2, then 0)
for cam_idx in [1, 2, 0, 3]:
    temp_cap = cv2.VideoCapture(cam_idx, backend)
    if not temp_cap.isOpened():
        temp_cap = cv2.VideoCapture(cam_idx)
    if temp_cap.isOpened():
        # Test reading a frame to confirm camera is active
        ret, test_frame = temp_cap.read()
        if ret and test_frame is not None:
            cap = temp_cap
            print(f"[*] Successfully connected to camera (index {cam_idx})")
            break
        else:
            temp_cap.release()

if cap is None or not cap.isOpened():
    print("ERROR: Could not open any camera. Please ensure external camera is plugged in.")
    exit(1)

# Set high resolution for maximum long-distance pixel clarity (1080p -> 720p fallback)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

WINDOW_NAME = "Drone Detection System - Long Range Tracker"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Real-time state: Default 35% sensitivity for long-distance drone capture
conf_percent = 35
brightness_boost = 0
dragging_slider = False

def on_trackbar(val):
    global conf_percent
    conf_percent = max(10, min(95, val))

try:
    cv2.createTrackbar("Conf (%)", WINDOW_NAME, conf_percent, 95, on_trackbar)
except Exception:
    pass

# Slider layout configuration for On-Screen HUD (OSD)
SLIDER_X1 = 145
SLIDER_X2 = 520
SLIDER_Y1 = 0
SLIDER_Y2 = 0

def handle_mouse(event, x, y, flags, param):
    global conf_percent, dragging_slider, SLIDER_X1, SLIDER_X2, SLIDER_Y1, SLIDER_Y2
    if event == cv2.EVENT_LBUTTONDOWN:
        if SLIDER_X1 - 15 <= x <= SLIDER_X2 + 15 and SLIDER_Y1 - 10 <= y <= SLIDER_Y2 + 10:
            dragging_slider = True
            norm_val = (x - SLIDER_X1) / max(1, (SLIDER_X2 - SLIDER_X1))
            conf_percent = int(max(10, min(95, 10 + norm_val * 85)))
            try:
                cv2.setTrackbarPos("Conf (%)", WINDOW_NAME, conf_percent)
            except Exception:
                pass
    elif event == cv2.EVENT_MOUSEMOVE and dragging_slider:
        norm_val = (x - SLIDER_X1) / max(1, (SLIDER_X2 - SLIDER_X1))
        conf_percent = int(max(10, min(95, 10 + norm_val * 85)))
        try:
            cv2.setTrackbarPos("Conf (%)", WINDOW_NAME, conf_percent)
        except Exception:
            pass
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_slider = False

cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

# Track persistence memory (track_id -> frame_hits)
track_hits = {}
MIN_CONSECUTIVE_FRAMES = 2  # Responsive confirmation for distant fast targets
MAX_MISSED_FRAMES = 15      # Keep locked even if drone momentarily turns/banks
track_last_seen = {}

prev_time = time.time()
frame_count = 0

print("\n" + "=" * 60)
print("  LONG-RANGE DRONE DETECTION SYSTEM ACTIVATED")
print("  - Optimized for distant objects with High-Res Inference (1280px)")
print("  - Sensitivity: Click/drag bottom bar or use [ / ] keys")
print("  - Light/Brightness: Press 'b' / 'B' to adjust light compensation")
print("  - Press 'q' or ESC to exit")
print("=" * 60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from camera.")
        break

    frame_count += 1
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Apply Light / Brightness adjustment if requested
    if brightness_boost != 0:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_boost)

    h, w, _ = frame.shape
    conf_threshold = conf_percent / 100.0

    # Run YOLO with ByteTrack tracker at FULL HIGH-RESOLUTION (imgsz=1280)
    # This prevents distant/small drones from being downscaled and blurred out
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

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # STRICT CLASS FILTER: Only drone class
            if cls_id != DRONE_CLASS_ID or confidence < conf_threshold:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = (x2 - x1), (y2 - y1)

            # Distant / Small Target Detection Logic:
            # Distant objects (small bounding box) get fast track lock to prevent distance drops
            is_confirmed = False
            if track_id is not None:
                track_hits[track_id] = track_hits.get(track_id, 0) + 1
                track_last_seen[track_id] = frame_count
                if track_hits[track_id] >= MIN_CONSECUTIVE_FRAMES or confidence >= 0.50:
                    is_confirmed = True
            else:
                if confidence >= 0.40:
                    is_confirmed = True

            if is_confirmed:
                confirmed_drone_count += 1

                # Target Alert Bounding Box (High-contrast Red with corner reticle)
                box_color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Corner crosshair brackets
                corner_len = max(8, min(24, box_w // 3, box_h // 3))
                # Top-Left
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
                # Top-Right
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
                # Bottom-Left
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 3)
                # Bottom-Right
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)

                # Center reticle dot for long-distance spot identification
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

                # Label Badge
                track_str = f"ID:{track_id} | " if track_id is not None else ""
                dist_str = " (DISTANT)" if max(box_w, box_h) < 65 else ""
                label = f"DRONE {track_str}{confidence * 100:.1f}%{dist_str}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                label_y1 = max(y1 - label_h - 10, 0)
                label_y2 = y1

                cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 10, label_y2), (0, 0, 200), -1)
                cv2.putText(frame, label, (x1 + 5, label_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # Clean up stale tracks
    stale_tracks = [tid for tid, last_f in track_last_seen.items() if frame_count - last_f > MAX_MISSED_FRAMES]
    for tid in stale_tracks:
        track_hits.pop(tid, None)
        track_last_seen.pop(tid, None)

    # --- TOP HUD HEADER ---
    hud_bg_color = (0, 0, 180) if confirmed_drone_count > 0 else (35, 35, 35)
    cv2.rectangle(frame, (0, 0), (w, 42), hud_bg_color, -1)

    status_text = f"ALERT: {confirmed_drone_count} DRONE(S) DETECTED" if confirmed_drone_count > 0 else "SCANNING AIRSPACE: CLEAR (1280px HI-RES)"
    status_color = (0, 255, 255) if confirmed_drone_count > 0 else (0, 255, 0)
    cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f} | Res: {w}x{h} | Light: {'+' if brightness_boost>=0 else ''}{brightness_boost}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(frame, fps_text, (w - tw - 15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # --- BOTTOM HUD FOOTER (Interactive Sensitivity Bar) ---
    footer_h = 44
    cv2.rectangle(frame, (0, h - footer_h), (w, h), (25, 25, 25), -1)
    cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (60, 60, 60), 1)

    cv2.putText(frame, "SENSITIVITY:", (15, h - 16), cv2.FONT_HERSHEY_DUPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    SLIDER_X1 = 145
    SLIDER_X2 = min(w - 280, 520)
    SLIDER_Y1 = h - 28
    SLIDER_Y2 = h - 14

    cv2.rectangle(frame, (SLIDER_X1, SLIDER_Y1), (SLIDER_X2, SLIDER_Y2), (55, 55, 55), -1)

    fill_ratio = (conf_percent - 10) / 85.0
    fill_x = int(SLIDER_X1 + fill_ratio * (SLIDER_X2 - SLIDER_X1))
    fill_color = (0, 165, 255) if conf_percent > 45 else (0, 220, 100)
    cv2.rectangle(frame, (SLIDER_X1, SLIDER_Y1), (fill_x, SLIDER_Y2), fill_color, -1)

    cv2.circle(frame, (fill_x, (SLIDER_Y1 + SLIDER_Y2) // 2), 8, (255, 255, 255), -1)
    cv2.circle(frame, (fill_x, (SLIDER_Y1 + SLIDER_Y2) // 2), 9, (0, 140, 255), 2)

    conf_label = f"{conf_percent}%"
    cv2.putText(frame, conf_label, (SLIDER_X2 + 15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    hint_text = "Adjust: [ / ] or Click Bar | Light: B | Quit: Q"
    (hw, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, hint_text, (w - hw - 15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Display result
    cv2.imshow(WINDOW_NAME, frame)

    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        break
    elif key in (ord("+"), ord("="), ord("]"), 0, 82):
        conf_percent = min(95, conf_percent + 5)
        try:
            cv2.setTrackbarPos("Conf (%)", WINDOW_NAME, conf_percent)
        except Exception:
            pass
    elif key in (ord("-"), ord("_"), ord("["), 1, 84):
        conf_percent = max(10, conf_percent - 5)
        try:
            cv2.setTrackbarPos("Conf (%)", WINDOW_NAME, conf_percent)
        except Exception:
            pass
    elif key in (ord("b"),):
        brightness_boost = (brightness_boost + 15) if brightness_boost < 60 else -30
    elif key in (ord("B"),):
        brightness_boost = max(-50, brightness_boost - 15)

cap.release()
cv2.destroyAllWindows()



