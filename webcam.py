import time
import cv2
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

# 2. Camera Setup (Use DirectShow on Windows for reliable, instant feed)
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    # Try index 1 if 0 is unavailable
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print(f"ERROR: Could not open camera (tried indices 0 and 1)")
    exit(1)

# Set resolution for crisp performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Window & Trackbar configuration for real-time calibration
WINDOW_NAME = "Drone Detection System"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

def nothing(x):
    pass

# Start with a strict 55% default confidence to filter out random motion/objects
DEFAULT_CONF_PERCENT = 55
cv2.createTrackbar("Conf Threshold (%)", WINDOW_NAME, DEFAULT_CONF_PERCENT, 95, nothing)

# Track persistence memory (track_id -> frame_hits)
# Requires an object to persist across multiple frames before confirming as a Drone
track_hits = {}
MIN_CONSECUTIVE_FRAMES = 3
MAX_MISSED_FRAMES = 10
track_last_seen = {}

prev_time = time.time()
frame_count = 0

print("\n" + "=" * 50)
print("  DRONE DETECTION SYSTEM ACTIVATED")
print("  - Adjust sensitivity with the Trackbar or [ / ] keys")
print("  - Press 'q' or ESC to exit")
print("=" * 50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from camera.")
        break

    frame_count += 1
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Get current confidence threshold from trackbar
    conf_trackbar = cv2.getTrackbarPos("Conf Threshold (%)", WINDOW_NAME)
    # Ensure minimum confidence of at least 20%
    conf_threshold = max(conf_trackbar, 20) / 100.0

    # Run YOLO with ByteTrack tracker
    # Low-confidence non-drone motion is filtered out before tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=conf_threshold,
        iou=0.5,
        verbose=False
    )

    active_track_ids_in_frame = set()
    confirmed_drone_count = 0

    h, w, _ = frame.shape

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # STRICT CLASS FILTER: Ignore anything that is not classified as a drone
            if cls_id != DRONE_CLASS_ID:
                continue

            # Confidence check against active threshold
            if confidence < conf_threshold:
                continue

            # Track ID handling
            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Temporal filtering: Require persistent track hits to avoid single-frame motion glitches
            is_confirmed = False
            if track_id is not None:
                active_track_ids_in_frame.add(track_id)
                track_hits[track_id] = track_hits.get(track_id, 0) + 1
                track_last_seen[track_id] = frame_count

                if track_hits[track_id] >= MIN_CONSECUTIVE_FRAMES or confidence >= 0.70:
                    is_confirmed = True
            else:
                # If no tracker ID, require very high confidence to display
                if confidence >= 0.65:
                    is_confirmed = True

            if is_confirmed:
                confirmed_drone_count += 1

                # Target Alert Bounding Box (Bright Red with corner accents)
                box_color = (0, 0, 255)  # Red BGR
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Corner brackets for modern HUD look
                corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                if corner_len > 0:
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

                # Label Badge
                track_str = f"ID:{track_id} | " if track_id is not None else ""
                label = f"DRONE {track_str}{confidence * 100:.1f}%"
                
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                
                label_y1 = max(y1 - label_h - 10, 0)
                label_y2 = y1
                
                # Background badge
                cv2.rectangle(
                    frame,
                    (x1, label_y1),
                    (x1 + label_w + 10, label_y2),
                    (0, 0, 200),
                    -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, label_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

    # Clean up stale tracks from memory
    stale_tracks = [
        tid for tid, last_frame in track_last_seen.items()
        if frame_count - last_frame > MAX_MISSED_FRAMES
    ]
    for tid in stale_tracks:
        track_hits.pop(tid, None)
        track_last_seen.pop(tid, None)

    # HUD Status Bar (Top)
    hud_bg_color = (0, 0, 180) if confirmed_drone_count > 0 else (40, 40, 40)
    cv2.rectangle(frame, (0, 0), (w, 45), hud_bg_color, -1)

    if confirmed_drone_count > 0:
        status_text = f"ALERT: {confirmed_drone_count} DRONE(S) DETECTED"
        status_color = (0, 255, 255)
    else:
        status_text = "SCANNING: AIRSPACE CLEAR"
        status_color = (0, 255, 0)

    cv2.putText(
        frame,
        status_text,
        (15, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        status_color,
        2,
        cv2.LINE_AA
    )

    # System telemetry info on right side of top bar
    telemetry_text = f"Conf: {int(conf_threshold * 100)}% | FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        frame,
        telemetry_text,
        (w - tw - 15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Display result
    cv2.imshow(WINDOW_NAME, frame)

    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC to quit
        break
    elif key in (ord("+"), ord("="), ord("]")):
        new_val = min(95, cv2.getTrackbarPos("Conf Threshold (%)", WINDOW_NAME) + 5)
        cv2.setTrackbarPos("Conf Threshold (%)", WINDOW_NAME, new_val)
    elif key in (ord("-"), ord("_"), ord("[")):
        new_val = max(20, cv2.getTrackbarPos("Conf Threshold (%)", WINDOW_NAME) - 5)
        cv2.setTrackbarPos("Conf Threshold (%)", WINDOW_NAME, new_val)

cap.release()
cv2.destroyAllWindows()

