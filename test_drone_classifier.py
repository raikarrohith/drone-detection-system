import sys
import argparse
import cv2
from ultralytics import YOLO
from drone_classifier import DroneTypeClassifier, get_drone_domain

parser = argparse.ArgumentParser(description="Test Drone Detection + Classification")
parser.add_argument("--video", "-v", type=str, default="sampledrone.mp4", help="Video file or camera index")
parser.add_argument("--model", "-m", type=str, default="models/drone_type_classifier.pt", help="Classifier model path (.pt or .pth)")
args = parser.parse_args()

# -----------------------------
# YOLO detector & Classifier
# -----------------------------
detector = YOLO("models/best.pt")
classifier = DroneTypeClassifier(args.model, confidence_threshold=0.30)

# -----------------------------
# Video Source
# -----------------------------
video_src = args.video
if video_src.isdigit():
    video_src = int(video_src)

cap = cv2.VideoCapture(video_src)
if not cap.isOpened():
    print(f"❌ Could not open video source: {args.video}")
    sys.exit(1)

print("✅ Starting detection + classification")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector(frame, conf=0.25, verbose=False)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            drone_type, class_conf = classifier.classify(crop)
            domain = get_drone_domain(drone_type)

            # Color coding: Red for Military, Green for Civilian, Amber for Unknown
            if domain == "MILITARY":
                box_color = (0, 0, 255)
            elif domain == "CIVILIAN":
                box_color = (0, 255, 120)
            else:
                box_color = (0, 215, 255)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Display information
            label = f"[{domain}] {drone_type} {class_conf*100:.1f}%"
            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            cv2.putText(
                frame,
                f"Drone: {confidence*100:.1f}%",
                (x1, min(frame.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

    cv2.imshow("Drone Detection + Classification", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Finished.")

