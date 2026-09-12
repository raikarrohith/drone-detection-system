import cv2
from ultralytics import YOLO

# Load our fine-tuned drone detector
model = YOLO("models/best.pt")

# Open external camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    # Detect + track drones
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.30,
        verbose=False
    )

    # Draw ONLY drone detections
    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # RED bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            label = f"DRONE {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    cv2.imshow("Drone Detection - Live Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
