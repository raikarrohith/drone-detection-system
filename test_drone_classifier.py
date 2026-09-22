import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# YOLO detector
# -----------------------------
detector = YOLO("models/best.pt")

# -----------------------------
# Drone classifier
# -----------------------------
class DroneCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))


checkpoint = torch.load(
    "drone_classifier/model/drone_cnn_best.pth",
    map_location="cpu"
)

classifier = DroneCNN(8)
classifier.load_state_dict(checkpoint["model_state"])
classifier.eval()

classes = checkpoint["classes"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -----------------------------
# Video
# -----------------------------
cap = cv2.VideoCapture("sampledrone.mp4")

if not cap.isOpened():
    print("❌ Could not open sampledrone.mp4")
    exit()

print("✅ Starting detection + classification")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = detector(frame, conf=0.25, verbose=False)

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Our detector has only one class: drone
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # OpenCV BGR -> RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(crop_rgb)

            tensor = transform(pil_image).unsqueeze(0)

            # Classification
            with torch.no_grad():
                output = classifier(tensor)
                probabilities = torch.softmax(output, dim=1)

                class_id = int(torch.argmax(probabilities, dim=1)[0])
                class_conf = float(probabilities[0, class_id])

            drone_type = classes[class_id]

            # Draw detection box
            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,0,255),
                2
            )

            # Display information
            label = f"{drone_type} {class_conf*100:.1f}%"

            cv2.putText(
                frame,
                label,
                (x1, max(30,y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

            cv2.putText(
                frame,
                f"Drone: {confidence*100:.1f}%",
                (x1, min(frame.shape[0]-10,y2+25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )

    cv2.imshow("Drone Detection + Classification", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Finished.")
