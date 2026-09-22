"""8-class drone identification using the trained custom CNN."""

from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


UNKNOWN = "UNKNOWN"


class DroneCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DroneTypeClassifier:

    def __init__(self, model_path=None, confidence_threshold=0.45):

        self.confidence_threshold = confidence_threshold
        self.model = None
        self.enabled = False
        self.classes = []

        if model_path is None:
            model_path = "drone_classifier/model/drone_cnn_best.pth"

        model_path = Path(model_path)

        if not model_path.is_file():
            print(f"[!] CNN classifier not found: {model_path}")
            return

        try:
            checkpoint = torch.load(
                model_path,
                map_location="cpu"
            )

            self.classes = checkpoint["classes"]

            self.model = DroneCNN(len(self.classes))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            image_size = checkpoint.get("image_size", 224)

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

            self.enabled = True

            print(
                f"[*] 8-Class Drone Classifier loaded "
                f"({len(self.classes)} classes)"
            )

        except Exception as exc:
            print(f"[!] Failed to load drone classifier: {exc}")

    def classify(self, image, device="cpu"):

        if not self.enabled or image is None or image.size == 0:
            return UNKNOWN, 0.0

        try:
            # OpenCV BGR -> RGB
            image_rgb = image[:, :, ::-1]

            pil_image = Image.fromarray(image_rgb)

            tensor = self.transform(pil_image).unsqueeze(0)

            with torch.no_grad():
                output = self.model(tensor)

                probabilities = torch.softmax(output, dim=1)

                class_id = int(
                    torch.argmax(probabilities, dim=1)[0]
                )

                confidence = float(
                    probabilities[0, class_id]
                )

            label = self.classes[class_id]

            if confidence < self.confidence_threshold:
                return UNKNOWN, confidence

            return label, confidence

        except Exception as exc:
            print(f"[!] Drone classifier error: {exc}")
            return UNKNOWN, 0.0

# Approximate physical widths in metres.
# Used by the temporary distance estimator until camera calibration.
DRONE_WIDTHS = {
    "DJI-Mavic": 0.35,
    "DJI-Phantom": 0.35,
    "Parrot_Bebop": 0.38,
    "Predator-Reaper": 20.0,
    "RQ11-Raven": 1.37,
    "RQ4-GlobalHawk": 39.90,
    "RQ7-Shadow": 4.57,
    "Yuneec-Typhoon": 0.54,
}

def get_drone_width(drone_type):
    return DRONE_WIDTHS.get(drone_type)
