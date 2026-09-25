"""8-class drone identification using the trained custom CNN."""

from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


UNKNOWN = "UNKNOWN"


def build_mobilenet(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


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

    def __init__(self, model_path=None, confidence_threshold=0.25):

        self.confidence_threshold = confidence_threshold
        self.model = None
        self.yolo_model = None
        self.enabled = False
        self.classes = []

        # Auto-detect candidates
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend([
            Path("drone_classifier/model/drone_cnn_best.pth"),
            Path("models/drone_type_classifier.pt"),
            Path("models/best_classifier.pt"),
        ])

        resolved_path = None
        for cand in candidates:
            if cand.is_file():
                resolved_path = cand
                break

        if resolved_path is None:
            print(f"[!] Classifier checkpoint not found (searched: {[str(c) for c in candidates]})")
            return

        model_path = resolved_path


        try:
            checkpoint = torch.load(
                model_path,
                map_location="cpu"
            )

            self.classes = checkpoint["classes"]
            state_dict = checkpoint["model_state"]

            if checkpoint.get("arch") == "mobilenet_v3_small" or "features.0.0.weight" in state_dict:
                self.model = build_mobilenet(len(self.classes))
            else:
                self.model = DroneCNN(len(self.classes))

            self.model.load_state_dict(state_dict)
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
                f"[*] Drone Model Classifier loaded "
                f"({len(self.classes)} classes: {', '.join(self.classes)})"
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
# Used by the auto-profile distance estimator.
DRONE_WIDTHS = {
    "DJI-Neo": 0.16,
    "DJI-Mini": 0.24,
    "DJI-Avata": 0.18,
    "DJI-Mavic": 0.35,
    "DJI-Phantom": 0.35,
    "Parrot_Bebop": 0.38,
    "Predator-Reaper": 20.0,
    "RQ11-Raven": 1.37,
    "RQ4-GlobalHawk": 39.90,
    "RQ7-Shadow": 4.57,
    "Yuneec-Typhoon": 0.54,
}

AIRFRAME_DOMAINS = {
    # Civilian
    "DJI-Neo": "CIVILIAN",
    "DJI-Mini": "CIVILIAN",
    "DJI-Avata": "CIVILIAN",
    "DJI-Mavic": "CIVILIAN",
    "DJI-Phantom": "CIVILIAN",
    "Parrot_Bebop": "CIVILIAN",
    "Yuneec-Typhoon": "CIVILIAN",
    "civilian": "CIVILIAN",
    "Civilian": "CIVILIAN",
    
    # Military
    "RQ11-Raven": "MILITARY",
    "RQ7-Shadow": "MILITARY",
    "Predator-Reaper": "MILITARY",
    "MQ9-Reaper": "MILITARY",
    "RQ4-GlobalHawk": "MILITARY",
    "Bayraktar-TB2": "MILITARY",
    "Shahed-136": "MILITARY",
    "military": "MILITARY",
    "Military": "MILITARY",
}

def get_drone_width(drone_type):
    return DRONE_WIDTHS.get(drone_type)

def get_drone_domain(drone_type):
    if not drone_type or str(drone_type).upper() == "UNKNOWN":
        return "UNKNOWN"
    norm = str(drone_type).strip().lower().replace("_", "-")
    if "civilian" in norm:
        return "CIVILIAN"
    if "military" in norm:
        return "MILITARY"
    for k, v in AIRFRAME_DOMAINS.items():
        if k.lower().replace("_", "-") == norm:
            return v
    if any(civ in norm for civ in ["dji", "neo", "mavic", "phantom", "mini", "avata", "parrot", "bebop", "yuneec", "typhoon", "autel", "skydio"]):
        return "CIVILIAN"
    if any(mil in norm for mil in ["raven", "shadow", "reaper", "predator", "globalhawk", "bayraktar", "shahed", "orlan", "switchblade", "wing"]):
        return "MILITARY"
    return "UNKNOWN"

def get_drone_dimensions(drone_type, aspect_ratio=2.0):
    """
    Returns auto-calibrated physical width and height for distance estimation.
    If the type is unknown, estimates based on morphological aspect ratio.
    """
    if drone_type in DRONE_WIDTHS:
        w = DRONE_WIDTHS[drone_type]
        domain = AIRFRAME_DOMAINS.get(drone_type, "UNKNOWN")
        # Fixed wing models have different height ratio compared to multirotors
        if domain == "MILITARY" or w > 1.0:
            h = w * 0.25
        else:
            h = w * 0.37
        return {"name": drone_type, "width": w, "height": h, "domain": domain, "auto": True}
    
    # Morphological fallback when type is UNKNOWN
    if aspect_ratio >= 2.8:
        return {"name": "Tactical Wing (Auto)", "width": 1.37, "height": 0.35, "domain": "UNKNOWN", "auto": True}
    elif aspect_ratio >= 1.6:
        return {"name": "Standard Quad (Auto)", "width": 0.38, "height": 0.14, "domain": "UNKNOWN", "auto": True}
    else:
        return {"name": "Micro/Palm (Auto)", "width": 0.16, "height": 0.05, "domain": "UNKNOWN", "auto": True}

