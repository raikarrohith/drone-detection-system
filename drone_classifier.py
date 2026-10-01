"""Drone identification and classification supporting both YOLO (.pt) and custom CNN (.pth) models."""

from pathlib import Path
import json
import os
import math

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


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
            try:
                checkpoint = torch.load(model_path, map_location="cpu")
            except Exception:
                # PyTorch 2.6+ weights_only compatibility fallback
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

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
            print(f"[!] Failed to load drone classifier from {model_path}: {exc}")

    def classify(self, image, device="cpu"):

        if not self.enabled or image is None or image.size == 0:
            return UNKNOWN, 0.0

        # Skip tiny slivers/micro crops to save CPU and avoid inaccurate classification
        ih, iw = image.shape[:2]
        if ih < 16 or iw < 16:
            return UNKNOWN, 0.0

        try:
            # 1. Ultralytics YOLO Classifier Inference (Fast imgsz=160, zero verbose overhead)
            if self.yolo_model is not None:
                results = self.yolo_model.predict(
                    image,
                    imgsz=160,
                    device=device,
                    verbose=False
                )
                if not results or len(results) == 0:
                    return UNKNOWN, 0.0

                probs = results[0].probs
                if probs is None:
                    return UNKNOWN, 0.0

                top1_idx = int(probs.top1)
                conf = float(probs.top1conf.cpu().item()) if hasattr(probs.top1conf, "cpu") else float(probs.top1conf)
                label = self.yolo_model.names.get(top1_idx, UNKNOWN)

                if conf < self.confidence_threshold or label.lower() == "unknown":
                    return UNKNOWN, conf

                return label, conf

            # 2. PyTorch Custom CNN Inference
            if self.model is not None:
                image_rgb = image[:, :, ::-1]  # OpenCV BGR -> RGB
                pil_image = Image.fromarray(image_rgb)
                tensor = self.transform(pil_image).unsqueeze(0)

                with torch.no_grad():
                    output = self.model(tensor)
                    probabilities = torch.softmax(output, dim=1)
                    class_id = int(torch.argmax(probabilities, dim=1)[0])
                    confidence = float(probabilities[0, class_id])

                label = self.classes[class_id]
                if confidence < self.confidence_threshold or label.lower() == "unknown":
                    return UNKNOWN, confidence

                return label, confidence

            return UNKNOWN, 0.0

        except Exception as exc:
            return UNKNOWN, 0.0


# Approximate physical widths in metres.
# Used by the auto-profile distance estimator.
DRONE_WIDTHS = {
    "DJI-Neo": 0.16,
    "DJI-Mini": 0.24,
    "DJI-Avata": 0.18,
    "DJI-Mavic": 0.35,
    "DJI-Mavic-Air": 0.25,
    "DJI-Mini": 0.24,
    "DJI-Phantom": 0.35,
    "DJI-Inspire": 0.58,
    "DJI-Matrice": 0.88,
    "DJI-FPV": 0.25,
    "DJI-Avata": 0.18,
    "Parrot_Bebop": 0.38,
    "Parrot-Anafi": 0.24,
    "Yuneec-Typhoon": 0.54,
    "Yuneec-H520": 0.52,
    "Autel-EVO": 0.36,
    "Skydio-2": 0.31,
    "Skydio-X2": 0.66,
    "Generic-Quadcopter": 0.38,
    "Generic-Hexacopter": 0.65,
    "FPV-Racing-Drone": 0.22,
    "Quad-UAV": 0.38,
    "Quadcopter-UAV": 0.38,
    "Micro-Mini": 0.24,

    # Military Airframes
    "Predator-Reaper": 20.0,
    "MQ9-Reaper": 20.0,
    "MQ1-Predator": 14.8,
    "RQ11-Raven": 1.37,
    "RQ4-GlobalHawk": 39.90,
    "RQ7-Shadow": 4.57,
    "Bayraktar-TB2": 12.0,
    "Shahed-136": 2.50,
    "Orlan-10": 3.10,
    "ScanEagle": 3.11,
    "Switchblade-300": 0.60,
    "Switchblade-600": 1.30,
    "IAI-Heron": 16.6,
    "Hermes-450": 10.5,
    "Hermes-900": 15.0,
    "Lancet-3": 1.65,
    "Tactical-Wing": 1.37,
}

AIRFRAME_DOMAINS = {
    # Civilian
    "DJI-Neo": "CIVILIAN",
    "DJI-Mini": "CIVILIAN",
    "DJI-Avata": "CIVILIAN",
    "DJI-Mavic": "CIVILIAN",
    "DJI-Mavic-Air": "CIVILIAN",
    "DJI-Mini": "CIVILIAN",
    "DJI-Phantom": "CIVILIAN",
    "DJI-Inspire": "CIVILIAN",
    "DJI-Matrice": "CIVILIAN",
    "DJI-FPV": "CIVILIAN",
    "DJI-Avata": "CIVILIAN",
    "Parrot_Bebop": "CIVILIAN",
    "Parrot-Anafi": "CIVILIAN",
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


# Verified physical dimensions (Width, Height, Diagonal in metres)
# Used by the CRLB Fisher-Information Monocular Distance Fusion Engine
DRONE_PHYSICAL_SPECS = {
    # Civilian Multirotors
    "DJI-Mavic": {"width": 0.354, "height": 0.098, "domain": "CIVILIAN"},
    "DJI-Mavic-Air": {"width": 0.252, "height": 0.084, "domain": "CIVILIAN"},
    "DJI-Mini": {"width": 0.245, "height": 0.056, "domain": "CIVILIAN"},
    "DJI-Phantom": {"width": 0.350, "height": 0.190, "domain": "CIVILIAN"},
    "DJI-Inspire": {"width": 0.580, "height": 0.300, "domain": "CIVILIAN"},
    "DJI-Matrice": {"width": 0.880, "height": 0.420, "domain": "CIVILIAN"},
    "DJI-FPV": {"width": 0.255, "height": 0.127, "domain": "CIVILIAN"},
    "DJI-Avata": {"width": 0.180, "height": 0.080, "domain": "CIVILIAN"},
    "Parrot_Bebop": {"width": 0.382, "height": 0.089, "domain": "CIVILIAN"},
    "Parrot-Anafi": {"width": 0.240, "height": 0.065, "domain": "CIVILIAN"},
    "Yuneec-Typhoon": {"width": 0.520, "height": 0.310, "domain": "CIVILIAN"},
    "Yuneec-H520": {"width": 0.520, "height": 0.310, "domain": "CIVILIAN"},
    "Autel-EVO": {"width": 0.360, "height": 0.110, "domain": "CIVILIAN"},
    "Skydio-2": {"width": 0.310, "height": 0.065, "domain": "CIVILIAN"},
    "Skydio-X2": {"width": 0.660, "height": 0.210, "domain": "CIVILIAN"},
    "Generic-Quadcopter": {"width": 0.380, "height": 0.140, "domain": "CIVILIAN"},
    "Generic-Hexacopter": {"width": 0.650, "height": 0.280, "domain": "CIVILIAN"},
    "FPV-Racing-Drone": {"width": 0.220, "height": 0.075, "domain": "CIVILIAN"},
    "Quadcopter-UAV": {"width": 0.380, "height": 0.140, "domain": "CIVILIAN"},
    "Micro-Mini": {"width": 0.245, "height": 0.080, "domain": "CIVILIAN"},

    # Military Tactical / Combat UAVs
    "Bayraktar-TB2": {"width": 12.00, "height": 2.20, "domain": "MILITARY"},
    "Shahed-136": {"width": 2.50, "height": 0.50, "domain": "MILITARY"},
    "Orlan-10": {"width": 3.10, "height": 0.65, "domain": "MILITARY"},
    "RQ11-Raven": {"width": 1.37, "height": 0.35, "domain": "MILITARY"},
    "RQ7-Shadow": {"width": 4.57, "height": 1.00, "domain": "MILITARY"},
    "RQ4-GlobalHawk": {"width": 39.90, "height": 4.70, "domain": "MILITARY"},
    "Predator-Reaper": {"width": 20.00, "height": 3.80, "domain": "MILITARY"},
    "MQ9-Reaper": {"width": 20.00, "height": 3.80, "domain": "MILITARY"},
    "MQ1-Predator": {"width": 14.80, "height": 2.10, "domain": "MILITARY"},
    "ScanEagle": {"width": 3.11, "height": 0.50, "domain": "MILITARY"},
    "Switchblade-300": {"width": 0.60, "height": 0.15, "domain": "MILITARY"},
    "Switchblade-600": {"width": 1.30, "height": 0.30, "domain": "MILITARY"},
    "IAI-Heron": {"width": 16.60, "height": 3.20, "domain": "MILITARY"},
    "Hermes-450": {"width": 10.50, "height": 2.30, "domain": "MILITARY"},
    "Hermes-900": {"width": 15.00, "height": 3.00, "domain": "MILITARY"},
    "Lancet-3": {"width": 1.65, "height": 0.40, "domain": "MILITARY"},
    "Tactical-Wing": {"width": 1.37, "height": 0.35, "domain": "MILITARY"},
}


def get_drone_width(drone_type):
    if not drone_type:
        return 0.38
    if drone_type in DRONE_PHYSICAL_SPECS:
        return DRONE_PHYSICAL_SPECS[drone_type]["width"]
    if drone_type in DRONE_WIDTHS:
        return DRONE_WIDTHS[drone_type]
    for k, v in DRONE_PHYSICAL_SPECS.items():
        if k.lower() == str(drone_type).lower():
            return v["width"]
    return 0.38


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
    Returns auto-calibrated exact physical width, height, and diagonal for CRLB distance estimation.
    If the type is unknown, estimates based on morphological aspect ratio and domain.
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

