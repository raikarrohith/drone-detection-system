"""Drone identification and classification supporting both YOLO (.pt) and custom CNN (.pth) models."""

from pathlib import Path
import json
import os
import math

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


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

    def __init__(self, model_path=None, confidence_threshold=0.20):

        self.confidence_threshold = confidence_threshold
        self.model = None
        self.yolo_model = None
        self.enabled = False
        self.classes = []

        # Auto-detect model if not provided or default doesn't exist
        resolved_path = None
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend([
            Path("models/drone_type_classifier.pt"),
            Path("models/best_classifier.pt"),
            Path("drone_classifier/model/drone_cnn_best.pth"),
            Path("drone_type_classifier.pt")
        ])

        for cand in candidates:
            if cand.is_file():
                resolved_path = cand
                break

        if resolved_path is None:
            print(f"[!] Drone classifier not found (searched: {[str(c) for c in candidates]})")
            return

        model_path = resolved_path

        # 1. Try loading as Ultralytics YOLO classification model (.pt)
        if model_path.suffix.lower() == ".pt" and YOLO is not None:
            try:
                self.yolo_model = YOLO(str(model_path))
                if hasattr(self.yolo_model, "names") and self.yolo_model.names:
                    self.classes = list(self.yolo_model.names.values())
                self.enabled = True
                print(f"[*] Ultralytics Drone Classifier loaded from {model_path} ({len(self.classes)} classes: {self.classes})")
                return
            except Exception as exc:
                print(f"[!] Ultralytics load attempt failed on {model_path}: {exc}")

        # 2. Try loading as PyTorch CNN checkpoint (.pth or .pt state_dict)
        try:
            try:
                checkpoint = torch.load(model_path, map_location="cpu")
            except Exception:
                # PyTorch 2.6+ weights_only compatibility fallback
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict) and "classes" in checkpoint and "model_state" in checkpoint:
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
                print(f"[*] Custom CNN Drone Classifier loaded ({len(self.classes)} classes: {self.classes})")
            elif YOLO is not None:
                self.yolo_model = YOLO(str(model_path))
                self.classes = list(self.yolo_model.names.values()) if hasattr(self.yolo_model, "names") else []
                self.enabled = True
                print(f"[*] YOLO Classifier loaded from {model_path} ({len(self.classes)} classes)")
            else:
                print(f"[!] Unrecognized model format in {model_path}")

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
    # Direct Category Labels
    "civilian": 0.38,
    "military": 2.20,
    "unknown": 0.38,
    "Civilian": 0.38,
    "Military": 2.20,
    
    # Civilian Airframes
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
    # Direct Category Labels
    "civilian": "CIVILIAN",
    "military": "MILITARY",
    "unknown": "UNKNOWN",
    "Civilian": "CIVILIAN",
    "Military": "MILITARY",
    
    # Civilian Airframes
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
    "Yuneec-H520": "CIVILIAN",
    "Autel-EVO": "CIVILIAN",
    "Skydio-2": "CIVILIAN",
    "Skydio-X2": "CIVILIAN",
    "Generic-Quadcopter": "CIVILIAN",
    "Generic-Hexacopter": "CIVILIAN",
    "FPV-Racing-Drone": "CIVILIAN",
    "Quad-UAV": "CIVILIAN",
    "Quadcopter-UAV": "CIVILIAN",
    "Micro-Mini": "CIVILIAN",

    # Military Airframes
    "RQ11-Raven": "MILITARY",
    "RQ7-Shadow": "MILITARY",
    "Predator-Reaper": "MILITARY",
    "MQ9-Reaper": "MILITARY",
    "MQ1-Predator": "MILITARY",
    "RQ4-GlobalHawk": "MILITARY",
    "Bayraktar-TB2": "MILITARY",
    "Shahed-136": "MILITARY",
    "Orlan-10": "MILITARY",
    "ScanEagle": "MILITARY",
    "Switchblade-300": "MILITARY",
    "Switchblade-600": "MILITARY",
    "IAI-Heron": "MILITARY",
    "Hermes-450": "MILITARY",
    "Hermes-900": "MILITARY",
    "Lancet-3": "MILITARY",
    "Tactical-Wing": "MILITARY",
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
    if drone_type in DRONE_PHYSICAL_SPECS:
        return DRONE_PHYSICAL_SPECS[drone_type]["domain"]
    if drone_type in AIRFRAME_DOMAINS:
        return AIRFRAME_DOMAINS[drone_type]
    for k, v in AIRFRAME_DOMAINS.items():
        if k.lower() == str(drone_type).lower():
            return v
    return "CIVILIAN"


def get_drone_dimensions(drone_type, aspect_ratio=2.0):
    """
    Returns auto-calibrated exact physical width, height, and diagonal for CRLB distance estimation.
    If the type is unknown, estimates based on morphological aspect ratio and domain.
    """
    domain = get_drone_domain(drone_type)

    # 1. Exact match from verified specifications database
    if drone_type and str(drone_type).lower() not in ["unknown", "civilian", "military"]:
        for model_name, specs in DRONE_PHYSICAL_SPECS.items():
            if model_name.lower() == str(drone_type).lower():
                w = specs["width"]
                h = specs["height"]
                d = math.sqrt(w**2 + h**2)
                return {"name": model_name, "width": w, "height": h, "diag": d, "domain": specs["domain"], "auto": True}

    # 2. Morphological airframe resolution when generic civilian/military or unknown
    if domain == "MILITARY" or aspect_ratio >= 2.6:
        model_name = "Tactical-Wing" if str(drone_type).lower() in ["military", "unknown"] else str(drone_type)
        w = 1.37
        h = 0.35
        d = math.sqrt(w**2 + h**2)
        return {"name": model_name, "width": w, "height": h, "diag": d, "domain": "MILITARY", "auto": True}
    elif aspect_ratio >= 1.4:
        model_name = "Quadcopter-UAV" if str(drone_type).lower() in ["civilian", "unknown"] else str(drone_type)
        w = 0.38
        h = 0.14
        d = math.sqrt(w**2 + h**2)
        return {"name": model_name, "width": w, "height": h, "diag": d, "domain": "CIVILIAN", "auto": True}
    else:
        model_name = "Micro-Mini" if str(drone_type).lower() in ["civilian", "unknown"] else str(drone_type)
        w = 0.245
        h = 0.08
        d = math.sqrt(w**2 + h**2)
        return {"name": model_name, "width": w, "height": h, "diag": d, "domain": "CIVILIAN", "auto": True}


