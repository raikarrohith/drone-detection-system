"""Optional civilian/military drone image classifier.

The detector answers "is this a drone?".  This module deliberately keeps the
type decision separate: a detector trained with only a `drone` label has no
evidence to distinguish civilian and military airframes.
"""

from pathlib import Path

from ultralytics import YOLO


UNKNOWN = "UNKNOWN"
VALID_TYPES = {"civilian": "CIVILIAN", "military": "MILITARY"}


class DroneTypeClassifier:
    """Classify cropped drone detections with an Ultralytics classification model.

    A missing model, unsupported model labels, or an uncertain prediction all
    return UNKNOWN. This avoids turning a visual guess into an alert.
    """

    def __init__(self, model_path=None, confidence_threshold=0.70):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.enabled = False

        if model_path and Path(model_path).is_file():
            self.model = YOLO(model_path)
            self.enabled = True

    @staticmethod
    def _label_from_name(name):
        normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        # Permit practical dataset labels such as civilian_drone / military_uav.
        if "civilian" in normalized or "commercial" in normalized:
            return VALID_TYPES["civilian"]
        if "military" in normalized or "defense" in normalized:
            return VALID_TYPES["military"]
        return UNKNOWN

    def classify(self, image, device="cpu"):
        """Return (label, confidence). `confidence` is 0 for UNKNOWN."""
        if not self.enabled or image is None or image.size == 0:
            return UNKNOWN, 0.0

        try:
            result = self.model(image, device=device, verbose=False)[0]
            if result.probs is None:
                return UNKNOWN, 0.0
            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            label = self._label_from_name(result.names[class_id])
            if label == UNKNOWN or confidence < self.confidence_threshold:
                return UNKNOWN, confidence
            return label, confidence
        except Exception as exc:
            # A bad optional classifier must not stop drone detection/tracking.
            print(f"[!] Drone type classifier unavailable: {exc}")
            self.enabled = False
            return UNKNOWN, 0.0
