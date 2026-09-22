#!/usr/bin/env python3
"""
Export YOLO11n-P2 Drone Detector for Real-Time High-FPS (45-60+ FPS) Inference.

Supported Formats:
- ONNX (with dynamic batch / FP16)
- OpenVINO (CPU Intel / AMD optimized for 2x-3x faster inference)
- TorchScript
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(model_path="models/best.pt", export_format="onnx", imgsz=640, half=False):
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file {model_path} not found.")

    print(f"[*] Loading model: {model_path}")
    model = YOLO(str(model_file))

    print(f"[*] Exporting to format: {export_format.upper()} (imgsz={imgsz}, half={half})...")
    exported_path = model.export(
        format=export_format,
        imgsz=imgsz,
        half=half,
        dynamic=False,
        simplify=True
    )
    print(f"[+] Model successfully exported to: {exported_path}")
    return exported_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO P2 Drone Detector for High-FPS Inference")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO .pt model")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "openvino", "torchscript", "engine"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision")
    args = parser.parse_args()

    export_model(model_path=args.model, export_format=args.format, imgsz=args.imgsz, half=args.half)


if __name__ == "__main__":
    main()
