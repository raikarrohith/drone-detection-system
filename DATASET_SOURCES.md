# Civilian vs. Military Drone Datasets Guide

This document lists recommended open datasets and instructions for expanding the civilian/military classifier dataset.

---

## 1. Recommended Open Datasets

### A. UAV-Identification Dataset (Civilian & Military)
Contains annotated images across multiple airframes:
* **Civilian Models:** DJI Mavic, DJI Phantom, Parrot Bebop, Yuneec Typhoon.
* **Military Models:** RQ-7 Shadow, RQ-4 Global Hawk, RQ-11 Raven, MQ-9 Reaper / Predator.
* **Download:** [UAV-Identification Repository / Zenodo](https://github.com/cui-research/UAV-Identification)

### B. Anti-UAV & Drone-vs-Bird Datasets (Civilian Multi-rotors & Distractors)
* **Anti-UAV Benchmark:** Large-scale dataset of civilian quadcopters and fixed-wing targets in visible/infrared.
* **Drone-vs-Bird Challenge:** High-resolution outdoor imagery distinguishing small hobbyist drones from birds and clouds.
* **Kaggle Drone Detection Dataset:** Over 1,300+ labeled images of consumer drones (DJI, Autel, Parrot).

### C. Military Fixed-Wing & Tactical UAV Datasets (Kaggle & Roboflow)
* **Roboflow Universe:** Search for `military-uav`, `combat-drones`, or `tactical-uav` to find pre-split datasets for models like Bayraktar TB2, Shahed-136, Orlan-10, Switchblade, and Heron.
* **Kaggle Military Aircraft / UAV Datasets:** Contains annotated aerial images of tactical unmanned systems.

---

## 2. Ingesting New Datasets

### Method 1: Using Airframe Folders (`prepare_uav_identification.py`)
If your downloaded dataset has airframe folders (e.g., `DJI-Mini/`, `Bayraktar-TB2/`):
```bash
python prepare_uav_identification.py --source datasets/my_downloaded_data/ --output dataset_raw/drone_type_classifier
```

### Method 2: Ingesting Any Raw Folder (`ingest_custom_dataset.py`)
If you downloaded an arbitrary folder of images for a specific class:

```bash
# Ingest civilian drone images
python ingest_custom_dataset.py --source path/to/civilian_images --type civilian

# Ingest military drone images
python ingest_custom_dataset.py --source path/to/military_images --type military

# Ingest confusable distractors (birds, planes, kites, clouds)
python ingest_custom_dataset.py --source path/to/distractor_images --type unknown
```

---

## 3. Retraining the Classifier

Once new images are ingested into `dataset_raw/drone_type_classifier/`:

```bash
yolo classify train data=dataset_raw/drone_type_classifier model=yolo11n-cls.pt epochs=50 imgsz=224
```

Copy the generated `runs/classify/train/weights/best.pt` to `models/drone_type_classifier.pt`.
