# Drone-type classifier

`models/best.pt` detects a drone but was trained with only the `drone` class.
It cannot determine whether an aircraft is civilian or military. The app now
accepts a separate Ultralytics **classification** model that receives each
detected drone crop.

Train with clearly licensed, representative images laid out as follows:

```text
dataset_raw/drone_type_classifier/
  train/civilian/  train/military/  train/unknown/
  val/civilian/    val/military/    val/unknown/
```

Then train from the classifier dataset directory and copy the resulting `best.pt`
to `models/drone_type_classifier.pt`:

```bash
yolo classify train data=dataset_raw/drone_type_classifier model=yolo11n-cls.pt epochs=50 imgsz=224
```

Run the tracker with it:

```bash
python webcam.py --type-model models/drone_type_classifier.pt --type-confidence 0.70
```

Without a model, with unrecognized class labels, or below the threshold, the
HUD says `TYPE: UNKNOWN`. This is intentional: visual appearance alone is not
a reliable basis for asserting military use.

Populate `unknown` with visually confusable targets (birds, helicopters,
fixed-wing aircraft, kites, occluded/too-small drones) and drones whose use
cannot be verified. Do not use military-looking scenery or image source alone
as a label; label by the documented aircraft model and provenance.
