import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE = Path("drone_classifier/raw/UAV-Identification/extracted")
DEST = Path("drone_classifier/dataset")

CLASSES = sorted([d.name for d in SOURCE.iterdir() if d.is_dir()])

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# Create destination directories
for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (DEST / split / class_name).mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    class_dir = SOURCE / class_name

    images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    # First split: 70% train, 30% temporary
    train_files, temp_files = train_test_split(
        images,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Second split: divide temporary 50/50 into val and test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, files in splits.items():
        for src in files:
            dst = DEST / split / class_name / src.name

            # Avoid accidental overwrite
            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                counter = 1

                while dst.exists():
                    dst = (
                        DEST / split / class_name /
                        f"{stem}_{counter}{suffix}"
                    )
                    counter += 1

            shutil.copy2(src, dst)

    print(
        f"{class_name}: "
        f"train={len(train_files)}, "
        f"val={len(val_files)}, "
        f"test={len(test_files)}, "
        f"total={len(images)}"
    )

print("\nDataset split complete.")
