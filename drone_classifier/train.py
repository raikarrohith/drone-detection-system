import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("drone_classifier/dataset")
MODEL_DIR = Path("drone_classifier/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device Selection (CUDA -> MPS -> CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# DATA AUGMENTATION & PIPELINE
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS & LOADERS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"\nDiscovered {num_classes} Classes:")
for i, name in enumerate(class_names):
    print(f"  [{i}] {name}")

print(f"\nTraining images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_dataset.targets,
    minlength=num_classes
)

class_weights = len(train_dataset) / (
    num_classes * np.maximum(class_counts, 1)
)

class_weights_t = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nBalanced Class weights:")
for name, weight in zip(class_names, class_weights_t):
    print(f"  {name}: {weight.item():.3f}")


# ============================================================
# PRETRAINED MOBILENET_V3 CLASSIFIER
# ============================================================

def build_model(num_classes):
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    
    # Fine-tune classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

model = build_model(num_classes).to(DEVICE)

print(f"\nModel: Pretrained MobileNetV3-Small ({num_classes} classes)")


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss(weight=class_weights_t)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# ============================================================
# TRAINING LOOP
# ============================================================

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0.0
best_model_state = copy.deepcopy(model.state_dict())

print("\n" + "=" * 65)
print("  STARTING MODEL TRAINING")
print("=" * 65 + "\n")

for epoch in range(EPOCHS):
    start_time = time.time()

    # --- TRAIN ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Time: {elapsed:.1f}s",
        flush=True
    )

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "arch": "mobilenet_v3_small",
                "model_state": best_model_state,
                "classes": class_names,
                "image_size": IMAGE_SIZE
            },
            MODEL_DIR / "drone_cnn_best.pth"
        )
        print(f"  [+] Saved best model (Val Accuracy: {val_acc * 100:.2f}%)", flush=True)

# Restore best weights
model.load_state_dict(best_model_state)
print(f"\nBest Validation Accuracy: {best_val_acc * 100:.2f}%", flush=True)


# ============================================================
# EVALUATION & METRICS
# ============================================================

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)

# Compute Confusion Matrix
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

np.savetxt(MODEL_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

# Compute Classification Report
report_lines = [
    f"{'Class':<20}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<8}",
    "-" * 64
]

precisions, recalls, f1s, supports = [], [], [], []

for i, cname in enumerate(class_names):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    sup = cm[i, :].sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    supports.append(sup)

    report_lines.append(f"{cname:<20}{prec:<12.4f}{rec:<12.4f}{f1:<12.4f}{sup:<8}")

total_support = sum(supports)
total_correct = np.trace(cm)
overall_acc = total_correct / total_support if total_support > 0 else 0.0

report_lines.append("-" * 64)
report_lines.append(f"{'Accuracy':<20}{'':<12}{'':<12}{overall_acc:<12.4f}{total_support:<8}")
report_lines.append(f"{'Macro Avg':<20}{np.mean(precisions):<12.4f}{np.mean(recalls):<12.4f}{np.mean(f1s):<12.4f}{total_support:<8}")

report_str = "\n".join(report_lines)

print("\n" + "=" * 65, flush=True)
print("  TEST CLASSIFICATION REPORT", flush=True)
print("=" * 65 + "\n", flush=True)
print(report_str, flush=True)

with open(MODEL_DIR / "classification_report.txt", "w") as f:
    f.write(report_str)

with open(MODEL_DIR / "classes.json", "w") as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# TRAINING GRAPHS
# ============================================================

epochs_range = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], label="Training Loss", color="#1976D2", lw=2)
plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="#FF9800", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Drone Classification Model Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(MODEL_DIR / "loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], label="Training Accuracy", color="#4CAF50", lw=2)
plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy", color="#E91E63", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Drone Classification Model Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(MODEL_DIR / "accuracy_curve.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n" + "=" * 65, flush=True)
print("Training successfully completed!", flush=True)
print(f"Model saved to: {MODEL_DIR / 'drone_cnn_best.pth'}", flush=True)
print(f"Classes saved to: {MODEL_DIR / 'classes.json'}", flush=True)
print("=" * 65, flush=True)
