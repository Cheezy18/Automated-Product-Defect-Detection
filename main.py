import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# AUTOMATED PRODUCT DEFECT DETECTION
# Binary Classification: Defective vs Normal
# Model: Pretrained ResNet18
# ============================================================

# -----------------------------
# 1. Configuration
# -----------------------------
# Set DATA_DIR to the location of the extracted AITEX dataset.
#
# Expected structure:
# DATA_DIR/
# ├── train/
# │   ├── defective/
# │   └── normal/
# └── val/
#     ├── defective/
#     └── normal/

DATA_DIR = os.getenv("DATA_DIR", "aitex_120_fast/aitex_120_fast")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

RESULTS_DIR = "results"
MODEL_DIR = "model"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------
# 2. Preprocessing
# -----------------------------
# Images are resized to 224 x 224, converted to 3 channels,
# converted to tensors and normalized.
#
# No random augmentation is used, following the assessment format.

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

print("Preprocessing and normalization ready!")


# -----------------------------
# 3. Prepare Dataset
# -----------------------------
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}\n"
        "Set the DATA_DIR environment variable to your extracted dataset path."
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found: {VAL_DIR}\n"
        "Set the DATA_DIR environment variable to your extracted dataset path."
    )

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# 4. Data Loaders
# -----------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# -----------------------------
# 5. Display Sample Images
# -----------------------------
images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

for i, ax in enumerate(axes.flat):
    img = images[i].permute(1, 2, 0).numpy()

    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.set_title(train_dataset.classes[labels[i]])
    ax.axis("off")

plt.suptitle("Sample Training Images", fontsize=16)
plt.tight_layout()

sample_path = os.path.join(RESULTS_DIR, "sample_training_images.png")
plt.savefig(sample_path, dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------
# 6. Pretrained ResNet18 Model
# -----------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classification layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Move model to GPU/CPU
model = model.to(device)

print("\nModel loaded successfully!")
print("Number of input features to final layer:", num_features)
print("Number of output classes:", 2)


# -----------------------------
# 7. Identify Frozen/Trainable Layers
# -----------------------------
trainable_params = []
frozen_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)
    else:
        frozen_params.append(name)

print("\nTRAINABLE LAYERS:")
for name in trainable_params:
    print(name)

print("\nNumber of frozen parameter tensors:", len(frozen_params))
print("Number of trainable parameter tensors:", len(trainable_params))


# -----------------------------
# 8. Loss Function and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

print("\nLoss Function: CrossEntropyLoss")
print("Optimizer: Adam")
print("Learning Rate: 0.001")


# -----------------------------
# 9. Model Training
# -----------------------------
num_epochs = 10

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):

    # ---- Training ----
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total


    # ---- Validation ----
    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = val_correct / val_total


    # ---- Store metrics ----
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


# -----------------------------
# 10. Training/Validation Loss
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

loss_path = os.path.join(
    RESULTS_DIR,
    "training_validation_loss.png"
)

plt.savefig(loss_path, dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------
# 11. Training/Validation Accuracy
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    RESULTS_DIR,
    "training_validation_accuracy.png"
)

plt.savefig(accuracy_path, dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------
# 12. Model Evaluation
# -----------------------------
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

defective_index = train_dataset.class_to_idx["defective"]

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=defective_index,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=defective_index,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=defective_index,
    zero_division=0
)

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(f"Training Accuracy  : {train_accuracies[-1]:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1-Score           : {f1:.4f}")


# -----------------------------
# 13. Classification Report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes,
        zero_division=0
    )
)


# -----------------------------
# 14. Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.xticks(
    [0, 1],
    train_dataset.classes
)

plt.yticks(
    [0, 1],
    train_dataset.classes
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

plt.colorbar()
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------
# 15. Error Analysis
# -----------------------------
misclassified = []

for i, (image_path, actual_label) in enumerate(
    val_dataset.samples
):

    predicted_label = all_predictions[i]

    if actual_label != predicted_label:

        misclassified.append({
            "path": image_path,
            "actual": val_dataset.classes[actual_label],
            "predicted": val_dataset.classes[predicted_label]
        })


print(
    "\nNumber of misclassified images:",
    len(misclassified)
)

for item in misclassified:

    print("\nImage:", os.path.basename(item["path"]))
    print("Actual:", item["actual"])
    print("Predicted:", item["predicted"])


# Display up to 6 misclassified images
num_images = min(6, len(misclassified))

if num_images > 0:

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 7)
    )

    axes = axes.flatten()

    for i in range(num_images):

        item = misclassified[i]

        img = Image.open(
            item["path"]
        ).convert("L")

        axes[i].imshow(
            img,
            cmap="gray"
        )

        axes[i].set_title(
            f"Actual: {item['actual']}\n"
            f"Predicted: {item['predicted']}"
        )

        axes[i].axis("off")

    for i in range(num_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "Misclassified Images - Error Analysis",
        fontsize=16
    )

    plt.tight_layout()

    error_path = os.path.join(
        RESULTS_DIR,
        "misclassified_images.png"
    )

    plt.savefig(
        error_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# -----------------------------
# 16. Save Error Analysis CSV
# -----------------------------
if len(misclassified) > 0:

    error_df = pd.DataFrame(
        misclassified
    )

    reasons = [
        "Defect may be small or visually subtle.",
        "Normal texture may resemble a defective pattern.",
        "Lighting or texture variation may confuse the model.",
        "Similar visual appearance between classes.",
        "Limited training examples may affect generalization.",
        "Background or fabric texture may resemble a defect.",
        "Defect may not be sufficiently prominent."
    ]

    error_df["possible_reason"] = [
        reasons[i % len(reasons)]
        for i in range(len(error_df))
    ]

    error_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "error_analysis.csv"
        ),
        index=False
    )


# -----------------------------
# 17. Save Results Table
# -----------------------------
results = pd.DataFrame({
    "Metric": [
        "Training Accuracy",
        "Validation Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Score": [
        train_accuracies[-1],
        accuracy,
        precision,
        recall,
        f1
    ]
})

results["Score"] = results["Score"].round(4)

print("\nResults Table:")
print(results.to_string(index=False))

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "metrics.csv"
    ),
    index=False
)


# -----------------------------
# 18. Save Trained Model
# -----------------------------
model_path = os.path.join(
    MODEL_DIR,
    "train-model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\nModel saved successfully!")
print("Saved at:", model_path)

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")
