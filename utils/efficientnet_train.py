import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_PATH = r"E:\Datasets\Glaucoma_detection\Fundus_Train_Val_Data\Fundus_Scanes_Sorted_And_Cropped"
TRAIN_PATH = os.path.join(BASE_PATH, "Train")
VAL_PATH = os.path.join(BASE_PATH, "Validation")
NUM_EPOCHS = 50
BATCH_SIZE = 2
LR = 3e-4
IMAGE_SIZE = 512
NUM_CLASSES = 2
CLASS_NAMES = ["Glaucoma_Negative", "Glaucoma_Positive"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GlaucomaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, crop=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.crop:
            image = crop_fundus(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def crop_fundus(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        cropped = img_array[y : y + h, x : x + w]
        return Image.fromarray(cropped)
    return image


def load_data(train_path, val_path):
    train_glaucoma_path = os.path.join(train_path, "Glaucoma_Positive")
    train_normal_path = os.path.join(train_path, "Glaucoma_Negative")
    val_glaucoma_path = os.path.join(val_path, "Glaucoma_Positive")
    val_normal_path = os.path.join(val_path, "Glaucoma_Negative")

    # Train data
    train_glaucoma_images = [
        os.path.join(train_glaucoma_path, img)
        for img in os.listdir(train_glaucoma_path)
        if img.endswith((".jpg", ".jpeg", ".png"))
    ]
    train_normal_images = [
        os.path.join(train_normal_path, img)
        for img in os.listdir(train_normal_path)
        if img.endswith((".jpg", ".jpeg", ".png"))
    ]
    train_images = train_glaucoma_images + train_normal_images
    train_labels = [1] * len(train_glaucoma_images) + [0] * len(train_normal_images)

    # Validation data
    val_glaucoma_images = [
        os.path.join(val_glaucoma_path, img)
        for img in os.listdir(val_glaucoma_path)
        if img.endswith((".jpg", ".jpeg", ".png"))
    ]
    val_normal_images = [
        os.path.join(val_normal_path, img)
        for img in os.listdir(val_normal_path)
        if img.endswith((".jpg", ".jpeg", ".png"))
    ]
    val_images = val_glaucoma_images + val_normal_images
    val_labels = [1] * len(val_glaucoma_images) + [0] * len(val_normal_images)

    train_df = pd.DataFrame({"image_path": train_images, "label": train_labels})
    val_df = pd.DataFrame({"image_path": val_images, "label": val_labels})

    print(
        f"Train: {len(train_df)} (Glaucoma: {len(train_glaucoma_images)}, Normal: {len(train_normal_images)})"
    )
    print(
        f"Validation: {len(val_df)} (Glaucoma: {len(val_glaucoma_images)}, Normal: {len(val_normal_images)})"
    )

    return train_df, val_df


def get_transforms():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_transforms


def create_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
):
    best_val_loss = float("inf")
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    history = {
        "train_loss": [],
        "val_loss": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    precision_metric = MulticlassPrecision(num_classes=NUM_CLASSES, average="macro").to(
        device
    )
    recall_metric = MulticlassRecall(num_classes=NUM_CLASSES, average="macro").to(
        device
    )
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader.dataset)
        all_preds = torch.cat(all_preds).to(device)
        all_labels = torch.cat(all_labels).to(device)

        precision = precision_metric(all_preds, all_labels)
        recall = recall_metric(all_preds, all_labels)
        f1 = f1_metric(all_preds, all_labels)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["precision"].append(precision.item())
        history["recall"].append(recall.item())
        history["f1"].append(f1.item())

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            best_metrics = {
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
            }

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print("-" * 60)

    model.load_state_dict(best_model_weights)
    return model, history, best_metrics


def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["precision"], label="Precision")
    plt.plot(history["recall"], label="Recall")
    plt.plot(history["f1"], label="F1 Score")
    plt.title("Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.axhline(0.85, color="r", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


def visualize_results(model, val_loader, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    samples_seen = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            for i in range(inputs.size(0)):
                if samples_seen >= num_samples:
                    break

                img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                img = np.clip(img, 0, 1)

                true_label = CLASS_NAMES[labels[i].item()]
                pred_label = CLASS_NAMES[torch.argmax(probs[i]).item()]

                axs[samples_seen, 0].imshow(img)
                axs[samples_seen, 0].set_title(f"True: {true_label}")
                axs[samples_seen, 0].axis("off")

                axs[samples_seen, 1].barh(CLASS_NAMES, probs[i].cpu().numpy())
                axs[samples_seen, 1].set_title(f"Pred: {pred_label}")
                axs[samples_seen, 1].set_xlim(0, 1)

                samples_seen += 1

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()


def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision = MulticlassPrecision(num_classes=NUM_CLASSES, average="macro")(
        all_preds, all_labels
    )
    recall = MulticlassRecall(num_classes=NUM_CLASSES, average="macro")(
        all_preds, all_labels
    )
    f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")(
        all_preds, all_labels
    )

    print(f"\nFinal Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    return precision.item(), recall.item(), f1.item()


def improve_model_performance(train_df, val_df):
    print("\nImproving model performance...")

    # Enhanced transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ]
    )

    # Class weights
    num_glaucoma = len(train_df[train_df["label"] == 1])
    num_normal = len(train_df[train_df["label"] == 0])
    class_weights = torch.tensor(
        [
            (num_glaucoma + num_normal) / (2.0 * num_normal),
            (num_glaucoma + num_normal) / (2.0 * num_glaucoma),
        ],
        device=device,
    )

    # Dataset and loaders
    train_dataset = GlaucomaDataset(
        train_df["image_path"].values,
        train_df["label"].values,
        transform=train_transforms,
        crop=True,
    )
    val_dataset = GlaucomaDataset(
        val_df["image_path"].values,
        val_df["label"].values,
        transform=get_transforms()[1],
        crop=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model, history, metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 30
    )

    torch.save(model.state_dict(), "improved_model.pth")
    plot_training_history(history)
    evaluate_model(model, val_loader)
    visualize_results(model, val_loader)

    return model, metrics


def main():
    train_df, val_df = load_data(TRAIN_PATH, VAL_PATH)
    train_transforms, val_transforms = get_transforms()

    train_dataset = GlaucomaDataset(
        train_df["image_path"].values,
        train_df["label"].values,
        transform=train_transforms,
        crop=False,
    )
    val_dataset = GlaucomaDataset(
        val_df["image_path"].values,
        val_df["label"].values,
        transform=val_transforms,
        crop=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    num_glaucoma = len(train_df[train_df["label"] == 1])
    num_normal = len(train_df[train_df["label"] == 0])
    class_weights = torch.tensor(
        [
            (num_glaucoma + num_normal) / (2.0 * num_normal),
            (num_glaucoma + num_normal) / (2.0 * num_glaucoma),
        ],
        device=device,
    )

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print("Training base model...")
    model, history, metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )

    torch.save(model.state_dict(), "base_model.pth")
    plot_training_history(history)
    evaluate_model(model, val_loader)
    visualize_results(model, val_loader)

    if metrics["f1"] < 0.85:
        improve_model_performance(train_df, val_df)


if __name__ == "__main__":
    main()
