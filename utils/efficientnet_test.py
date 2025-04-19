import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score, Precision, Recall
from torchvision.models import efficientnet_b0

# Конфигурация
CLASS_NAMES = ["Glaucoma_Negative", "Glaucoma_Positive"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_test_data(test_path):
    test_data = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(class_dir, img_name)
                test_data.append(
                    {"image_path": img_path, "label": CLASS_NAMES.index(class_name)}
                )
    return pd.DataFrame(test_data)


def load_model(model_path, num_classes):
    model = efficientnet_b0(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def calculate_metrics(model, test_loader, num_classes):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
    recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    precision = precision(torch.tensor(all_preds), torch.tensor(all_labels))
    recall = recall(torch.tensor(all_preds), torch.tensor(all_labels))
    f1 = f1(torch.tensor(all_preds), torch.tensor(all_labels))

    return precision.item(), recall.item(), f1.item(), all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("test_confusion_matrix.png")
    plt.show()


def arguments_parser():
    parser = argparse.ArgumentParser(
        description="Скрипт для тестирования классификатора EfficientNet в формате PyTorch"
    )
    parser.add_argument(
        "-s", "--subset_path", type=str, help="Путь к выборке для тестирования"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Путь к чекпойнту обученной модели EfficientNet",
    )
    parser.add_argument(
        "-is",
        "--image_size",
        type=int,
        help="Размерность входа модели EfficientNet",
        default=512,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="Размерность батча для тестирования",
        default=16,
    )
    parser.add_argument(
        "-nc", "--num_classes", type=int, help="Число классов в модели", default=2
    )
    args = parser.parse_args()
    return args


def main():
    # Загрузка данных
    args = arguments_parser()
    test_path = args.subset_path
    model_path = args.checkpoint_path
    image_size = args.image_size
    batch_size = args.batch_size
    num_classes = args.num_classes

    test_df = load_test_data(test_path)
    print(f"Loaded {len(test_df)} test samples")

    # Трансформы (должны совпадать с тренировочными)
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Создание Dataset и DataLoader
    test_dataset = TestDataset(test_df, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Загрузка модели
    model = load_model(model_path, num_classes)
    print("Model loaded successfully")

    # Расчет метрик
    precision, recall, f1, all_preds, all_labels = calculate_metrics(
        model, test_loader, num_classes
    )

    # Вывод результатов
    print("\nTest Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds)


if __name__ == "__main__":
    main()
