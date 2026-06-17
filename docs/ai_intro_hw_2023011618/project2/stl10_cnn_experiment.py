import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]
MEAN = np.array([0.4467, 0.4398, 0.4066], dtype=np.float32)
STD = np.array([0.2603, 0.2566, 0.2713], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimpleTransform:
    def __init__(self, train=False, augment=False, image_size=64):
        self.train = train
        self.augment = augment
        self.image_size = image_size

    def __call__(self, image):
        if self.train and self.augment:
            image = self.random_crop(image, padding=8)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image = self.center_crop(image)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    @staticmethod
    def random_crop(image, padding):
        arr = np.asarray(image)
        arr = np.pad(arr, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
        h, w = image.size[1], image.size[0]
        top = random.randint(0, padding * 2)
        left = random.randint(0, padding * 2)
        return Image.fromarray(arr[top : top + h, left : left + w])

    @staticmethod
    def center_crop(image):
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return image.crop((left, top, left + side, top + side))


class ImageFolderDataset(Dataset):
    def __init__(self, root, split, transform):
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []
        for label, class_name in enumerate(CLASSES):
            class_dir = self.root / class_name
            paths = sorted(class_dir.glob("*.png"))
            self.samples.extend((path, label) for path in paths)
        if not self.samples:
            raise RuntimeError(f"No PNG images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label, str(path)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, use_bn, pool):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(make_activation(activation))
        layers.append(
            nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def make_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class STL10CNN(nn.Module):
    def __init__(self, num_classes=10, activation="relu", pool="max", use_bn=True, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, activation, use_bn, pool),
            ConvBlock(32, 64, activation, use_bn, pool),
            ConvBlock(64, 128, activation, use_bn, pool),
            ConvBlock(128, 256, activation, use_bn, pool),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def compute_metrics(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    precision = []
    recall = []
    f1 = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    accuracy = np.trace(cm) / cm.sum()
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(num_classes)
        },
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true = []
    y_pred = []
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total
    return metrics


def denormalize(tensor):
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * STD + MEAN
    return np.clip(arr, 0.0, 1.0)


def save_ppm(path, array):
    path = Path(path)
    image = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    with path.open("wb") as f:
        f.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        f.write(image.tobytes())


def make_colormap(gray):
    gray = np.clip(gray, 0.0, 1.0)
    return np.stack(
        [
            np.clip(2.0 * gray, 0.0, 1.0),
            np.clip(1.0 - np.abs(gray - 0.5) * 2.0, 0.0, 1.0),
            np.clip(2.0 * (1.0 - gray), 0.0, 1.0),
        ],
        axis=-1,
    )


def save_gradcam(model, dataset, output_dir, device, count=6):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    target_layer = model.features[-1].block[0]
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    chosen = []
    step = max(1, len(dataset) // count)
    for index in range(0, len(dataset), step):
        chosen.append(index)
        if len(chosen) >= count:
            break
    rows = []
    for index in chosen:
        image, label, path = dataset[index]
        x = image.unsqueeze(0).to(device)
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
        score = logits[0, pred]
        model.zero_grad()
        score.backward()
        grads = gradients["value"][0]
        acts = activations["value"][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
        base = denormalize(image)
        heatmap = make_colormap(cam.detach().cpu().numpy())
        overlay = np.clip(0.55 * base + 0.45 * heatmap, 0.0, 1.0)
        out_name = f"gradcam_{index:04d}_{CLASSES[label]}_pred_{CLASSES[pred]}.ppm"
        save_ppm(output_dir / out_name, overlay)
        rows.append({"image": Path(path).name, "label": CLASSES[label], "prediction": CLASSES[pred], "file": out_name})
    handle_f.remove()
    handle_b.remove()
    with (output_dir / "gradcam_index.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_history_csv(path, history):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "precision_macro", "recall_macro", "f1_macro"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def run_experiment(args, name, activation, pool, use_bn, dropout, augment):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ImageFolderDataset(args.data_dir, "train", SimpleTransform(train=True, augment=augment, image_size=args.image_size))
    test_dataset = ImageFolderDataset(args.data_dir, "test", SimpleTransform(train=False, augment=False, image_size=args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = STL10CNN(activation=activation, pool=pool, use_bn=use_bn, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    history = []
    best_f1 = -math.inf
    best_metrics = None
    best_path = output_dir / "best_model.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "f1_macro": test_metrics["f1_macro"],
        }
        history.append(row)
        print(
            f"{name} epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
            f"f1={test_metrics['f1_macro']:.4f}",
            flush=True,
        )
        if test_metrics["f1_macro"] > best_f1:
            best_f1 = test_metrics["f1_macro"]
            best_metrics = dict(test_metrics)
            torch.save({"model": model.state_dict(), "config": vars(args), "experiment": name}, best_path)
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, test_loader, criterion, device)
    if best_metrics is not None:
        final_metrics["best_epoch_metrics"] = best_metrics
    final_metrics["seconds"] = time.time() - start
    final_metrics["experiment"] = {
        "name": name,
        "activation": activation,
        "pool": pool,
        "batch_norm": use_bn,
        "dropout": dropout,
        "augment": augment,
        "epochs": args.epochs,
        "image_size": args.image_size,
    }
    write_history_csv(output_dir / "history.csv", history)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    if args.gradcam:
        save_gradcam(model, test_dataset, output_dir / "gradcam", device, count=args.gradcam_count)
    return final_metrics


def get_experiments(args):
    if args.preset == "main":
        return [
            ("baseline_relu_max", "relu", "max", False, 0.0, False),
            ("aug_bn_dropout_relu_max", "relu", "max", True, 0.35, True),
            ("tanh_avg_bn_dropout", "tanh", "avg", True, 0.35, True),
            ("sigmoid_max_bn_dropout", "sigmoid", "max", True, 0.35, True),
        ]
    if args.preset == "single":
        return [(args.name, args.activation, args.pool, args.batch_norm, args.dropout, args.augment)]
    raise ValueError(args.preset)


def main():
    parser = argparse.ArgumentParser(description="STL10 CNN experiments without torchvision dependency.")
    parser.add_argument("--data-dir", default="../08_项目2_CNN_STL10/STL10")
    parser.add_argument("--output-dir", default="../results/cnn_stl10")
    parser.add_argument("--preset", choices=["main", "single"], default="main")
    parser.add_argument("--name", default="custom")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--pool", choices=["max", "avg"], default="max")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--gradcam-count", type=int, default=6)
    args = parser.parse_args()

    all_metrics = []
    for exp in get_experiments(args):
        all_metrics.append(run_experiment(args, *exp))

    output_dir = Path(args.output_dir)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "activation",
                "pool",
                "batch_norm",
                "dropout",
                "augment",
                "loss",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "seconds",
            ],
        )
        writer.writeheader()
        for metric in all_metrics:
            exp = metric["experiment"]
            writer.writerow(
                {
                    "name": exp["name"],
                    "activation": exp["activation"],
                    "pool": exp["pool"],
                    "batch_norm": exp["batch_norm"],
                    "dropout": exp["dropout"],
                    "augment": exp["augment"],
                    "loss": metric["loss"],
                    "accuracy": metric["accuracy"],
                    "precision_macro": metric["precision_macro"],
                    "recall_macro": metric["recall_macro"],
                    "f1_macro": metric["f1_macro"],
                    "seconds": metric["seconds"],
                }
            )
    print(f"Summary written to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
