"""Train the optional hybrid seedling-versus-hard-negative classifier.

Expected directory layout::

    dataset/{train,valid,test}/{seedling,negative}/*.jpg

Use ``prepare_seedling_dataset.py`` to build this layout from flight/site
point annotations.  The script never changes the Detectree2 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from .seedling_detector import TinySeedlingClassifier
except ImportError:
    from seedling_detector import TinySeedlingClassifier


CLASS_NAMES = ("negative", "seedling")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _datasets(root: Path, input_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.20, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train = datasets.ImageFolder(str(root / "train"), transform=train_transform)
    valid = datasets.ImageFolder(str(root / "valid"), transform=eval_transform)
    test = datasets.ImageFolder(str(root / "test"), transform=eval_transform)
    for name, dataset in (("train", train), ("valid", valid), ("test", test)):
        if tuple(dataset.classes) != CLASS_NAMES:
            raise ValueError(f"{name} classes must be exactly {CLASS_NAMES}, got {dataset.classes}")
        if not dataset:
            raise ValueError(f"{name} split is empty")
    return train, valid, test


@torch.inference_mode()
def _scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_scores = []
    all_labels = []
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1]
        all_scores.extend(float(value) for value in probabilities.cpu().tolist())
        all_labels.extend(int(value) for value in labels.tolist())
    return np.asarray(all_scores, dtype=np.float32), np.asarray(all_labels, dtype=np.int64)


def _metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _select_threshold(scores: np.ndarray, labels: np.ndarray, min_precision: float) -> Tuple[float, Dict[str, float]]:
    candidates = np.linspace(0.50, 0.99, 100)
    metrics = [_metrics(scores, labels, float(threshold)) for threshold in candidates]
    constrained = [item for item in metrics if item["precision"] >= min_precision]
    selected = max(constrained or metrics, key=lambda item: (item["f1"], item["precision"], item["recall"]))
    return float(selected["threshold"]), selected


def train(args: argparse.Namespace) -> Dict[str, object]:
    _seed_everything(args.seed)
    device = torch.device(args.device)
    train_set, valid_set, test_set = _datasets(args.data_root, args.input_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    counts = np.bincount(train_set.targets, minlength=2).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts * 2.0, 1.0), device=device)
    model = TinySeedlingClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)

        valid_scores, valid_labels = _scores(model, valid_loader, device)
        threshold, valid_metrics = _select_threshold(valid_scores, valid_labels, args.min_precision)
        candidate = (valid_metrics["f1"], valid_metrics["precision"], valid_metrics["recall"])
        if best is None or candidate > best:
            best = candidate
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} loss={running_loss / max(len(train_set), 1):.4f} "
            f"valid_precision={valid_metrics['precision']:.3f} "
            f"valid_recall={valid_metrics['recall']:.3f} "
            f"valid_f1={valid_metrics['f1']:.3f} threshold={threshold:.2f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    valid_scores, valid_labels = _scores(model, valid_loader, device)
    threshold, valid_metrics = _select_threshold(valid_scores, valid_labels, args.min_precision)
    test_scores, test_labels = _scores(model, test_loader, device)
    test_metrics = _metrics(test_scores, test_labels, threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "tiny_seedling_classifier_v1",
        "state_dict": model.state_dict(),
        "metadata": {
            "class_names": list(CLASS_NAMES),
            "input_size": args.input_size,
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "classifier_threshold": threshold,
            "min_precision_target": args.min_precision,
            "train_samples": len(train_set),
            "valid_samples": len(valid_set),
            "test_samples": len(test_set),
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    }
    torch.save(checkpoint, args.output)
    metadata_path = args.output.with_name("model_metadata.json")
    metadata_path.write_text(json.dumps(checkpoint["metadata"], indent=2), encoding="utf-8")
    return checkpoint["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/seedling_classifier/best.pt"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-precision", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
