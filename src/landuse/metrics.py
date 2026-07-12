from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, confidences = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
    return y_true, y_pred, confidences


def write_classification_artifacts(y_true, y_pred, classes: list[str], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()

