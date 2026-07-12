from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import RocCurveDisplay, roc_curve


@torch.no_grad()
def extract_embeddings(extractor, loader, device) -> tuple[np.ndarray, np.ndarray]:
    extractor.eval()
    embeddings, labels = [], []
    for images, y in loader:
        emb = extractor(images.to(device))
        emb = torch.nn.functional.normalize(emb, dim=1)
        embeddings.append(emb.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pixel_change_heatmap(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    before = cv2.resize(before_rgb, (224, 224))
    after = cv2.resize(after_rgb, (224, 224))
    diff = cv2.absdiff(before, after)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def save_roc(y_true_changed, similarity_scores, out_dir: str | Path) -> float:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    change_scores = 1 - np.asarray(similarity_scores)
    fpr, tpr, thresholds = roc_curve(y_true_changed, change_scores)
    j_scores = tpr - fpr
    best_threshold = float(1 - thresholds[int(np.argmax(j_scores))])
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=180)
    plt.close()
    return best_threshold


def save_region_heatmaps(
    t1_paths: Sequence[str],
    t2_paths: Sequence[str],
    t1_labels: Sequence[int],
    t2_labels: Sequence[int],
    classes: list[str],
    out_dir: str | Path,
    n_pairs: int = 5,
) -> None:
    """Save side-by-side triptych PNGs (Before | After | Heatmap) for *n_pairs* pairs.

    Pairs are selected to include a mix of changed and unchanged regions so the
    report shows both scenarios.  At least the first *n_pairs* pairs are saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer changed pairs first, then fill with unchanged.
    changed_idx = [i for i in range(min(len(t1_paths), len(t2_paths))) if t1_labels[i] != t2_labels[i]]
    unchanged_idx = [i for i in range(min(len(t1_paths), len(t2_paths))) if t1_labels[i] == t2_labels[i]]
    selected = (changed_idx + unchanged_idx)[:n_pairs]

    for rank, idx in enumerate(selected, start=1):
        before_img = np.array(Image.open(t1_paths[idx]).convert("RGB"))
        after_img = np.array(Image.open(t2_paths[idx]).convert("RGB"))
        heatmap = pixel_change_heatmap(before_img, after_img)

        status = "changed" if t1_labels[idx] != t2_labels[idx] else "unchanged"
        t1_cls = classes[t1_labels[idx]] if t1_labels[idx] < len(classes) else str(t1_labels[idx])
        t2_cls = classes[t2_labels[idx]] if t2_labels[idx] < len(classes) else str(t2_labels[idx])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(before_img)
        axes[0].set_title(f"T1 — {t1_cls}", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(after_img)
        axes[1].set_title(f"T2 — {t2_cls}", fontsize=11)
        axes[1].axis("off")
        axes[2].imshow(heatmap)
        axes[2].set_title(f"Pixel-diff heatmap ({status})", fontsize=11)
        axes[2].axis("off")

        fig.suptitle(f"Region pair {rank} — {status.upper()}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fname = out_dir / f"heatmap_pair_{rank:02d}_{status}.png"
        plt.savefig(fname, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")
