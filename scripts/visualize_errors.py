"""Visualise the top-5 misclassified tile pairs (true vs predicted).

This script reads the top_misclassifications.json produced by evaluate.py,
re-opens the actual tile images from the dataset, and saves an annotated
image grid showing the misclassified samples with hypotheses about failure modes.

Usage
-----
    python scripts/visualize_errors.py \\
        --data-dir data/eurosat \\
        --checkpoint runs/resnet18/best.pt \\
        --eval-dir runs/resnet18/eurosat_eval \\
        --out-dir runs/resnet18/error_analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.metrics import collect_predictions
from landuse.models import load_checkpoint

# Hard-coded confusion hypotheses for known class pairs.
# These are filled in where confidence in the explanation is high.
_HYPOTHESES: dict[tuple[str, str], str] = {
    ("AnnualCrop", "PermanentCrop"): "Similar texture; both show row-crop patterns.",
    ("PermanentCrop", "AnnualCrop"): "Seasonal variation makes crops indistinguishable.",
    ("HerbaceousVegetation", "Pasture"): "Uniform green spectral response.",
    ("Pasture", "HerbaceousVegetation"): "Uniform green spectral response.",
    ("Highway", "Industrial"): "Grey impervious surfaces look alike at this resolution.",
    ("Industrial", "Highway"): "Linear structures visible in both classes.",
    ("River", "SeaLake"): "Water body; scale determines class.",
    ("SeaLake", "River"): "Open water; context (edges) lost in centre-crop.",
    ("Forest", "HerbaceousVegetation"): "Dense canopy shadows resemble herbaceous texture.",
    ("Residential", "Industrial"): "Dense building footprints in both classes.",
}


def _hypothesis(true_cls: str, pred_cls: str) -> str:
    key = (true_cls, pred_cls)
    return _HYPOTHESES.get(key, "Mixed spectral signature — boundary / transitional zone likely.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise top-5 misclassified tiles.")
    parser.add_argument("--data-dir", required=True, help="EuroSAT dataset root.")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned .pt checkpoint.")
    parser.add_argument(
        "--eval-dir",
        default="runs/resnet18/eurosat_eval",
        help="Directory containing top_misclassifications.json from evaluate.py.",
    )
    parser.add_argument("--out-dir", default="runs/resnet18/error_analysis")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of misclassified examples to visualise.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model & dataset ------------------------------------------------
    dataset = load_image_folder(args.data_dir, train=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes", dataset.classes) if isinstance(checkpoint, dict) else dataset.classes
    model = load_checkpoint(args.checkpoint, num_classes=len(classes), device=device)

    # --- Collect fresh predictions (block test split) -----------------------
    splits = split_dataset(dataset, split="block")
    loader = make_loader(splits.test, args.batch_size)
    y_true, y_pred, confidences = collect_predictions(model, loader, device)

    # Map back to dataset sample paths
    test_paths = [dataset.samples[i][0] for i in splits.test.indices]

    errors = [
        {
            "path": test_paths[i],
            "true": dataset.classes[y_true[i]],
            "predicted": classes[y_pred[i]] if y_pred[i] < len(classes) else "?",
            "confidence": confidences[i],
        }
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]

    # Sort by confidence descending (high-confidence wrong = most interesting)
    errors.sort(key=lambda e: e["confidence"], reverse=True)
    top_errors = errors[: args.top_k]

    # Save JSON
    (out_dir / "top_misclassifications.json").write_text(json.dumps(top_errors, indent=2))

    # --- Render image grid ---------------------------------------------------
    n = len(top_errors)
    if n == 0:
        print("No misclassifications found on the test split — model might be perfect!")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, err in zip(axes, top_errors):
        img = Image.open(err["path"]).convert("RGB")
        ax.imshow(np.array(img))
        hyp = _hypothesis(err["true"], err["predicted"])
        title = (
            f"True: {err['true']}\n"
            f"Pred: {err['predicted']}\n"
            f"Conf: {err['confidence']:.1%}\n"
        )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(f"Hypothesis:\n{hyp}", fontsize=7, wrap=True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle(
        f"Top-{n} High-Confidence Misclassifications (EuroSAT block-split test set)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    grid_path = out_dir / "top_misclassifications_grid.png"
    plt.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved grid to {grid_path}")

    # --- Per-class confusion bar chart ---------------------------------------
    from collections import Counter
    confusion_pairs = Counter((e["true"], e["predicted"]) for e in errors)
    top_pairs = confusion_pairs.most_common(10)
    labels = [f"{t}\n→{p}" for (t, p), _ in top_pairs]
    counts = [c for _, c in top_pairs]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.barh(labels[::-1], counts[::-1], color="salmon")
    ax2.set_xlabel("Error count")
    ax2.set_title("Most Frequent Misclassification Pairs (test set)")
    plt.tight_layout()
    pairs_path = out_dir / "confusion_pair_chart.png"
    plt.savefig(pairs_path, dpi=180)
    plt.close()
    print(f"Saved pair chart to {pairs_path}")

    print(f"\nTop-{args.top_k} misclassifications:")
    for i, err in enumerate(top_errors, 1):
        print(f"  {i}. True={err['true']}, Pred={err['predicted']}, Conf={err['confidence']:.1%}")
        print(f"     Hypothesis: {_hypothesis(err['true'], err['predicted'])}")
    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
