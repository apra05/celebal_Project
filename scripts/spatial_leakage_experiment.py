"""Spatial leakage experiment: compare block-split vs random-split accuracy.

This script evaluates the fine-tuned model under two train/test split strategies
and writes a quantified markdown report comparing per-class F1 and macro-F1.

Usage
-----
    python scripts/spatial_leakage_experiment.py \\
        --data-dir data/eurosat \\
        --checkpoint runs/resnet18/best.pt \\
        --out-dir runs/spatial_leakage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report

from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.metrics import collect_predictions, write_classification_artifacts
from landuse.models import load_checkpoint


def _evaluate_split(model, dataset, split_mode: str, batch_size: int, device, out_dir: Path) -> dict:
    """Run evaluation under a given split mode and return the classification report dict."""
    splits = split_dataset(dataset, split=split_mode)
    loader = make_loader(splits.test, batch_size)
    y_true, y_pred, _ = collect_predictions(model, loader, device)
    write_classification_artifacts(y_true, y_pred, dataset.classes, out_dir)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    return report


def _plot_comparison(block_report: dict, random_report: dict, classes: list[str], out_dir: Path) -> None:
    """Bar chart comparing per-class F1 under block vs random split."""
    block_f1 = [block_report.get(c, {}).get("f1-score", 0.0) for c in classes]
    random_f1 = [random_report.get(c, {}).get("f1-score", 0.0) for c in classes]

    x = range(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar([i - width / 2 for i in x], block_f1, width, label="Block split (spatial)")
    bars2 = ax.bar([i + width / 2 for i in x], random_f1, width, label="Random split")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=40, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1: Block split vs Random split (Spatial Leakage Experiment)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "split_comparison_f1.png", dpi=180)
    plt.close()


def _write_markdown(block_report: dict, random_report: dict, classes: list[str], out_dir: Path) -> None:
    """Write a 1-page markdown analysis of the spatial leakage experiment."""
    block_macro = block_report["macro avg"]["f1-score"]
    random_macro = random_report["macro avg"]["f1-score"]
    delta = random_macro - block_macro

    lines = [
        "# Spatial Leakage Experiment",
        "",
        "## Overview",
        "",
        "Satellite imagery datasets typically tile a continuous geographic area into",
        "overlapping or adjacent patches.  A **random split** may assign spatially",
        "adjacent tiles to both the training and test sets, causing the model to",
        '"memorise" local texture patterns rather than learning generalizable land-use',
        "features.  This is called **spatial leakage**.",
        "",
        "We compare two split strategies on the EuroSAT dataset:",
        "",
        "| Strategy | Description |",
        "|---|---|",
        "| **Block split** | Tiles are grouped by a SHA-1 hash of their filename into 20 spatial blocks.  Blocks are assigned to train / val / test without overlap, approximating a geographic hold-out. |",
        "| **Random split** | Tiles are shuffled with a fixed seed and split 70 / 15 / 15.  Adjacent tiles can appear in train AND test. |",
        "",
        "## Results",
        "",
        f"| Metric | Block split | Random split | Delta |",
        f"|---|---|---|---|",
        f"| Macro-F1 | {block_macro:.4f} | {random_macro:.4f} | {delta:+.4f} |",
        "",
        "### Per-class F1",
        "",
        "| Class | Block split | Random split | Delta |",
        "|---|---|---|---|",
    ]

    for cls in classes:
        b = block_report.get(cls, {}).get("f1-score", 0.0)
        r = random_report.get(cls, {}).get("f1-score", 0.0)
        lines.append(f"| {cls} | {b:.4f} | {r:.4f} | {r - b:+.4f} |")

    lines += [
        "",
        "## Analysis",
        "",
    ]

    if delta > 0.02:
        lines += [
            f"The random split yields a macro-F1 **{delta:.1%} higher** than the block split.",
            "This gap is consistent with **spatial leakage**: the model exploits local texture",
            "correlation between adjacent training and test tiles, inflating apparent accuracy.",
            "The block split provides a more realistic estimate of generalisation to unseen geography.",
        ]
    elif delta < -0.02:
        lines += [
            f"The block split yields a macro-F1 **{abs(delta):.1%} higher** than the random split.",
            "This suggests the model generalises well across geographic regions — spatial leakage",
            "does not appear to be inflating the random-split score, possibly because EuroSAT tiles",
            "are already diverse enough across the spatial blocks.",
        ]
    else:
        lines += [
            f"The two strategies produce similar macro-F1 scores (delta = {delta:+.4f}).",
            "Spatial leakage has a minor effect on this dataset, likely because the 10 land-use",
            "classes have visually distinct spectral signatures that transfer across geographic blocks.",
        ]

    lines += [
        "",
        "## Conclusion",
        "",
        "The block split is the recommended evaluation strategy for this project because it",
        "prevents artificially inflated metrics caused by spatially correlated train/test tiles.",
        "All reported results in the main report use the block split.",
        "",
        "![Per-class F1 comparison](split_comparison_f1.png)",
    ]

    (out_dir / "spatial_leakage_writeup.md").write_text("\n".join(lines))
    print(f"  Wrote {out_dir / 'spatial_leakage_writeup.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial leakage experiment: block vs random split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs/spatial_leakage")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_image_folder(args.data_dir, train=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes", dataset.classes) if isinstance(checkpoint, dict) else dataset.classes
    model = load_checkpoint(args.checkpoint, num_classes=len(classes), device=device)

    print("Evaluating with block split ...")
    block_report = _evaluate_split(model, dataset, "block", args.batch_size, device, out_dir / "block")

    print("Evaluating with random split ...")
    random_report = _evaluate_split(model, dataset, "random", args.batch_size, device, out_dir / "random")

    # Save raw reports
    (out_dir / "block_report.json").write_text(json.dumps(block_report, indent=2))
    (out_dir / "random_report.json").write_text(json.dumps(random_report, indent=2))

    print("Generating comparison plot ...")
    _plot_comparison(block_report, random_report, dataset.classes, out_dir)

    print("Writing markdown analysis ...")
    _write_markdown(block_report, random_report, dataset.classes, out_dir)

    block_macro = block_report["macro avg"]["f1-score"]
    random_macro = random_report["macro avg"]["f1-score"]
    print(f"\nBlock  macro-F1 : {block_macro:.4f}")
    print(f"Random macro-F1 : {random_macro:.4f}")
    print(f"Delta           : {random_macro - block_macro:+.4f}")
    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
