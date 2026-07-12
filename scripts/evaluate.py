from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.metrics import collect_predictions, write_classification_artifacts
from landuse.models import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=["block", "random", "all"], default="all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_image_folder(args.data_dir, train=False)
    eval_set = dataset if args.split == "all" else split_dataset(dataset, split=args.split).test
    loader = make_loader(eval_set, args.batch_size)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes", dataset.classes) if isinstance(checkpoint, dict) else dataset.classes
    model = load_checkpoint(args.checkpoint, num_classes=len(classes), device=device)
    y_true, y_pred, confidences = collect_predictions(model, loader, device)
    write_classification_artifacts(y_true, y_pred, dataset.classes, args.out_dir)

    rows = [
        {"index": i, "true": dataset.classes[t], "predicted": classes[p], "confidence": c}
        for i, (t, p, c) in enumerate(zip(y_true, y_pred, confidences))
        if t != p and p < len(classes)
    ]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "top_misclassifications.json").write_text(json.dumps(rows[:5], indent=2))


if __name__ == "__main__":
    main()

