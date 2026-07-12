"""Save visual change heatmaps for ≥5 sample region pairs.

Usage
-----
    python scripts/save_change_heatmaps.py \\
        --data-dir data/eurosat \\
        --checkpoint runs/resnet18/best.pt \\
        --out-dir runs/change_detection/heatmaps \\
        --n-pairs 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from landuse.change import extract_embeddings, save_region_heatmaps
from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.models import ResNetEmbeddingExtractor, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Save change heatmap triptychs for n region pairs.")
    parser.add_argument("--data-dir", required=True, help="Path to EuroSAT dataset root.")
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned .pt checkpoint.")
    parser.add_argument("--out-dir", default="runs/change_detection/heatmaps")
    parser.add_argument("--n-pairs", type=int, default=5, help="Number of region pairs to save.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load dataset and checkpoint ----------------------------------------
    dataset = load_image_folder(args.data_dir, train=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes", dataset.classes) if isinstance(checkpoint, dict) else dataset.classes
    model = load_checkpoint(args.checkpoint, num_classes=len(classes), device=device)
    extractor = ResNetEmbeddingExtractor(model).to(device)

    # --- Spatial block split (T1 = train, T2 = test) ------------------------
    splits = split_dataset(dataset, split="block")
    t1_loader = make_loader(splits.train, args.batch_size)
    t2_loader = make_loader(splits.test, args.batch_size)

    print("Extracting T1 embeddings ...")
    _, t1_labels = extract_embeddings(extractor, t1_loader, device)
    print("Extracting T2 embeddings ...")
    _, t2_labels = extract_embeddings(extractor, t2_loader, device)

    # Resolve raw file paths from the Subset indices
    t1_paths = [dataset.samples[i][0] for i in splits.train.indices]
    t2_paths = [dataset.samples[i][0] for i in splits.test.indices]

    # Labels from split are aligned with the paths list ordering
    t1_label_list = [int(dataset.samples[i][1]) for i in splits.train.indices]
    t2_label_list = [int(dataset.samples[i][1]) for i in splits.test.indices]

    print(f"\nSaving {args.n_pairs} heatmap triptychs to '{args.out_dir}' ...")
    save_region_heatmaps(
        t1_paths=t1_paths,
        t2_paths=t2_paths,
        t1_labels=t1_label_list,
        t2_labels=t2_label_list,
        classes=classes,
        out_dir=args.out_dir,
        n_pairs=args.n_pairs,
    )

    # Write a summary JSON for the report
    out_dir = Path(args.out_dir)
    summary = {
        "n_pairs_saved": args.n_pairs,
        "t1_size": len(t1_paths),
        "t2_size": len(t2_paths),
        "changed_pairs": sum(1 for a, b in zip(t1_label_list, t2_label_list) if a != b),
        "unchanged_pairs": sum(1 for a, b in zip(t1_label_list, t2_label_list) if a == b),
    }
    (out_dir / "heatmap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary written to {out_dir / 'heatmap_summary.json'}")
    print(f"Changed pairs found  : {summary['changed_pairs']}")
    print(f"Unchanged pairs found: {summary['unchanged_pairs']}")


if __name__ == "__main__":
    main()
