from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from landuse.change import cosine_similarity, save_roc
from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.models import ResNetEmbeddingExtractor, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs/change_detection")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_image_folder(args.data_dir, train=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes", dataset.classes) if isinstance(checkpoint, dict) else dataset.classes
    model = load_checkpoint(args.checkpoint, num_classes=len(classes), device=device)
    extractor = ResNetEmbeddingExtractor(model).to(device)

    splits = split_dataset(dataset, split="block")
    t1_loader = make_loader(splits.train, args.batch_size)
    t2_loader = make_loader(splits.test, args.batch_size)

    from landuse.change import extract_embeddings

    t1_embeddings, t1_labels = extract_embeddings(extractor, t1_loader, device)
    t2_embeddings, t2_labels = extract_embeddings(extractor, t2_loader, device)
    pair_count = min(len(t1_embeddings), len(t2_embeddings))
    similarities = [cosine_similarity(t1_embeddings[i], t2_embeddings[i]) for i in range(pair_count)]
    changed = [int(t1_labels[i] != t2_labels[i]) for i in range(pair_count)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = save_roc(changed, similarities, out_dir)
    summary = {
        "pair_count": pair_count,
        "selected_similarity_threshold": threshold,
        "mean_similarity": float(np.mean(similarities)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

