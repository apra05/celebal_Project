from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.models import ScratchCNN
from landuse.train import fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/baseline")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--split", choices=["block", "random"], default="block")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_image_folder(args.data_dir, train=True)
    splits = split_dataset(dataset, split=args.split)
    train_loader = make_loader(splits.train, args.batch_size, shuffle=True)
    val_loader = make_loader(splits.val, args.batch_size)

    model = ScratchCNN(num_classes=len(dataset.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = fit(model, train_loader, val_loader, optimizer, args.epochs, device, args.out_dir)

    out_dir = Path(args.out_dir)
    (out_dir / "classes.json").write_text(json.dumps(dataset.classes, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

