from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from landuse.data import load_image_folder, make_loader, split_dataset
from landuse.models import build_resnet18, freeze_backbone, unfreeze_last_two_blocks
from landuse.train import fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=["block", "random"], default="block")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_image_folder(args.data_dir, train=True)
    splits = split_dataset(dataset, split=args.split)
    train_loader = make_loader(splits.train, args.batch_size, shuffle=True)
    val_loader = make_loader(splits.val, args.batch_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet18(num_classes=len(dataset.classes), pretrained=not args.no_pretrained).to(device)

    freeze_backbone(model)
    phase1_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    phase1_history = fit(model, train_loader, val_loader, phase1_opt, epochs=3, device=device, out_dir=out_dir / "phase1")

    unfreeze_last_two_blocks(model)
    phase2_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    phase2_history = fit(model, train_loader, val_loader, phase2_opt, epochs=5, device=device, out_dir=out_dir / "phase2")

    torch.save({"model_state": model.state_dict(), "classes": dataset.classes}, out_dir / "best.pt")
    (out_dir / "classes.json").write_text(json.dumps(dataset.classes, indent=2))
    (out_dir / "history.json").write_text(json.dumps({"phase1": phase1_history, "phase2": phase2_history}, indent=2))


if __name__ == "__main__":
    main()

