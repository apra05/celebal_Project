from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        total += criterion(model(images), labels).item() * images.size(0)
    return total / len(loader.dataset)


def fit(model, train_loader, val_loader, optimizer, epochs: int, device, out_dir: str | Path) -> list[dict]:
    criterion = nn.CrossEntropyLoss()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "history": history}, out_dir / "best.pt")
        print(row)
    return history

