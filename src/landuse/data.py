from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class Split:
    train: Subset
    val: Subset
    test: Subset


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def load_image_folder(data_dir: str | Path, image_size: int = 224, train: bool = False) -> datasets.ImageFolder:
    return datasets.ImageFolder(str(data_dir), transform=build_transforms(image_size=image_size, train=train))


def _block_id(path: str, blocks: int = 20) -> int:
    digest = hashlib.sha1(Path(path).stem.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % blocks


def split_dataset(
    dataset: datasets.ImageFolder,
    split: str = "block",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Split:
    indices = list(range(len(dataset)))
    if split == "random":
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[i] for i in perm]
    elif split == "block":
        indices.sort(key=lambda i: (_block_id(dataset.samples[i][0]), dataset.samples[i][0]))
    else:
        raise ValueError("split must be 'block' or 'random'")

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)
    return Split(
        train=Subset(dataset, indices[:train_end]),
        val=Subset(dataset, indices[train_end:val_end]),
        test=Subset(dataset, indices[val_end:]),
    )


def make_loader(dataset, batch_size: int = 32, shuffle: bool = False, workers: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)

