import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from landuse.change import cosine_similarity, pixel_change_heatmap, save_region_heatmaps
from landuse.models import ScratchCNN, build_resnet18


def test_models_forward():
    x = torch.randn(2, 3, 224, 224)
    assert ScratchCNN(num_classes=10)(x).shape == (2, 10)
    assert build_resnet18(num_classes=10, pretrained=False)(x).shape == (2, 10)


def test_change_helpers():
    a = np.ones(512)
    b = np.ones(512)
    assert cosine_similarity(a, b) > 0.99
    before = np.zeros((64, 64, 3), dtype=np.uint8)
    after = np.ones((64, 64, 3), dtype=np.uint8) * 255
    assert pixel_change_heatmap(before, after).shape == (224, 224, 3)


def test_save_region_heatmaps():
    """save_region_heatmaps should write PNG triptychs for each requested pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 4 tiny synthetic tile images
        img_paths = []
        for i in range(4):
            p = Path(tmpdir) / f"tile_{i}.png"
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(p)
            img_paths.append(str(p))

        out_dir = Path(tmpdir) / "heatmaps"
        classes = ["ClassA", "ClassB"]
        t1_labels = [0, 0, 1, 1]
        t2_labels = [1, 0, 1, 0]  # pairs 0 and 3 are "changed"

        save_region_heatmaps(
            t1_paths=img_paths,
            t2_paths=img_paths,
            t1_labels=t1_labels,
            t2_labels=t2_labels,
            classes=classes,
            out_dir=out_dir,
            n_pairs=3,
        )

        saved = list(out_dir.glob("heatmap_pair_*.png"))
        assert len(saved) == 3, f"Expected 3 PNGs, got {len(saved)}"

