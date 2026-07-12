from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from landuse.data import load_image_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/data_profile")
    args = parser.parse_args()

    dataset = load_image_folder(args.data_dir, train=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.Series([dataset.classes[y] for _, y in dataset.samples]).value_counts().sort_index()
    counts.to_csv(out_dir / "class_distribution.csv", header=["count"])
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()

