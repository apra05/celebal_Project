from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from landuse.change import cosine_similarity, pixel_change_heatmap
from landuse.config import EUROSAT_CLASSES
from landuse.data import build_transforms
from landuse.models import ResNetEmbeddingExtractor, build_resnet18, load_checkpoint

ACCEPTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--thresholds", default="configs/thresholds.json")
    return parser.parse_known_args()[0]


@st.cache_resource
def load_model(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint and Path(checkpoint).exists():
        raw = torch.load(checkpoint, map_location=device)
        classes = raw.get("classes", EUROSAT_CLASSES) if isinstance(raw, dict) else EUROSAT_CLASSES
        model = load_checkpoint(checkpoint, num_classes=len(classes), device=device)
    else:
        classes = EUROSAT_CLASSES
        model = build_resnet18(num_classes=len(classes), pretrained=False).to(device)
    model.eval()
    return model, ResNetEmbeddingExtractor(model).to(device).eval(), classes, device


def preprocess(image: Image.Image):
    transform = build_transforms(train=False)
    return transform(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(model, image, classes, device):
    logits = model(preprocess(image).to(device))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    confidence, idx = probs.max(dim=0)
    return classes[int(idx)], float(confidence)


@torch.no_grad()
def embedding(extractor, image, device):
    emb = extractor(preprocess(image).to(device))
    emb = torch.nn.functional.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]


def open_uploaded_image(uploaded_file, label: str) -> Image.Image | None:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(f"{label} could not be opened. Please upload a valid JPG, JPEG, PNG, TIF, or TIFF image.")
        return None


def main() -> None:
    st.set_page_config(page_title="Land-Use Change Detector", layout="wide")
    st.title("Satellite Land-Use Change Detector")

    # Hardcoding paths to avoid arg parsing issues across Streamlit reruns
    checkpoint_path = "runs/resnet18/best.pt"
    thresholds_path = "configs/thresholds.json"
    
    thresholds = json.loads(Path(thresholds_path).read_text()) if Path(thresholds_path).exists() else {"balanced": 0.74}
    operating_point = st.sidebar.selectbox("Operating point", list(thresholds.keys()), index=min(1, len(thresholds) - 1))
    threshold = thresholds[operating_point]

    model, extractor, classes, device = load_model(checkpoint_path)
    if not Path(checkpoint_path).exists():
        st.sidebar.warning("No checkpoint found at `runs/resnet18/best.pt`. Running with random weights.")

    st.caption("Drag and drop satellite tile images into the boxes below, or click Browse files.")
    
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Before tile", type=ACCEPTED_IMAGE_TYPES)
    with col2:
        after_file = st.file_uploader("After tile", type=ACCEPTED_IMAGE_TYPES)

    if before_file and after_file:
        before = open_uploaded_image(before_file, "Before tile")
        after = open_uploaded_image(after_file, "After tile")
        
        if before is not None and after is not None:
            st.divider()
            before_class, before_conf = predict(model, before, classes, device)
            after_class, after_conf = predict(model, after, classes, device)
            similarity = cosine_similarity(embedding(extractor, before, device), embedding(extractor, after, device))
            changed = similarity < threshold

            metric_cols = st.columns(4)
            metric_cols[0].metric("Before class", before_class, f"{before_conf:.1%}")
            metric_cols[1].metric("After class", after_class, f"{after_conf:.1%}")
            metric_cols[2].metric("Cosine similarity", f"{similarity:.3f}")
            metric_cols[3].metric("Change flag", "Changed" if changed else "Unchanged", f"threshold {threshold:.2f}")

            image_cols = st.columns(3)
            image_cols[0].image(before, caption="Before", use_container_width=True)
            image_cols[1].image(after, caption="After", use_container_width=True)
            heatmap = pixel_change_heatmap(np.array(before), np.array(after))
            image_cols[2].image(heatmap, caption="Change heatmap", use_container_width=True)
    else:
        st.info("Please upload both a 'Before' and 'After' tile image to see the comparison.")

if __name__ == "__main__":
    main()
