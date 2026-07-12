from __future__ import annotations

import argparse
import json
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
from PIL import Image

from landuse.change import cosine_similarity, pixel_change_heatmap
from landuse.config import EUROSAT_CLASSES
from landuse.data import build_transforms
from landuse.models import ResNetEmbeddingExtractor, build_resnet18, load_checkpoint

ACCEPTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]

# --- Custom CSS for Styling ---
def local_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 14px;
        color: #A0A0A0;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        color: #FFFFFF;
        font-weight: bold;
    }
    .similarity-high { color: #4CAF50; }
    .similarity-med { color: #FF9800; }
    .similarity-low { color: #F44336; }
    .decision-text {
        font-size: 16px;
        font-style: italic;
        color: #CCCCCC;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Checkpoint Handling & Setup ---
@st.cache_resource
def load_model(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(checkpoint, map_location=device)
    classes = raw.get("classes", EUROSAT_CLASSES) if isinstance(raw, dict) else EUROSAT_CLASSES
    model = load_checkpoint(checkpoint, num_classes=len(classes), device=device)
    model.eval()
    extractor = ResNetEmbeddingExtractor(model).to(device).eval()
    return model, extractor, classes, device

def preprocess(image: Image.Image):
    transform = build_transforms(train=False)
    return transform(image.convert("RGB")).unsqueeze(0)

@torch.no_grad()
def predict_top3(model, image, classes, device):
    logits = model(preprocess(image).to(device))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_probs, top_idxs = probs.topk(3)
    return [(classes[int(idx)], float(prob)) for idx, prob in zip(top_idxs, top_probs)]

@torch.no_grad()
def get_embedding(extractor, image, device):
    emb = extractor(preprocess(image).to(device))
    emb = torch.nn.functional.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]

def open_uploaded_image(uploaded_file, label: str) -> Image.Image | None:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(f"{label} could not be opened. Please upload a valid image.")
        return None

# --- Main App ---
def main() -> None:
    st.set_page_config(page_title="Satellite Land-Use Change Detector", layout="wide", page_icon="🛰️")
    local_css()

    checkpoint_path = Path("runs/resnet18/best.pt")
    if not checkpoint_path.exists():
        st.error("🚨 **Model checkpoint not found.** Please train the model before launching the dashboard.")
        st.stop()

    # Layout Header
    st.title("🛰️ Satellite Land-Use Classifier & Temporal Change Detector")
    st.markdown("Compare two satellite images using deep learning, classify land-use, generate feature embeddings, detect temporal land-cover changes, and visualize embedding differences.")
    st.divider()

    # Load Model
    model, extractor, classes, device = load_model(str(checkpoint_path))

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        show_top5 = st.toggle("Show Top-3 Predictions", value=True)
        show_conf_bars = st.toggle("Display Confidence Bars", value=True)
        
        st.divider()
        st.header("📊 Model Information")
        st.markdown("**Backbone:** ResNet18")
        st.markdown("**Framework:** PyTorch")
        st.markdown("**Dataset:** EuroSAT (10 Classes)")
        st.markdown("**Evaluation Dataset:** UC Merced")
        st.markdown("**Embedding Size:** 512")
        st.markdown(f"**Loaded Checkpoint:** `{checkpoint_path.name}`")

        thresholds_path = Path("configs/thresholds.json")
        thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {"balanced": 0.74}
        operating_point = st.selectbox("Operating point preset", list(thresholds.keys()), index=min(1, len(thresholds) - 1))
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, float(thresholds[operating_point]))
        st.markdown(f"**Current Threshold:** {threshold}")

        st.divider()
        st.header("📈 Training Strategy")
        st.markdown("**Phase 1:** Frozen Backbone, 3 Epochs")
        st.markdown("**Phase 2:** Last Two Residual Blocks Unfrozen, LR reduced by 10×, 5 Epochs")

        st.divider()
        st.header("💻 System Information")
        st.markdown(f"**Python:** {platform.python_version()}")
        st.markdown(f"**Torch:** {torch.__version__}")
        st.markdown(f"**Torchvision:** {torchvision.__version__}")
        st.markdown(f"**CUDA:** {'Available' if torch.cuda.is_available() else 'Not Available'}")
        st.markdown(f"**Inference Device:** {device.type.upper()}")

    # Image Upload Section
    st.header("📸 Image Upload")
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Upload Before Tile", type=ACCEPTED_IMAGE_TYPES)
    with col2:
        after_file = st.file_uploader("Upload After Tile", type=ACCEPTED_IMAGE_TYPES)

    if before_file and after_file:
        before = open_uploaded_image(before_file, "Before tile")
        after = open_uploaded_image(after_file, "After tile")

        if before and after:
            # Display Images
            st.markdown("### Preview")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(before, caption=f"File: {before_file.name} | Res: {before.size[0]}x{before.size[1]}", use_container_width=True)
            with img_col2:
                st.image(after, caption=f"File: {after_file.name} | Res: {after.size[0]}x{after.size[1]}", use_container_width=True)
            
            st.divider()

            # Prediction & Timing
            t0 = time.perf_counter()
            before_preds = predict_top3(model, before, classes, device)
            after_preds = predict_top3(model, after, classes, device)
            t_pred = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            emb_before = get_embedding(extractor, before, device)
            emb_after = get_embedding(extractor, after, device)
            t_emb = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            similarity = cosine_similarity(emb_before, emb_after)
            t_sim = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            heatmap = pixel_change_heatmap(np.array(before), np.array(after))
            t_heat = (time.perf_counter() - t0) * 1000

            total_time = t_pred + t_emb + t_sim + t_heat

            # Prediction Cards
            st.header("🎯 Predictions")
            pred_col1, pred_col2 = st.columns(2)
            
            for col, preds, label in zip([pred_col1, pred_col2], [before_preds, after_preds], ["Before", "After"]):
                with col:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-title">{label} Image Analysis</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">🏷️ {preds[0][0]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<span style="color:#A0A0A0">Confidence: </span> <b>{preds[0][1]:.1%}</b>', unsafe_allow_html=True)
                    if show_conf_bars:
                        st.progress(preds[0][1])
                    
                    if show_top5:
                        st.markdown("<br><b>Top 3 Predictions:</b>", unsafe_allow_html=True)
                        for rank, (cls_name, prob) in enumerate(preds):
                            st.write(f"{rank+1}. {cls_name} ({prob:.1%})")
                            st.progress(prob)
                    st.markdown(f'</div>', unsafe_allow_html=True)

            # Similarity & Decision
            st.divider()
            st.header("⚖️ Change Decision")
            changed = similarity < threshold
            sim_col, dec_col = st.columns(2)
            
            with sim_col:
                sim_color = "similarity-high" if similarity > 0.8 else "similarity-med" if similarity > 0.5 else "similarity-low"
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">Cosine Similarity</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {sim_color}">{similarity:.1%}</div>', unsafe_allow_html=True)
                st.progress(max(0.0, min(1.0, similarity)))
                st.markdown(f'</div>', unsafe_allow_html=True)
            
            with dec_col:
                dec_icon = "⚠ CHANGED" if changed else "✓ UNCHANGED"
                dec_color = "#F44336" if changed else "#4CAF50"
                st.markdown(f'<div class="metric-card" style="border-left: 5px solid {dec_color};">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">Threshold Comparison ({similarity:.3f} {"<" if changed else ">="} {threshold:.3f})</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color:{dec_color}">{dec_icon}</div>', unsafe_allow_html=True)
                explanation = "The similarity score falls below the operating threshold, indicating significant land-cover change." if changed else "The similarity score exceeds the operating threshold, indicating no significant land-cover change."
                st.markdown(f'<div class="decision-text">{explanation}</div>', unsafe_allow_html=True)
                st.markdown(f'</div>', unsafe_allow_html=True)

            # Heatmap
            st.divider()
            st.header("🗺️ Feature Heatmap")
            h_col1, h_col2 = st.columns([1, 2])
            with h_col1:
                mean_diff = np.mean(np.abs(emb_before - emb_after))
                max_diff = np.max(np.abs(emb_before - emb_after))
                st.metric("Mean Embedding Diff", f"{mean_diff:.4f}")
                st.metric("Max Embedding Diff", f"{max_diff:.4f}")
                st.markdown("**Interpretation:** Brighter regions in the heatmap indicate larger pixel-level differences, corresponding to potential land-cover changes.")
            with h_col2:
                st.image(heatmap, caption="Embedding Difference Heatmap", use_container_width=True)
            
            # Analytics Expanders
            st.divider()
            with st.expander("⚡ Performance Metrics"):
                st.write(f"- **Prediction Time:** {t_pred:.1f} ms")
                st.write(f"- **Embedding Time:** {t_emb:.1f} ms")
                st.write(f"- **Similarity Time:** {t_sim:.1f} ms")
                st.write(f"- **Heatmap Time:** {t_heat:.1f} ms")
                st.write(f"- **Total Inference Time:** {total_time:.1f} ms")

            with st.expander("📊 Dataset Information"):
                st.markdown("""
                **EuroSAT** (Transfer Learning Dataset)
                - 27,000 Images
                - 10 Classes
                
                **UC Merced** (External Evaluation Dataset)
                - 2,100 Images
                - 21 Classes
                """)

            # Load evaluation results if available
            with st.expander("📈 Model Performance & Training Visualizations"):
                block_report_path = Path("runs/spatial_leakage/block_report.json")
                if block_report_path.exists():
                    report = json.loads(block_report_path.read_text())
                    macro = report.get("macro avg", {})
                    m_cols = st.columns(4)
                    m_cols[0].metric("Macro F1", f"{macro.get('f1-score', 0):.3f}")
                    m_cols[1].metric("Precision", f"{macro.get('precision', 0):.3f}")
                    m_cols[2].metric("Recall", f"{macro.get('recall', 0):.3f}")
                    m_cols[3].metric("Accuracy", f"{report.get('accuracy', 0):.3f}")
                
                # Show images if present
                cm_img = Path("runs/resnet18/eurosat_eval/confusion_matrix.png")
                roc_img = Path("runs/change_detection/roc_curve.png")
                loss_img = Path("runs/data_profile/class_distribution.png") 
                
                img_tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Class Distribution"])
                with img_tabs[0]:
                    if cm_img.exists(): st.image(Image.open(cm_img), use_container_width=True)
                    else: st.info("Confusion matrix not found.")
                with img_tabs[1]:
                    if roc_img.exists(): st.image(Image.open(roc_img), use_container_width=True)
                    else: st.info("ROC curve not found.")
                with img_tabs[2]:
                    if loss_img.exists(): st.image(Image.open(loss_img), use_container_width=True)
                    else: st.info("Class distribution not found.")
                
if __name__ == "__main__":
    main()
