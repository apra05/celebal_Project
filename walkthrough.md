# Walkthrough — Satellite Land-Use Classifier & Temporal Change Detector

We have successfully addressed all identified requirements gaps, set up datasets (EuroSAT & UC Merced), executed baseline and fine-tuning training, and compiled all results into the final report.

## Changes Implemented

1. **Change Detection Heatmaps**:
   * Appended `save_region_heatmaps()` to [change.py](file:///Users/admin/Desktop/project_1/src/landuse/change.py).
   * Created [save_change_heatmaps.py](file:///Users/admin/Desktop/project_1/scripts/save_change_heatmaps.py) which extracts 5 region pairs (changed/unchanged mix) and dumps before/after/heatmap triptychs.
2. **Spatial Leakage Experiment**:
   * Created [spatial_leakage_experiment.py](file:///Users/admin/Desktop/project_1/scripts/spatial_leakage_experiment.py) to compare block-split vs random-split validation strategies and write [spatial_leakage_writeup.md](file:///Users/admin/Desktop/project_1/runs/spatial_leakage/spatial_leakage_writeup.md).
3. **Visual Error Analysis**:
   * Created [visualize_errors.py](file:///Users/admin/Desktop/project_1/scripts/visualize_errors.py) to draw high-confidence misclassifications as an annotated image grid, labeled with failure mode hypotheses.
4. **PDF Report Compilation**:
   * Created [generate_report.py](file:///Users/admin/Desktop/project_1/scripts/generate_report.py) to compile data distributions, loss curves, confusion matrices, change detector ROC, region heatmaps, spatial leakage comparison, and error analysis into a single document: [project_report.pdf](file:///Users/admin/Desktop/project_1/reports/project_report.pdf).
5. **Notebooks**:
   * Created [01_data_exploration.ipynb](file:///Users/admin/Desktop/project_1/notebooks/01_data_exploration.ipynb) and [02_results_and_analysis.ipynb](file:///Users/admin/Desktop/project_1/notebooks/02_results_and_analysis.ipynb) to ensure reproducibility.
6. **Documentation**:
   * Updated the project [README.md](file:///Users/admin/Desktop/project_1/README.md) with comprehensive installation instructions, run commands, notebook details, and bonus task documentation.

---

## Verification Results

### 🧪 Automated Tests
Running `pytest tests/ -v` successfully passes 15 tests across the new comprehensive test suite covering:
* `test_data.py` (Transformations, dataset loader, and splits)
* `test_metrics.py` (Classification reports and confusion matrices)
* `test_change.py` (Embeddings, cosine similarity, ROC calculations, and heatmaps)
* `test_dashboard.py` (Streamlit utilities and preprocessing)
* `test_smoke.py` (Basic model forward passes and helpers)

### 📈 Training & Validation Metrics
* **Baseline CNN** (3 epochs):
  * Final Val Loss: `0.7273`
* **Fine-Tuned ResNet-18**:
  * Phase 1 (Frozen Backbone - 3 epochs): Final Val Loss: `0.3007`
  * Phase 2 (Unfrozen conv blocks - 5 epochs): Final Val Loss: `0.1032` (on Epoch 2)

### 🛡️ Spatial Leakage Results
* **Block Split (Geographic hold-out)**: Macro-F1 = `0.9771`
* **Random Split**: Macro-F1 = `0.9852`
* **Delta (Random - Block)**: `+0.0081` (Confirms spatial correlation slightly inflates random split metrics)

### 📄 Final PDF Report Pages
Compiled successfully into [`reports/project_report.pdf`](file:///Users/admin/Desktop/project_1/reports/project_report.pdf):
1. **Title & Table of Contents**
2. **Dataset & Data Profile** (EuroSAT distributions)
3. **Training Loss Curves** (Baseline vs Fine-tuned phases)
4. **EuroSAT Classification Metrics Table**
5. **EuroSAT & UC Merced Confusion Matrices**
6. **ROC Curve (Change Detection)**
7. **Simulation Heatmap Pairs** (triptych grids)
8. **Spatial Leakage Analysis**
9. **Visual Error Grid**
10. **Conclusions & Limitations**

---

## 📹 Interactive Demo
The running dashboard was manually recorded to demonstrate all functionality:

![Dashboard Demo Video](demo_video/Dashboard_Demo_Real_Images.webp)
