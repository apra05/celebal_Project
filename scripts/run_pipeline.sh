#!/usr/bin/env bash
# run_pipeline.sh — Full end-to-end pipeline for the satellite land-use project.
# Run with: bash scripts/run_pipeline.sh
set -euo pipefail

VENV=".venv/bin/activate"
DATA_EUROSAT="data/eurosat"
DATA_UCMERCED="data/uc_merced"
CHECKPOINT="runs/resnet18/best.pt"

log() { echo -e "\n\033[1;36m>>> $1\033[0m"; }

source "$VENV"

# ── Step 1: Data profile ─────────────────────────────────────────────────────
log "Step 1/9 — Data profile"
python scripts/plot_dataset.py --data-dir "$DATA_EUROSAT" --out-dir runs/data_profile

# ── Step 2: Baseline scratch CNN ─────────────────────────────────────────────
log "Step 2/9 — Baseline scratch CNN (3 epochs)"
python scripts/train_baseline.py \
    --data-dir "$DATA_EUROSAT" \
    --out-dir runs/baseline \
    --epochs 3

# ── Step 3: Fine-tune ResNet-18 (two-phase) ───────────────────────────────────
log "Step 3/9 — Fine-tune ResNet-18 (Phase 1: freeze 3ep + Phase 2: unfreeze 5ep)"
python scripts/train_finetune.py \
    --data-dir "$DATA_EUROSAT" \
    --out-dir runs/resnet18

# ── Step 4: Evaluate on EuroSAT (block split) ────────────────────────────────
log "Step 4a/9 — Evaluate on EuroSAT (block split)"
python scripts/evaluate.py \
    --data-dir "$DATA_EUROSAT" \
    --checkpoint "$CHECKPOINT" \
    --out-dir runs/resnet18/eurosat_eval \
    --split block

log "Step 4b/9 — Evaluate on UC Merced holdout"
python scripts/evaluate.py \
    --data-dir "$DATA_UCMERCED" \
    --checkpoint "$CHECKPOINT" \
    --out-dir runs/resnet18/uc_merced_eval \
    --split all

# ── Step 5: Change detection ─────────────────────────────────────────────────
log "Step 5/9 — Change detection (ROC curve + threshold)"
python scripts/run_change_detection.py \
    --data-dir "$DATA_EUROSAT" \
    --checkpoint "$CHECKPOINT" \
    --out-dir runs/change_detection

# ── Step 6: Heatmaps for 5 region pairs ──────────────────────────────────────
log "Step 6/9 — Save visual heatmaps for 5 region pairs"
python scripts/save_change_heatmaps.py \
    --data-dir "$DATA_EUROSAT" \
    --checkpoint "$CHECKPOINT" \
    --out-dir runs/change_detection/heatmaps \
    --n-pairs 5

# ── Step 7: Spatial leakage experiment ───────────────────────────────────────
log "Step 7/9 — Spatial leakage experiment (block vs random)"
python scripts/spatial_leakage_experiment.py \
    --data-dir "$DATA_EUROSAT" \
    --checkpoint "$CHECKPOINT" \
    --out-dir runs/spatial_leakage

# ── Step 8: Error analysis ───────────────────────────────────────────────────
log "Step 8/9 — Visual error analysis (top-5 misclassified tiles)"
python scripts/visualize_errors.py \
    --data-dir "$DATA_EUROSAT" \
    --checkpoint "$CHECKPOINT" \
    --out-dir runs/resnet18/error_analysis

# ── Step 9: Generate PDF report ──────────────────────────────────────────────
log "Step 9/9 — Generate PDF report"
python scripts/generate_report.py \
    --runs-dir runs \
    --out-path reports/project_report.pdf

log "ALL STEPS COMPLETE!"
echo ""
echo "Key outputs:"
echo "  Checkpoint   : $CHECKPOINT"
echo "  EuroSAT eval : runs/resnet18/eurosat_eval/"
echo "  UC Merced    : runs/resnet18/uc_merced_eval/"
echo "  Change detect: runs/change_detection/"
echo "  Leakage      : runs/spatial_leakage/"
echo "  Error analysis: runs/resnet18/error_analysis/"
echo "  PDF report   : reports/project_report.pdf"
echo ""
echo "Launch dashboard with:"
echo "  streamlit run app/streamlit_app.py -- --checkpoint $CHECKPOINT"
