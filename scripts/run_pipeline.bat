@echo off
setlocal

set DATA_EUROSAT=data\eurosat
set DATA_UCMERCED=data\uc_merced
set CHECKPOINT=runs\resnet18\best.pt

echo ========================================
echo Step 1/9 - Data profile
echo ========================================
python scripts\plot_dataset.py --data-dir %DATA_EUROSAT% --out-dir runs\data_profile

echo ========================================
echo Step 2/9 - Baseline scratch CNN (3 epochs)
echo ========================================
python scripts\train_baseline.py --data-dir %DATA_EUROSAT% --out-dir runs\baseline --epochs 3

echo ========================================
echo Step 3/9 - Fine-tune ResNet-18
echo ========================================
python scripts\train_finetune.py --data-dir %DATA_EUROSAT% --out-dir runs\resnet18

echo ========================================
echo Step 4a/9 - Evaluate on EuroSAT (block split)
echo ========================================
python scripts\evaluate.py --data-dir %DATA_EUROSAT% --checkpoint %CHECKPOINT% --out-dir runs\resnet18\eurosat_eval --split block

echo ========================================
echo Step 4b/9 - Evaluate on UC Merced holdout
echo ========================================
python scripts\evaluate.py --data-dir %DATA_UCMERCED% --checkpoint %CHECKPOINT% --out-dir runs\resnet18\uc_merced_eval --split all

echo ========================================
echo Step 5/9 - Change detection
echo ========================================
python scripts\run_change_detection.py --data-dir %DATA_EUROSAT% --checkpoint %CHECKPOINT% --out-dir runs\change_detection

echo ========================================
echo Step 6/9 - Save visual heatmaps
echo ========================================
python scripts\save_change_heatmaps.py --data-dir %DATA_EUROSAT% --checkpoint %CHECKPOINT% --out-dir runs\change_detection\heatmaps --n-pairs 5

echo ========================================
echo Step 7/9 - Spatial leakage experiment
echo ========================================
python scripts\spatial_leakage_experiment.py --data-dir %DATA_EUROSAT% --checkpoint %CHECKPOINT% --out-dir runs\spatial_leakage

echo ========================================
echo Step 8/9 - Visual error analysis
echo ========================================
python scripts\visualize_errors.py --data-dir %DATA_EUROSAT% --checkpoint %CHECKPOINT% --out-dir runs\resnet18\error_analysis

echo ========================================
echo Step 9/9 - Generate PDF report
echo ========================================
python scripts\generate_report.py --runs-dir runs --out-path reports\project_report.pdf

echo.
echo ALL STEPS COMPLETE!
echo.
echo Key outputs:
echo   Checkpoint   : %CHECKPOINT%
echo   EuroSAT eval : runs\resnet18\eurosat_eval\
echo   UC Merced    : runs\resnet18\uc_merced_eval\
echo   Change detect: runs\change_detection\
echo   Leakage      : runs\spatial_leakage\
echo   Error analysis: runs\resnet18\error_analysis\
echo   PDF report   : reports\project_report.pdf
echo.
echo Launch dashboard with:
echo   streamlit run app\streamlit_app.py -- --checkpoint %CHECKPOINT%
