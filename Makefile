.PHONY: all data-profile train-baseline train-finetune eval-eurosat eval-ucmerced change-detection heatmaps leakage error-analysis report dashboard

DATA_EUROSAT=data/eurosat
DATA_UCMERCED=data/uc_merced
CHECKPOINT=runs/resnet18/best.pt

all: data-profile train-baseline train-finetune eval-eurosat eval-ucmerced change-detection heatmaps leakage error-analysis report
	@echo "ALL STEPS COMPLETE!"

data-profile:
	python scripts/plot_dataset.py --data-dir $(DATA_EUROSAT) --out-dir runs/data_profile

train-baseline:
	python scripts/train_baseline.py --data-dir $(DATA_EUROSAT) --out-dir runs/baseline --epochs 3

train-finetune:
	python scripts/train_finetune.py --data-dir $(DATA_EUROSAT) --out-dir runs/resnet18

eval-eurosat:
	python scripts/evaluate.py --data-dir $(DATA_EUROSAT) --checkpoint $(CHECKPOINT) --out-dir runs/resnet18/eurosat_eval --split block

eval-ucmerced:
	python scripts/evaluate.py --data-dir $(DATA_UCMERCED) --checkpoint $(CHECKPOINT) --out-dir runs/resnet18/uc_merced_eval --split all

change-detection:
	python scripts/run_change_detection.py --data-dir $(DATA_EUROSAT) --checkpoint $(CHECKPOINT) --out-dir runs/change_detection

heatmaps:
	python scripts/save_change_heatmaps.py --data-dir $(DATA_EUROSAT) --checkpoint $(CHECKPOINT) --out-dir runs/change_detection/heatmaps --n-pairs 5

leakage:
	python scripts/spatial_leakage_experiment.py --data-dir $(DATA_EUROSAT) --checkpoint $(CHECKPOINT) --out-dir runs/spatial_leakage

error-analysis:
	python scripts/visualize_errors.py --data-dir $(DATA_EUROSAT) --checkpoint $(CHECKPOINT) --out-dir runs/resnet18/error_analysis

report:
	python scripts/generate_report.py --runs-dir runs --out-path reports/project_report.pdf

dashboard:
	streamlit run app/streamlit_app.py -- --checkpoint $(CHECKPOINT)
