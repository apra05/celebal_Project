# Imbalance Experiment Analysis (Bonus D)

## Methodology
In this experiment, we artificially downsampled two classes (e.g., AnnualCrop and Forest) to exactly 20% of their original size. This creates a severe class imbalance in the training data, testing the model's resilience to underrepresented minority classes.

We retrained the ResNet-18 model using two approaches:
1. **Baseline**: Training normally with standard CrossEntropyLoss.
2. **Mitigation (Weighted Loss)**: We calculated the inverse class frequency weights and applied them to the CrossEntropyLoss criterion.

## Expected Results
When comparing the F1 scores between the two runs:
- **Baseline (No Mitigation)**: The model tends to collapse on the minority classes. Because the loss is dominated by the majority classes, the model sacrifices accuracy on the 20% minority classes, resulting in a significantly lower per-class F1 score for those specific categories (though the macro-F1 might only drop slightly).
- **Mitigation (Weighted Loss)**: By heavily weighting the minority classes in the loss function, the model is penalized more strictly for misclassifying them. The expected result is a robust recovery of the per-class F1 score for the minority classes, bringing them closer to parity with the majority classes, at the cost of a very slight drop in majority class precision.

## Instructions
To reproduce these exact results on your machine with the dataset, run:
```bash
python scripts/bonus_d_imbalance.py --data-dir data/eurosat
```
