import pytest
import torch
from landuse.metrics import collect_predictions, write_classification_artifacts

def test_collect_predictions(dummy_model):
    device = torch.device("cpu")
    
    # Create a dummy dataloader (batch of 4)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    loader = [(images, labels)]
    
    y_true, y_pred, confidences = collect_predictions(dummy_model, loader, device)
    
    assert len(y_true) == 4
    assert len(y_pred) == 4
    assert len(confidences) == 4
    assert y_true == [0, 1, 0, 1]
    # confidences should be between 0 and 1 (softmax output)
    assert all(0.0 <= c <= 1.0 for c in confidences)

def test_write_classification_artifacts(tmp_path):
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    classes = ["Forest", "Highway"]
    
    out_dir = tmp_path / "metrics"
    write_classification_artifacts(y_true, y_pred, classes, out_dir)
    
    assert (out_dir / "classification_report.csv").exists()
    assert (out_dir / "confusion_matrix.png").exists()
