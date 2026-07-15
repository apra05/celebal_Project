import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.streamlit_app import preprocess, predict_top3, get_embedding

def test_preprocess(dummy_image):
    tensor_img = preprocess(dummy_image)
    
    # Preprocess adds a batch dimension, so it should be (1, 3, 224, 224)
    assert tensor_img.shape == (1, 3, 224, 224)
    assert isinstance(tensor_img, torch.Tensor)

def test_predict_top3(dummy_model, dummy_image):
    device = torch.device("cpu")
    classes = [f"Class_{i}" for i in range(10)]
    
    top3 = predict_top3(dummy_model, dummy_image, classes, device)
    
    assert len(top3) == 3
    # Ensure it returns tuples of (class_name, probability)
    for class_name, prob in top3:
        assert isinstance(class_name, str)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

def test_get_embedding(dummy_extractor, dummy_image):
    device = torch.device("cpu")
    
    emb = get_embedding(dummy_extractor, dummy_image, device)
    
    # Check shape
    assert emb.shape == (512,)
    
    # Check L2 norm is 1
    norm = (emb ** 2).sum() ** 0.5
    assert pytest.approx(norm, 1e-5) == 1.0
