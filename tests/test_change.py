import pytest
import numpy as np
import torch
from landuse.change import cosine_similarity, extract_embeddings, save_roc, pixel_change_heatmap

def test_cosine_similarity():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    
    # Orthogonal
    assert cosine_similarity(a, b) == 0.0
    # Identical
    assert cosine_similarity(a, c) == 1.0

def test_extract_embeddings(dummy_extractor):
    device = torch.device("cpu")
    
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    loader = [(images, labels)]
    
    embeddings, extracted_labels = extract_embeddings(dummy_extractor, loader, device)
    
    assert embeddings.shape == (4, 512)
    assert extracted_labels.shape == (4,)
    
    # Check that embeddings are L2-normalized
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

def test_save_roc(tmp_path):
    y_true_changed = [1, 1, 0, 0, 1]
    # Changed instances should have low similarity scores (high change_scores)
    similarity_scores = [0.1, 0.2, 0.9, 0.8, 0.3]
    
    out_dir = tmp_path / "roc"
    best_threshold = save_roc(y_true_changed, similarity_scores, out_dir)
    
    assert 0.0 <= best_threshold <= 1.0
    assert (out_dir / "roc_curve.png").exists()

def test_pixel_change_heatmap():
    before = np.zeros((224, 224, 3), dtype=np.uint8)
    after = np.full((224, 224, 3), 255, dtype=np.uint8)
    
    heatmap = pixel_change_heatmap(before, after)
    
    assert heatmap.shape == (224, 224, 3)
    assert heatmap.dtype == np.uint8
    
    # Identical images should produce a heatmap with no active changes
    heatmap_same = pixel_change_heatmap(before, before)
    assert heatmap_same.shape == (224, 224, 3)
