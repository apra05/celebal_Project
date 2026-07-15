import pytest
import torch
import torch.nn as nn
from PIL import Image

@pytest.fixture
def dummy_image():
    """Return a plain red 224x224 RGB Image."""
    return Image.new("RGB", (224, 224), color="red")

@pytest.fixture
def dummy_data_dir(tmp_path, dummy_image):
    """Create a temporary directory simulating an ImageFolder with 2 classes."""
    data_dir = tmp_path / "dummy_data"
    
    classes = ["Forest", "Highway"]
    for c in classes:
        class_dir = data_dir / c
        class_dir.mkdir(parents=True, exist_ok=True)
        # Create 10 images per class so splitting works better
        for i in range(10):
            dummy_image.save(class_dir / f"{c}_{i}.jpg")
            
    return data_dir

class DummyModel(nn.Module):
    """A minimal mock model that outputs logits matching num_classes."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # Use an adaptive pool so it works on any input shape
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_classes)
        
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DummyExtractor(nn.Module):
    """A minimal mock model that outputs embeddings."""
    def __init__(self, emb_size=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, emb_size)
        
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def dummy_extractor():
    return DummyExtractor()
