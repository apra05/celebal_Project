import pytest
from torchvision.datasets import ImageFolder
from landuse.data import build_transforms, load_image_folder, split_dataset

def test_build_transforms(dummy_image):
    tf_train = build_transforms(train=True)
    tf_eval = build_transforms(train=False)
    
    # Train transforms should have data augmentation (more operations)
    assert len(tf_train.transforms) > len(tf_eval.transforms)
    
    # Check output shape on a dummy image
    tensor_img = tf_eval(dummy_image)
    assert tensor_img.shape == (3, 224, 224)

def test_load_image_folder(dummy_data_dir):
    dataset = load_image_folder(dummy_data_dir, image_size=224, train=False)
    
    assert isinstance(dataset, ImageFolder)
    # 2 classes * 10 images
    assert len(dataset) == 20
    assert len(dataset.classes) == 2
    assert dataset.classes == ["Forest", "Highway"]

def test_split_dataset(dummy_data_dir):
    dataset = load_image_folder(dummy_data_dir)
    
    # Test random split
    splits_rand = split_dataset(dataset, split="random", train_ratio=0.7, val_ratio=0.15)
    total = len(dataset)
    
    assert len(splits_rand.train) + len(splits_rand.val) + len(splits_rand.test) == total
    assert len(splits_rand.train) == int(total * 0.7)
    assert len(splits_rand.val) == int(total * 0.15)
    
    # Test block split
    splits_block = split_dataset(dataset, split="block")
    assert len(splits_block.train) + len(splits_block.val) + len(splits_block.test) == total
