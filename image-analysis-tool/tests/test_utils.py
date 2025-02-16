import pytest
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from src.utils.image_utils import (
    load_image,
    draw_boxes,
    resize_image,
    apply_text_overlay
)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

def test_load_image_from_numpy(sample_image):
    """Test loading image from numpy array."""
    bgr, rgb = load_image(sample_image)
    assert isinstance(bgr, np.ndarray)
    assert isinstance(rgb, np.ndarray)
    assert bgr.shape == sample_image.shape
    assert rgb.shape == sample_image.shape

def test_load_image_from_pil(sample_image):
    """Test loading image from PIL Image."""
    pil_image = Image.fromarray(sample_image)
    bgr, rgb = load_image(pil_image)
    assert isinstance(bgr, np.ndarray)
    assert isinstance(rgb, np.ndarray)
    assert bgr.shape == sample_image.shape
    assert rgb.shape == sample_image.shape

def test_draw_boxes(sample_image):
    """Test drawing bounding boxes on image."""
    boxes = [[10, 10, 50, 50], [20, 20, 60, 60]]
    labels = ["Object1", "Object2"]
    scores = [0.9, 0.8]
    
    result = draw_boxes(sample_image, boxes, labels, scores)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape

def test_resize_image(sample_image):
    """Test image resizing functionality."""
    # Test with target size as tuple
    result = resize_image(sample_image, (50, 50))
    assert result.shape == (50, 50, 3)
    
    # Test with max dimension
    result = resize_image(sample_image, 50)
    assert max(result.shape[:2]) == 50

def test_apply_text_overlay(sample_image):
    """Test adding text overlay to image."""
    result = apply_text_overlay(
        sample_image,
        "Test Text",
        (10, 10)
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape
