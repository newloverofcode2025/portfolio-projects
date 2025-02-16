import pytest
import numpy as np
from src.detection.object_detector import ObjectDetector

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

@pytest.fixture
def detector():
    """Create an ObjectDetector instance."""
    return ObjectDetector()

def test_detector_initialization(detector):
    """Test if detector initializes correctly."""
    assert detector.model is not None
    assert detector.class_names is not None

def test_detect_method(detector, sample_image):
    """Test object detection."""
    annotated_image, detections = detector.detect(sample_image)
    
    # Check output types
    assert isinstance(annotated_image, np.ndarray)
    assert isinstance(detections, list)
    
    # Check image dimensions
    assert annotated_image.shape[:2] == sample_image.shape[:2]
    
    # Check detection format
    for detection in detections:
        assert 'bbox' in detection
        assert 'class' in detection
        assert 'confidence' in detection
        assert len(detection['bbox']) == 4
        assert isinstance(detection['class'], str)
        assert 0 <= detection['confidence'] <= 1
