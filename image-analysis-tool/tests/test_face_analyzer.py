import pytest
import numpy as np
from src.face_analysis.face_analyzer import FaceAnalyzer

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

@pytest.fixture
def analyzer():
    """Create a FaceAnalyzer instance."""
    return FaceAnalyzer()

def test_analyzer_initialization(analyzer):
    """Test if analyzer initializes correctly."""
    assert analyzer.face_mesh is not None
    assert analyzer.mp_drawing is not None

def test_analyze_method(analyzer, sample_image):
    """Test face analysis."""
    annotated_image, analyses = analyzer.analyze(sample_image)
    
    # Check output types
    assert isinstance(annotated_image, np.ndarray)
    assert isinstance(analyses, list)
    
    # Check image dimensions
    assert annotated_image.shape == sample_image.shape
    
    # Check analysis format
    for analysis in analyses:
        assert 'bbox' in analysis
        assert 'landmarks' in analysis
        assert 'facial_features' in analysis
        assert len(analysis['bbox']) == 4
        assert isinstance(analysis['landmarks'], list)
        assert isinstance(analysis['facial_features'], dict)
