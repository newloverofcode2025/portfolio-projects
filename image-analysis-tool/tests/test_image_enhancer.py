import pytest
import numpy as np
from src.enhancement.image_enhancer import ImageEnhancer

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def enhancer():
    """Create an ImageEnhancer instance."""
    return ImageEnhancer()

def test_enhancer_initialization(enhancer):
    """Test if enhancer initializes correctly."""
    assert enhancer.clahe is not None

def test_enhance_method(enhancer, sample_image):
    """Test image enhancement."""
    enhanced = enhancer.enhance(sample_image)
    
    # Check output type and shape
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == sample_image.shape
    
    # Test individual enhancements
    contrast_enhanced = enhancer.enhance(sample_image, enhance_contrast=True,
                                       reduce_noise=False, sharpen=False)
    denoised = enhancer.enhance(sample_image, enhance_contrast=False,
                               reduce_noise=True, sharpen=False)
    sharpened = enhancer.enhance(sample_image, enhance_contrast=False,
                                reduce_noise=False, sharpen=True)
    
    assert all(isinstance(img, np.ndarray) for img in
              [contrast_enhanced, denoised, sharpened])

def test_adjust_brightness_contrast(enhancer, sample_image):
    """Test brightness and contrast adjustment."""
    adjusted = enhancer.adjust_brightness_contrast(
        sample_image,
        brightness=1.2,
        contrast=1.1
    )
    
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == sample_image.shape
    assert adjusted.dtype == np.uint8

def test_remove_background(enhancer, sample_image):
    """Test background removal."""
    no_bg, mask = enhancer.remove_background(sample_image)
    
    # Check output types and shapes
    assert isinstance(no_bg, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert no_bg.shape[:2] == sample_image.shape[:2]
    assert mask.shape == sample_image.shape[:2]
    
    # Check if alpha channel was added
    assert no_bg.shape[2] == 4  # BGRA
    assert mask.dtype == np.uint8
