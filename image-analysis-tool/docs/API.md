# AI Image Analysis Tool API Documentation

## Table of Contents
1. [Object Detection](#object-detection)
2. [Face Analysis](#face-analysis)
3. [Image Enhancement](#image-enhancement)
4. [Batch Processing](#batch-processing)
5. [Utility Functions](#utility-functions)

## Object Detection

### ObjectDetector Class
```python
from src.detection import ObjectDetector

detector = ObjectDetector(model_name="yolov8n.pt")
```

#### Methods

##### detect()
```python
def detect(
    image: np.ndarray,
    conf_threshold: float = 0.25,
    max_size: int = 1280
) -> Tuple[np.ndarray, List[dict]]
```
Detects objects in the input image.

**Parameters:**
- `image`: Input image as numpy array (BGR format)
- `conf_threshold`: Confidence threshold for detections (0-1)
- `max_size`: Maximum image dimension for processing

**Returns:**
- Tuple containing:
  - Annotated image with detections
  - List of detection dictionaries with 'bbox', 'class', and 'confidence'

## Face Analysis

### FaceAnalyzer Class
```python
from src.face_analysis import FaceAnalyzer

analyzer = FaceAnalyzer()
```

#### Methods

##### analyze()
```python
def analyze(
    image: np.ndarray,
    draw_landmarks: bool = True,
    detect_facial_features: bool = True
) -> Tuple[np.ndarray, List[Dict]]
```
Analyzes faces in the input image.

**Parameters:**
- `image`: Input image as numpy array (BGR format)
- `draw_landmarks`: Whether to draw facial landmarks
- `detect_facial_features`: Whether to detect facial features

**Returns:**
- Tuple containing:
  - Annotated image with face analysis
  - List of face analysis dictionaries with 'bbox', 'landmarks', and 'facial_features'

## Image Enhancement

### ImageEnhancer Class
```python
from src.enhancement import ImageEnhancer

enhancer = ImageEnhancer()
```

#### Methods

##### enhance()
```python
def enhance(
    image: np.ndarray,
    enhance_contrast: bool = True,
    reduce_noise: bool = True,
    sharpen: bool = True
) -> np.ndarray
```
Enhances the input image.

**Parameters:**
- `image`: Input image as numpy array (BGR format)
- `enhance_contrast`: Whether to enhance contrast
- `reduce_noise`: Whether to reduce noise
- `sharpen`: Whether to sharpen the image

**Returns:**
- Enhanced image as numpy array

##### adjust_brightness_contrast()
```python
def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> np.ndarray
```
Adjusts image brightness and contrast.

**Parameters:**
- `image`: Input image
- `brightness`: Brightness factor (1.0 = original)
- `contrast`: Contrast factor (1.0 = original)

**Returns:**
- Adjusted image

##### remove_background()
```python
def remove_background(
    image: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]
```
Removes image background.

**Parameters:**
- `image`: Input image
- `threshold`: Threshold for background removal

**Returns:**
- Tuple containing:
  - Image with transparent background (BGRA)
  - Binary mask of foreground

## Batch Processing

### BatchProcessor Class
```python
from src.utils.batch_processor import BatchProcessor

processor = BatchProcessor(max_workers=4, show_progress=True)
```

#### Methods

##### process_batch()
```python
def process_batch(
    items: List[Any],
    process_fn: Callable,
    save_results: bool = True,
    output_dir: Union[str, Path] = None,
    **kwargs
) -> Dict[str, Any]
```
Processes multiple items in parallel.

**Parameters:**
- `items`: List of items to process
- `process_fn`: Function to process each item
- `save_results`: Whether to save results
- `output_dir`: Directory to save results
- `**kwargs`: Additional arguments for process_fn

**Returns:**
- Dictionary containing:
  - 'results': Processing results
  - 'errors': Any errors encountered
  - 'stats': Processing statistics

## Utility Functions

### Image Utilities
```python
from src.utils.image_utils import (
    load_image,
    draw_boxes,
    resize_image,
    apply_text_overlay
)
```

#### load_image()
```python
def load_image(
    image: Union[str, np.ndarray, Image.Image]
) -> Tuple[np.ndarray, np.ndarray]
```
Loads and preprocesses images from various input types.

#### draw_boxes()
```python
def draw_boxes(
    image: np.ndarray,
    boxes: list,
    labels: list,
    scores: list = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray
```
Draws bounding boxes with labels on the image.

#### resize_image()
```python
def resize_image(
    image: np.ndarray,
    target_size: Union[Tuple[int, int], int],
    keep_aspect_ratio: bool = True
) -> np.ndarray
```
Resizes image while optionally maintaining aspect ratio.

#### apply_text_overlay()
```python
def apply_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    color: tuple = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray
```
Adds text overlay to image with background box.
