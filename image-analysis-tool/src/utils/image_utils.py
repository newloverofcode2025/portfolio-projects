import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple

def load_image(image: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess image from various input types.
    
    Args:
        image: Can be a file path, numpy array, or PIL Image

    Returns:
        Tuple of (BGR image for OpenCV, RGB image for display)
    """
    if isinstance(image, str):
        # Load from file path
        bgr_image = cv2.imread(image)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        # Handle numpy array
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            rgb_image = image
        else:
            raise ValueError("Input numpy array must be RGB format")
    elif isinstance(image, Image.Image):
        # Handle PIL Image
        rgb_image = np.array(image)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image type")
    
    return bgr_image, rgb_image

def draw_boxes(
    image: np.ndarray,
    boxes: list,
    labels: list,
    scores: list = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes with labels on the image.
    
    Args:
        image: Input image
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        labels: List of labels for each box
        scores: Optional list of confidence scores
        color: Box color in BGR format
        thickness: Line thickness

    Returns:
        Image with drawn boxes
    """
    image_copy = image.copy()
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if scores:
            label_text = f"{label}: {scores[idx]:.2f}"
        else:
            label_text = label
            
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            image_copy,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image_copy,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    return image_copy

def resize_image(
    image: np.ndarray,
    target_size: Union[Tuple[int, int], int],
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize image while optionally maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Either (width, height) or maximum dimension
        keep_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    if isinstance(target_size, int):
        height, width = image.shape[:2]
        if height > width:
            target_height = target_size
            target_width = int(width * (target_size / height))
        else:
            target_width = target_size
            target_height = int(height * (target_size / width))
    else:
        target_width, target_height = target_size
        
        if keep_aspect_ratio:
            current_height, current_width = image.shape[:2]
            scale = min(
                target_width / current_width,
                target_height / current_height
            )
            target_width = int(current_width * scale)
            target_height = int(current_height * scale)
    
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

def apply_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    color: tuple = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Add text overlay to image with background box.
    
    Args:
        image: Input image
        text: Text to overlay
        position: (x, y) position for text
        font_scale: Font scale factor
        color: Text color in BGR format
        thickness: Line thickness

    Returns:
        Image with text overlay
    """
    image_copy = image.copy()
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(
        image_copy,
        (x, y - text_height - baseline - 10),
        (x + text_width + 10, y),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        image_copy,
        text,
        (x + 5, y - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return image_copy
