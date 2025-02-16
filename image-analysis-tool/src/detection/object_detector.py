import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple, List, Union
import torch
from ..config import config
from ..models.model_manager import ModelManager
from ..utils.image_utils import draw_boxes, resize_image

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", custom_model: bool = False):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_name: Name of the YOLO model to use
            custom_model: Whether to use a custom model
        """
        self.model_manager = ModelManager()
        
        if custom_model:
            model_path = self.model_manager.custom_models_dir / model_name / f"{model_name}.pt"
        else:
            model_type, size = model_name.replace('.pt', '').split('v8')
            model_path = self.model_manager.download_model('yolov8', size)
        
        # Optimize model for device
        self.model_manager.optimize_for_device(model_path)
        
        # Load model with device configuration
        self.model = YOLO(str(model_path))
        self.model.to(config.device)
        
        # Get class names
        self.class_names = self.model.names
        
        # Configure model based on config
        conf = config.model_configs['object_detection']['confidence_threshold']
        self.model.conf = conf
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = None,
        max_size: int = 1280
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect objects in the image.
        
        Args:
            image: Input image
            conf_threshold: Optional override for confidence threshold
            max_size: Maximum image dimension for processing

        Returns:
            Tuple of (annotated image, list of detections)
        """
        # Resize image if needed
        orig_size = image.shape[:2]
        if max(orig_size) > max_size:
            image = resize_image(image, max_size)
        
        # Convert image to tensor and move to device
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(config.device)
        
        # Update confidence threshold if provided
        if conf_threshold is not None:
            self.model.conf = conf_threshold
        
        # Run inference
        results = self.model(image)[0]
        
        # Process results
        boxes = []
        labels = []
        scores = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > self.model.conf:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_names[int(class_id)])
                scores.append(score)
        
        # Convert image back to CPU for drawing
        image = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        
        # Draw detections
        annotated_image = draw_boxes(
            image,
            boxes,
            labels,
            scores,
            color=(0, 255, 0),
            thickness=2
        )
        
        # Prepare detection results
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                'bbox': box,
                'class': label,
                'confidence': score
            })
        
        return annotated_image, detections

def detect(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to run object detection.
    
    Args:
        image: Input image

    Returns:
        Annotated image with detections
    """
    detector = ObjectDetector()
    annotated_image, _ = detector.detect(image)
    return annotated_image
