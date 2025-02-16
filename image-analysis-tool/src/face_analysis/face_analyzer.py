import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import torch
from typing import List, Dict, Tuple, Union, Any
from ..utils.image_utils import draw_boxes, apply_text_overlay
from ..config import config
import os
from pathlib import Path

class FaceAnalyzer:
    def __init__(self):
        """Initialize the face analysis components."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load face analysis config
        face_config = config.model_configs['face_analysis']
        
        # Initialize face mesh with config settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face recognition if enabled
        self.enable_recognition = face_config['enable_recognition']
        if self.enable_recognition:
            self.face_encoder = torch.hub.load('timesler/facenet-pytorch', 'inception_resnet_v1', 
                                             pretrained='vggface2').to(config.device)
            self.face_encoder.eval()
        
        # Load custom face database if specified
        self.custom_face_db = None
        if face_config['custom_face_db']:
            self.load_face_database(face_config['custom_face_db'])
    
    def load_face_database(self, db_path: str) -> None:
        """
        Load a custom face database for recognition.
        
        Args:
            db_path: Path to the face database file
        """
        if db_path and Path(db_path).exists():
            self.custom_face_db = torch.load(db_path, map_location=config.device)
    
    def analyze(
        self,
        image: np.ndarray,
        draw_landmarks: bool = True,
        detect_facial_features: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Analyze faces in the image.
        
        Args:
            image: Input image
            draw_landmarks: Whether to draw facial landmarks
            detect_facial_features: Whether to detect facial features

        Returns:
            Tuple of (annotated image, list of face analysis results)
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        # Create copy for drawing
        annotated_image = image.copy()
        
        face_analyses = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get face bounding box
                h, w = image.shape[:2]
                x_min = w
                x_max = 0
                y_min = h
                y_max = 0
                
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                
                # Draw landmarks if requested
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    )
                
                # Detect facial features if requested
                facial_features = {}
                if detect_facial_features and self.enable_recognition:
                    face_image = image[y_min:y_max, x_min:x_max]
                    if face_image.size > 0:
                        # Prepare face image for recognition
                        face_tensor = torch.from_numpy(face_image).float()
                        face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
                        face_tensor = face_tensor.to(config.device)
                        
                        # Get face embedding
                        with torch.no_grad():
                            embedding = self.face_encoder(face_tensor)
                            embedding = embedding.cpu().numpy()[0]
                        
                        facial_features['embedding'] = embedding.tolist()
                        
                        # Match with custom database if available
                        if self.custom_face_db is not None:
                            matches = self.find_matches(embedding)
                            if matches:
                                facial_features['matches'] = matches
                
                # Add face analysis results
                face_analyses.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'landmarks': [[int(l.x * w), int(l.y * h)] 
                                for l in face_landmarks.landmark],
                    'facial_features': facial_features
                })
                
                # Draw bounding box
                cv2.rectangle(
                    annotated_image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )
                
                # Add face number label
                apply_text_overlay(
                    annotated_image,
                    f"Face {len(face_analyses)}",
                    (x_min, y_min - 10)
                )
        
        return annotated_image, face_analyses
    
    def find_matches(
        self,
        embedding: np.ndarray,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find matches for a face embedding in the custom database.
        
        Args:
            embedding: Face embedding to match
            threshold: Similarity threshold

        Returns:
            List of matching faces with similarity scores
        """
        matches = []
        if self.custom_face_db is None:
            return matches
        
        embedding_tensor = torch.from_numpy(embedding).to(config.device)
        
        for name, stored_embedding in self.custom_face_db.items():
            stored_tensor = torch.from_numpy(stored_embedding).to(config.device)
            similarity = torch.nn.functional.cosine_similarity(
                embedding_tensor, stored_tensor, dim=0
            ).item()
            
            if similarity > threshold:
                matches.append({
                    'name': name,
                    'similarity': similarity
                })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

def analyze(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to run face analysis.
    
    Args:
        image: Input image

    Returns:
        Annotated image with face analysis
    """
    analyzer = FaceAnalyzer()
    annotated_image, _ = analyzer.analyze(image)
    return annotated_image
