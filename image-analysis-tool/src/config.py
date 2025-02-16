import torch
import os
from pathlib import Path
from typing import Dict, Any
import json

class Config:
    """Global configuration for the AI Image Analysis Tool."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_threads = os.cpu_count()
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # Default model configurations
        self.model_configs = {
            'object_detection': {
                'model_type': 'yolov8',
                'model_size': 'n',  # n, s, m, l, x
                'confidence_threshold': 0.25,
                'custom_classes': None
            },
            'face_analysis': {
                'enable_landmarks': True,
                'enable_recognition': True,
                'min_face_size': 20,
                'custom_face_db': None
            },
            'enhancement': {
                'use_super_resolution': False,
                'denoise_strength': 10,
                'sharpen_strength': 0.5,
                'enable_auto_contrast': True
            }
        }
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config_dict: Dictionary containing new configuration values
        """
        for section, values in config_dict.items():
            if section in self.model_configs:
                self.model_configs[section].update(values)
    
    def save_config(self, filepath: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration file
        """
        config_data = {
            'device': str(self.device),
            'num_threads': self.num_threads,
            'models_dir': str(self.models_dir),
            'model_configs': self.model_configs
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, filepath: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
        """
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.device = torch.device(config_data['device'])
        self.num_threads = config_data['num_threads']
        self.models_dir = Path(config_data['models_dir'])
        self.model_configs = config_data['model_configs']

# Global configuration instance
config = Config()
