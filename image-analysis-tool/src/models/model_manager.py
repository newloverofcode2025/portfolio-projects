import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
from tqdm import tqdm
import hashlib
import yaml
from ..config import config

class ModelManager:
    """Manages AI models, including downloading, validation, and customization."""
    
    MODEL_URLS = {
        'yolov8': {
            'n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            's': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'm': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt'
        }
    }
    
    def __init__(self):
        """Initialize the model manager."""
        self.models_dir = config.models_dir
        self.models_dir.mkdir(exist_ok=True)
        self.custom_models_dir = self.models_dir / 'custom'
        self.custom_models_dir.mkdir(exist_ok=True)
    
    def download_model(
        self,
        model_type: str,
        model_size: str,
        force: bool = False
    ) -> Path:
        """
        Download a model if it doesn't exist.
        
        Args:
            model_type: Type of model (e.g., 'yolov8')
            model_size: Size of model (e.g., 'n', 's', 'm', 'l', 'x')
            force: Force download even if model exists

        Returns:
            Path to the downloaded model
        """
        if model_type not in self.MODEL_URLS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if model_size not in self.MODEL_URLS[model_type]:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        url = self.MODEL_URLS[model_type][model_size]
        filename = f"{model_type}_{model_size}.pt"
        model_path = self.models_dir / filename
        
        if model_path.exists() and not force:
            return model_path
        
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f, tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        return model_path
    
    def create_custom_model(
        self,
        base_model_type: str,
        base_model_size: str,
        custom_config: Dict[str, Any],
        model_name: str
    ) -> Path:
        """
        Create a custom model based on a pre-trained model.
        
        Args:
            base_model_type: Type of base model
            base_model_size: Size of base model
            custom_config: Custom configuration
            model_name: Name for the custom model

        Returns:
            Path to the custom model
        """
        # Download base model
        base_model_path = self.download_model(base_model_type, base_model_size)
        
        # Create custom model directory
        custom_model_dir = self.custom_models_dir / model_name
        custom_model_dir.mkdir(exist_ok=True)
        
        # Save custom configuration
        config_path = custom_model_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(custom_config, f)
        
        # Copy base model
        custom_model_path = custom_model_dir / f"{model_name}.pt"
        if not custom_model_path.exists():
            import shutil
            shutil.copy2(base_model_path, custom_model_path)
        
        return custom_model_path
    
    def optimize_for_device(
        self,
        model_path: Path,
        device: torch.device = None
    ) -> None:
        """
        Optimize model for the specified device.
        
        Args:
            model_path: Path to the model
            device: Target device (defaults to config.device)
        """
        if device is None:
            device = config.device
        
        model = torch.load(model_path)
        model.to(device)
        
        if device.type == 'cuda':
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            if hasattr(model, 'half'):
                model = model.half()  # Convert to FP16
        
        torch.save(model, model_path)
    
    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_path: Path to the model

        Returns:
            Dictionary containing model information
        """
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        
        # Load model to get size and device
        model = torch.load(model_path, map_location='cpu')
        
        return {
            'path': str(model_path),
            'size': model_path.stat().st_size,
            'hash': model_hash,
            'device': next(model.parameters()).device.type,
            'parameters': sum(p.numel() for p in model.parameters()),
            'is_custom': str(self.custom_models_dir) in str(model_path)
        }
    
    def list_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available models.

        Returns:
            Dictionary containing lists of standard and custom models
        """
        standard_models = []
        custom_models = []
        
        # List standard models
        for model_file in self.models_dir.glob('*.pt'):
            standard_models.append(self.get_model_info(model_file))
        
        # List custom models
        for model_dir in self.custom_models_dir.iterdir():
            if model_dir.is_dir():
                model_file = model_dir / f"{model_dir.name}.pt"
                if model_file.exists():
                    custom_models.append(self.get_model_info(model_file))
        
        return {
            'standard_models': standard_models,
            'custom_models': custom_models
        }
