import cv2
import numpy as np
import torch
from typing import Tuple, Union
from ..config import config

class ImageEnhancer:
    def __init__(self):
        """Initialize image enhancement parameters."""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Load enhancement config
        self.config = config.model_configs['enhancement']
        
        # Initialize super-resolution model if enabled
        self.sr_model = None
        if self.config['use_super_resolution']:
            self.sr_model = torch.hub.load('xinntao/Real-ESRGAN', 'RRDBNet_x4')
            self.sr_model.to(config.device)
            self.sr_model.eval()

    def enhance(
        self,
        image: np.ndarray,
        enhance_contrast: bool = True,
        reduce_noise: bool = True,
        sharpen: bool = True,
        use_super_resolution: bool = None
    ) -> np.ndarray:
        """
        Enhance the image using various techniques.
        
        Args:
            image: Input image
            enhance_contrast: Whether to enhance contrast
            reduce_noise: Whether to reduce noise
            sharpen: Whether to sharpen the image
            use_super_resolution: Override config setting for super-resolution

        Returns:
            Enhanced image
        """
        # Move image to GPU if available
        if config.device.type == 'cuda':
            image_tensor = torch.from_numpy(image).to(config.device)
        else:
            image_tensor = torch.from_numpy(image)
        
        # Work with a copy of the image
        enhanced = image_tensor.float() / 255.0
        
        if enhance_contrast and self.config['enable_auto_contrast']:
            # Convert to LAB color space
            enhanced = enhanced.cpu().numpy()
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l = self.clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhanced = torch.from_numpy(enhanced).to(config.device)
        
        if reduce_noise:
            # Apply denoising
            denoise_strength = self.config['denoise_strength']
            if config.device.type == 'cuda':
                # Use GPU-accelerated denoising
                enhanced = self._cuda_denoise(enhanced, strength=denoise_strength)
            else:
                enhanced = enhanced.cpu().numpy()
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced,
                    None,
                    denoise_strength,
                    denoise_strength,
                    7,
                    21
                )
                enhanced = torch.from_numpy(enhanced).to(config.device)
        
        if sharpen:
            # Create sharpening kernel
            kernel = torch.tensor([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ], dtype=torch.float32).to(config.device)
            
            # Apply sharpening with GPU acceleration if available
            if config.device.type == 'cuda':
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                enhanced = enhanced.permute(2, 0, 1).unsqueeze(0)
                enhanced = torch.nn.functional.conv2d(
                    enhanced,
                    kernel,
                    padding=1
                )
                enhanced = enhanced.squeeze(0).permute(1, 2, 0)
            else:
                enhanced = enhanced.cpu().numpy()
                enhanced = cv2.filter2D(enhanced, -1, kernel.cpu().numpy())
                enhanced = torch.from_numpy(enhanced).to(config.device)
        
        # Apply super-resolution if enabled
        use_sr = use_super_resolution if use_super_resolution is not None else self.config['use_super_resolution']
        if use_sr and self.sr_model is not None:
            enhanced = self._apply_super_resolution(enhanced)
        
        # Convert back to uint8
        enhanced = (enhanced * 255).clamp(0, 255).byte()
        
        # Move back to CPU if needed
        if config.device.type == 'cuda':
            enhanced = enhanced.cpu()
        
        return enhanced.numpy()

    def _cuda_denoise(
        self,
        image: torch.Tensor,
        strength: float = 10.0,
        kernel_size: int = 3
    ) -> torch.Tensor:
        """
        GPU-accelerated image denoising.
        
        Args:
            image: Input image tensor
            strength: Denoising strength
            kernel_size: Size of the denoising kernel

        Returns:
            Denoised image tensor
        """
        # Implement bilateral filtering on GPU
        padding = kernel_size // 2
        channels = image.shape[2]
        
        # Prepare gaussian kernels
        spatial_kernel = self._gaussian_kernel(kernel_size, strength).to(config.device)
        
        # Pad input
        padded = torch.nn.functional.pad(image, (padding, padding, padding, padding), mode='reflect')
        
        # Process each channel
        output = torch.zeros_like(image)
        for c in range(channels):
            channel = padded[:, :, c]
            filtered = torch.nn.functional.conv2d(
                channel.unsqueeze(0).unsqueeze(0),
                spatial_kernel.unsqueeze(0).unsqueeze(0),
                padding=padding
            )
            output[:, :, c] = filtered.squeeze()
        
        return output

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """
        Create a Gaussian kernel.
        
        Args:
            size: Kernel size
            sigma: Gaussian sigma

        Returns:
            Gaussian kernel tensor
        """
        coords = torch.arange(size).float() - (size - 1) / 2
        coords = coords.view(-1, 1).repeat(1, size)
        coords_y = coords.t()
        
        kernel = torch.exp(-(coords**2 + coords_y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel

    def _apply_super_resolution(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply super-resolution to the image.
        
        Args:
            image: Input image tensor

        Returns:
            Super-resolved image tensor
        """
        with torch.no_grad():
            # Prepare input
            x = image.permute(2, 0, 1).unsqueeze(0)
            
            # Run super-resolution
            output = self.sr_model(x)
            
            # Post-process output
            output = output.squeeze(0).permute(1, 2, 0)
            
            return output

    def adjust_brightness_contrast(
        self,
        image: np.ndarray,
        brightness: float = 1.0,
        contrast: float = 1.0
    ) -> np.ndarray:
        """
        Adjust image brightness and contrast.
        
        Args:
            image: Input image
            brightness: Brightness factor (1.0 = original)
            contrast: Contrast factor (1.0 = original)

        Returns:
            Adjusted image
        """
        # Convert to float for calculations
        adjusted = image.astype(float)
        
        # Apply brightness
        adjusted *= brightness
        
        # Apply contrast
        adjusted = (adjusted - 128) * contrast + 128
        
        # Clip values and convert back to uint8
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def remove_background(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove image background using OpenCV.
        
        Args:
            image: Input image
            threshold: Threshold for background removal

        Returns:
            Tuple of (image with transparent background, mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Add alpha channel
        b, g, r = cv2.split(image)
        alpha = mask
        
        # Merge channels with alpha
        return cv2.merge([b, g, r, alpha]), mask

def enhance(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to enhance an image.
    
    Args:
        image: Input image

    Returns:
        Enhanced image
    """
    enhancer = ImageEnhancer()
    return enhancer.enhance(image)
