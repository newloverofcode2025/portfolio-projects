import torch
import torchaudio
import numpy as np
from typing import Dict, List
import librosa

class GenreClassifier:
    def __init__(self):
        """Initialize the genre classifier model."""
        # Load pre-trained model (using a simple CNN architecture)
        self.model = self._load_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.sample_rate = 22050
        self.duration = 30  # Process 30 seconds of audio
        self.genres = [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        
    def predict(self, audio_path: str) -> str:
        """
        Predict the genre of an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Predicted genre
        """
        # Get probabilities
        probs = self.predict_proba(audio_path)
        
        # Return genre with highest probability
        return self.genres[np.argmax(list(probs.values()))]
    
    def predict_proba(self, audio_path: str) -> Dict[str, float]:
        """
        Get genre probabilities for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of genre probabilities
        """
        # Load and preprocess audio
        features = self._extract_features(audio_path)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Convert to dictionary
        result = {genre: float(prob) for genre, prob in zip(self.genres, probs[0])}
        return result
    
    def _extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract audio features for genre classification.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Feature array
        """
        # Load audio
        y, sr = librosa.load(audio_path, duration=self.duration)
        
        # Resample if necessary
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        
        # Extract features
        features = []
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.append(mel_spec_db)
        
        # Chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        features.append(chroma)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)
        features.append(contrast)
        
        # Stack and normalize features
        features = np.vstack(features)
        features = (features - np.mean(features)) / np.std(features)
        
        return features
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load the genre classification model.
        
        Returns:
            PyTorch model
        """
        # Define model architecture
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, len(self.genres))
        )
        
        # Load pre-trained weights if available
        try:
            model.load_state_dict(torch.load('models/genre_classifier.pth'))
        except:
            print("No pre-trained weights found. Using initialized model.")
        
        return model
