import librosa
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Tuple, Union

class MusicAnalyzer:
    def __init__(self):
        """Initialize music analysis components."""
        self.sample_rate = 22050  # Default sample rate for analysis
        self.hop_length = 512     # Number of samples between successive frames
        self.n_fft = 2048        # Length of the FFT window
        
    def detect_key(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> str:
        """
        Detect the musical key of the audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Detected musical key
        """
        # Compute chromagram
        chromagram = librosa.feature.chroma_cqt(
            y=audio_data,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Calculate key profile correlation
        key_profiles = self._get_key_profiles()
        correlations = []
        
        # Compare with each key profile
        chroma_avg = np.mean(chromagram, axis=1)
        for profile in key_profiles:
            correlation = np.correlate(chroma_avg, profile)
            correlations.append(correlation[0])
        
        # Find the key with highest correlation
        max_correlation_idx = np.argmax(correlations)
        key_names = self._get_key_names()
        
        return key_names[max_correlation_idx]
    
    def detect_tempo(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> float:
        """
        Detect the tempo (BPM) of the audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Tempo in BPM
        """
        # Extract onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_data,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Estimate tempo
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length
        )[0]
        
        return tempo
    
    def recognize_chords(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, List[float]]:
        """
        Recognize chord progressions in the audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with chord names and their corresponding timestamps
        """
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(
            y=audio_data,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Define chord templates
        chord_templates = self._get_chord_templates()
        
        # Initialize chord progression
        timestamps = librosa.times_like(chroma, sr=sample_rate, hop_length=self.hop_length)
        chord_progression = []
        
        # Analyze each frame
        for i in range(chroma.shape[1]):
            frame = chroma[:, i]
            
            # Compare with chord templates
            correlations = []
            for template in chord_templates:
                correlation = np.correlate(frame, template)
                correlations.append(correlation[0])
            
            # Get the most likely chord
            max_correlation_idx = np.argmax(correlations)
            chord_names = self._get_chord_names()
            chord_progression.append(chord_names[max_correlation_idx])
        
        # Create result dictionary
        result = {
            "timestamps": timestamps.tolist(),
            "chords": chord_progression
        }
        
        return result
    
    def _get_key_profiles(self) -> List[np.ndarray]:
        """Get Krumhansl-Schmuckler key profiles."""
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        profiles = []
        # Generate profiles for all keys
        for i in range(12):  # 12 possible keys
            # Rotate profiles for each key
            profiles.append(np.roll(major_profile, i))
            profiles.append(np.roll(minor_profile, i))
        
        return profiles
    
    def _get_key_names(self) -> List[str]:
        """Get list of key names."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        keys = []
        for note in notes:
            keys.append(f"{note} major")
            keys.append(f"{note} minor")
        return keys
    
    def _get_chord_templates(self) -> List[np.ndarray]:
        """Get chord templates for recognition."""
        # Major triad template
        major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        # Minor triad template
        minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        # Diminished triad template
        dim = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        
        templates = []
        # Generate templates for all root notes
        for i in range(12):
            templates.append(np.roll(major, i))  # Major chords
            templates.append(np.roll(minor, i))  # Minor chords
            templates.append(np.roll(dim, i))    # Diminished chords
        
        return templates
    
    def _get_chord_names(self) -> List[str]:
        """Get list of chord names."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chords = []
        for note in notes:
            chords.append(f"{note}")       # Major
            chords.append(f"{note}m")      # Minor
            chords.append(f"{note}dim")    # Diminished
        return chords
