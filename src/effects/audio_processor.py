import numpy as np
from typing import Dict, List, Union
from pedalboard import Pedalboard, Chorus, Reverb, Delay, Distortion, Gain, PitchShift
from pedalboard.io import AudioFile
import librosa

class AudioProcessor:
    def __init__(self):
        """Initialize audio processing components."""
        self.available_effects = {
            "Reverb": self._create_reverb,
            "Delay": self._create_delay,
            "Chorus": self._create_chorus,
            "Distortion": self._create_distortion,
            "PitchShift": self._create_pitch_shift
        }
    
    def apply_effects(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        effects: List[str],
        parameters: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply audio effects to the input audio.
        
        Args:
            audio_data: Input audio data
            sample_rate: Sample rate of audio
            effects: List of effects to apply
            parameters: Dictionary of effect parameters
            
        Returns:
            Processed audio data
        """
        # Create pedalboard with selected effects
        board = Pedalboard()
        
        for effect in effects:
            if effect in self.available_effects:
                effect_params = {k: v for k, v in parameters.items() 
                               if k.startswith(effect.lower())}
                board.append(self.available_effects[effect](effect_params))
        
        # Process audio
        processed = board.process(audio_data, sample_rate)
        
        return processed
    
    def _create_reverb(self, params: Dict[str, float]) -> Reverb:
        """Create reverb effect with given parameters."""
        return Reverb(
            room_size=params.get('reverb_room_size', 0.5),
            damping=params.get('reverb_damping', 0.5),
            width=params.get('reverb_width', 1.0),
            wet_level=params.get('reverb_wet', 0.33),
            dry_level=params.get('reverb_dry', 0.4)
        )
    
    def _create_delay(self, params: Dict[str, float]) -> Delay:
        """Create delay effect with given parameters."""
        return Delay(
            delay_seconds=params.get('delay_time', 0.5) / 1000.0,  # Convert ms to seconds
            feedback=params.get('delay_feedback', 0.3),
            mix=params.get('delay_mix', 0.5)
        )
    
    def _create_chorus(self, params: Dict[str, float]) -> Chorus:
        """Create chorus effect with given parameters."""
        return Chorus(
            rate_hz=params.get('chorus_rate', 1.0),
            depth=params.get('chorus_depth', 0.25),
            centre_delay_ms=params.get('chorus_delay', 7.0),
            feedback=params.get('chorus_feedback', 0.0),
            mix=params.get('chorus_mix', 0.5)
        )
    
    def _create_distortion(self, params: Dict[str, float]) -> Distortion:
        """Create distortion effect with given parameters."""
        return Distortion(
            drive_db=params.get('distortion_drive', 25.0)
        )
    
    def _create_pitch_shift(self, params: Dict[str, float]) -> PitchShift:
        """Create pitch shift effect with given parameters."""
        return PitchShift(
            semitones=params.get('pitch_shift_semitones', 0.0)
        )
    
    def normalize_audio(
        self,
        audio_data: np.ndarray,
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            audio_data: Input audio data
            target_db: Target dB level
            
        Returns:
            Normalized audio data
        """
        # Calculate current dB level
        current_db = 20 * np.log10(np.sqrt(np.mean(audio_data**2)))
        
        # Calculate gain needed
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized = audio_data * gain_linear
        
        return normalized
    
    def time_stretch(
        self,
        audio_data: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        
        Args:
            audio_data: Input audio data
            rate: Time stretch factor (1.0 = normal speed)
            
        Returns:
            Time-stretched audio data
        """
        return librosa.effects.time_stretch(audio_data, rate=rate)
    
    def pitch_shift(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        semitones: float
    ) -> np.ndarray:
        """
        Shift the pitch of audio without changing tempo.
        
        Args:
            audio_data: Input audio data
            sample_rate: Sample rate of audio
            semitones: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio data
        """
        return librosa.effects.pitch_shift(
            audio_data,
            sr=sample_rate,
            n_steps=semitones
        )
