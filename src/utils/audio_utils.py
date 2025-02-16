import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Union

def load_audio(
    file_path: str,
    sr: int = None,
    mono: bool = True,
    duration: float = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with librosa.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None for original)
        mono: Convert to mono if True
        duration: Duration to load in seconds
        offset: Start reading after this time (in seconds)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio_data, sample_rate = librosa.load(
            file_path,
            sr=sr,
            mono=mono,
            duration=duration,
            offset=offset
        )
        return audio_data, sample_rate
    except Exception as e:
        raise Exception(f"Error loading audio file: {str(e)}")

def save_audio(
    file_path: str,
    audio_data: np.ndarray,
    sample_rate: int,
    format: str = None
) -> None:
    """
    Save audio data to file.
    
    Args:
        file_path: Output file path
        audio_data: Audio data to save
        sample_rate: Sample rate of audio
        format: Output format (None for automatic)
    """
    try:
        sf.write(file_path, audio_data, sample_rate, format=format)
    except Exception as e:
        raise Exception(f"Error saving audio file: {str(e)}")

def convert_audio_format(
    input_path: str,
    output_path: str,
    target_sr: int = None,
    target_channels: int = None
) -> None:
    """
    Convert audio file format and properties.
    
    Args:
        input_path: Input audio file path
        output_path: Output audio file path
        target_sr: Target sample rate (None for original)
        target_channels: Target number of channels (None for original)
    """
    try:
        # Load audio
        audio_data, sr = load_audio(input_path, sr=target_sr, mono=(target_channels == 1))
        
        # Save in new format
        save_audio(output_path, audio_data, sr)
    except Exception as e:
        raise Exception(f"Error converting audio file: {str(e)}")

def split_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    segment_duration: float
) -> list:
    """
    Split audio into fixed-duration segments.
    
    Args:
        audio_data: Audio data to split
        sample_rate: Sample rate of audio
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of audio segments
    """
    segment_length = int(segment_duration * sample_rate)
    segments = []
    
    for start in range(0, len(audio_data), segment_length):
        end = start + segment_length
        segment = audio_data[start:end]
        
        # Pad last segment if needed
        if len(segment) < segment_length:
            segment = np.pad(
                segment,
                (0, segment_length - len(segment)),
                mode='constant'
            )
        
        segments.append(segment)
    
    return segments

def mix_audio(
    audio_data1: np.ndarray,
    audio_data2: np.ndarray,
    weight1: float = 0.5,
    weight2: float = 0.5
) -> np.ndarray:
    """
    Mix two audio signals with specified weights.
    
    Args:
        audio_data1: First audio signal
        audio_data2: Second audio signal
        weight1: Weight for first signal (0-1)
        weight2: Weight for second signal (0-1)
        
    Returns:
        Mixed audio signal
    """
    # Ensure equal lengths
    min_length = min(len(audio_data1), len(audio_data2))
    audio_data1 = audio_data1[:min_length]
    audio_data2 = audio_data2[:min_length]
    
    # Mix signals
    mixed = (weight1 * audio_data1 + weight2 * audio_data2)
    
    # Normalize to prevent clipping
    mixed = mixed / np.max(np.abs(mixed))
    
    return mixed

def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        return librosa.get_duration(filename=file_path)
    except Exception as e:
        raise Exception(f"Error getting audio duration: {str(e)}")

def get_audio_info(file_path: str) -> dict:
    """
    Get audio file information.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    try:
        info = sf.info(file_path)
        return {
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'format': info.format,
            'subtype': info.subtype
        }
    except Exception as e:
        raise Exception(f"Error getting audio info: {str(e)}")

def normalize_audio(
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
