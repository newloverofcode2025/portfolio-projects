import torch
import torchaudio
import transformers
import numpy as np
import noisereduce as nr
from typing import Dict, List, Union
from transformers import pipeline

class SpeechRecognizer:
    def __init__(self):
        """Initialize speech recognition models and processors."""
        # Initialize ASR model
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-large-960h",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize speaker diarization model
        self.diarization_pipeline = pipeline(
            "audio-classification",
            model="speechbrain/spkrec-ecapa-voxceleb",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize emotion detection model
        self.emotion_pipeline = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            device=0 if torch.cuda.is_available() else -1
        )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe speech to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Transcribe
        result = self.asr_pipeline(waveform.numpy().squeeze())
        return result["text"]

    def diarize(self, audio_path: str) -> List[Dict[str, Union[str, float]]]:
        """
        Perform speaker diarization.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments with timing information
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Process in segments
        segment_length = 4 * sample_rate  # 4 second segments
        segments = []
        
        for i in range(0, waveform.shape[1], segment_length):
            segment = waveform[:, i:i + segment_length]
            if segment.shape[1] < segment_length:
                # Pad last segment if needed
                segment = torch.nn.functional.pad(
                    segment, (0, segment_length - segment.shape[1])
                )
            
            # Get speaker embedding
            result = self.diarization_pipeline(segment.numpy().squeeze())
            
            segments.append({
                "speaker": result[0]["label"],
                "start": i / sample_rate,
                "end": (i + segment.shape[1]) / sample_rate
            })
        
        return self._merge_speaker_segments(segments)

    def detect_emotion(self, audio_path: str) -> Dict[str, float]:
        """
        Detect emotions in speech.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of emotion probabilities
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Process audio
        result = self.emotion_pipeline(waveform.numpy().squeeze())
        
        # Convert to dictionary
        emotions = {item["label"]: item["score"] for item in result}
        return emotions

    def reduce_noise(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        noise_reduce_strength: float = 0.75
    ) -> np.ndarray:
        """
        Reduce noise in audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            noise_reduce_strength: Strength of noise reduction (0-1)
            
        Returns:
            Noise-reduced audio data
        """
        # Ensure audio is in the correct format
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(1, -1)
        
        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            prop_decrease=noise_reduce_strength,
            n_fft=2048,
            win_length=2048,
            hop_length=512
        )
        
        return reduced

    def _merge_speaker_segments(
        self,
        segments: List[Dict[str, Union[str, float]]]
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Merge consecutive segments from the same speaker.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Merged speaker segments
        """
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            if (segment["speaker"] == merged[-1]["speaker"] and
                abs(segment["start"] - merged[-1]["end"]) < 0.1):
                # Merge segments
                merged[-1]["end"] = segment["end"]
            else:
                merged.append(segment)
        
        return merged
