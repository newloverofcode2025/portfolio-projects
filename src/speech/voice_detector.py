import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Union
from transformers import pipeline
import torch.nn.functional as F

class VoiceDetector:
    def __init__(self):
        """Initialize voice detection models."""
        # Initialize voice activity detection model
        self.vad_model = pipeline(
            "audio-classification",
            model="microsoft/wavlm-base-plus-sv",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize speaker verification model
        self.sv_model = pipeline(
            "audio-classification",
            model="microsoft/wavlm-base-plus-sv",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Speaker embeddings database
        self.speaker_db = {}
        
    def detect_voice_activity(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        threshold: float = 0.5
    ) -> List[Dict[str, Union[float, bool]]]:
        """
        Detect segments with voice activity.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            threshold: Voice activity detection threshold
            
        Returns:
            List of segments with voice activity information
        """
        # Process in segments
        segment_length = int(0.5 * sample_rate)  # 500ms segments
        segments = []
        
        for i in range(0, len(audio_data), segment_length):
            segment = audio_data[i:i + segment_length]
            if len(segment) < segment_length:
                # Pad last segment if needed
                segment = np.pad(segment, (0, segment_length - len(segment)))
            
            # Get voice activity probability
            result = self.vad_model(segment)
            voice_prob = result[0]["score"] if result[0]["label"] == "speech" else 1 - result[0]["score"]
            
            segments.append({
                "start": i / sample_rate,
                "end": (i + len(segment)) / sample_rate,
                "has_voice": voice_prob > threshold,
                "voice_probability": voice_prob
            })
        
        return self._merge_segments(segments)
    
    def enroll_speaker(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_id: str
    ) -> None:
        """
        Enroll a new speaker in the database.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            speaker_id: Unique identifier for the speaker
        """
        # Get speaker embedding
        embedding = self._get_speaker_embedding(audio_data, sample_rate)
        
        # Store in database
        if speaker_id in self.speaker_db:
            # Average with existing embedding
            self.speaker_db[speaker_id] = (
                self.speaker_db[speaker_id] + embedding
            ) / 2
        else:
            self.speaker_db[speaker_id] = embedding
    
    def verify_speaker(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_id: str,
        threshold: float = 0.7
    ) -> Dict[str, Union[bool, float]]:
        """
        Verify if audio matches enrolled speaker.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            speaker_id: Speaker to verify against
            threshold: Similarity threshold
            
        Returns:
            Dictionary with verification result and similarity score
        """
        if speaker_id not in self.speaker_db:
            raise ValueError(f"Speaker {speaker_id} not enrolled")
        
        # Get speaker embedding
        embedding = self._get_speaker_embedding(audio_data, sample_rate)
        
        # Compare with enrolled embedding
        similarity = F.cosine_similarity(
            torch.tensor(embedding).unsqueeze(0),
            torch.tensor(self.speaker_db[speaker_id]).unsqueeze(0)
        ).item()
        
        return {
            "is_match": similarity > threshold,
            "similarity": similarity
        }
    
    def _get_speaker_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Get speaker embedding from audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Speaker embedding vector
        """
        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data)
        else:
            audio_tensor = audio_data
        
        # Ensure correct shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.sv_model(audio_tensor.numpy())[0]["hidden_states"]
            embedding = torch.mean(torch.tensor(embedding), dim=0)
        
        return embedding.numpy()
    
    def _merge_segments(
        self,
        segments: List[Dict[str, Union[float, bool]]]
    ) -> List[Dict[str, Union[float, bool]]]:
        """
        Merge consecutive voice segments.
        
        Args:
            segments: List of voice activity segments
            
        Returns:
            Merged segments
        """
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            if (segment["has_voice"] == merged[-1]["has_voice"] and
                abs(segment["start"] - merged[-1]["end"]) < 0.1):
                # Merge segments
                merged[-1]["end"] = segment["end"]
                merged[-1]["voice_probability"] = (
                    merged[-1]["voice_probability"] + segment["voice_probability"]
                ) / 2
            else:
                merged.append(segment)
        
        return merged
