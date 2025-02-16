import numpy as np
import librosa
from typing import Dict, List, Tuple, Union
import scipy.signal as signal

class BeatTracker:
    def __init__(self):
        """Initialize beat tracking components."""
        self.hop_length = 512
        self.win_length = 2048
        
    def find_beats(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find beat positions and tempo in audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with beat positions and tempo information
        """
        # Get onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio_data,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Dynamic programming beat tracker
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length,
            tightness=100
        )
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sample_rate, hop_length=self.hop_length)
        
        # Get beat strength
        beat_strength = onset_env[beats]
        
        return {
            "tempo": tempo,
            "beat_times": beat_times,
            "beat_strength": beat_strength
        }
    
    def analyze_rhythm(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze rhythmic properties of audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with rhythm analysis results
        """
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_data,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Compute tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Get tempo estimates
        tempo_bpm = librosa.tempo_frequencies(tempogram.shape[0])
        tempo_dist = np.mean(tempogram, axis=1)
        
        # Find peaks in tempo distribution
        peaks = signal.find_peaks(tempo_dist, height=np.mean(tempo_dist))[0]
        peak_tempos = tempo_bpm[peaks]
        peak_strengths = tempo_dist[peaks]
        
        # Sort by strength
        sort_idx = np.argsort(peak_strengths)[::-1]
        peak_tempos = peak_tempos[sort_idx]
        peak_strengths = peak_strengths[sort_idx]
        
        # Calculate rhythm regularity
        regularity = self._calculate_rhythm_regularity(onset_env)
        
        return {
            "main_tempo": peak_tempos[0] if len(peak_tempos) > 0 else 0,
            "secondary_tempos": peak_tempos[1:].tolist(),
            "tempo_strengths": peak_strengths.tolist(),
            "rhythm_regularity": regularity
        }
    
    def find_groove(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze groove characteristics of audio.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with groove analysis results
        """
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(
            y=audio_data,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Find beats
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Calculate microtiming deviations
        microtiming = self._calculate_microtiming(onset_env, beats)
        
        # Calculate syncopation
        syncopation = self._calculate_syncopation(onset_env, beats)
        
        # Calculate swing ratio
        swing_ratio = self._calculate_swing_ratio(onset_env, beats)
        
        return {
            "microtiming": microtiming.tolist(),
            "syncopation": float(syncopation),
            "swing_ratio": float(swing_ratio)
        }
    
    def _calculate_rhythm_regularity(
        self,
        onset_env: np.ndarray
    ) -> float:
        """
        Calculate rhythm regularity from onset envelope.
        
        Args:
            onset_env: Onset strength envelope
            
        Returns:
            Rhythm regularity score (0-1)
        """
        # Calculate autocorrelation
        acf = librosa.autocorrelate(onset_env)
        acf = acf[:len(acf)//2]
        
        # Find peaks in autocorrelation
        peaks = signal.find_peaks(acf)[0]
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate regularity from peak heights
        peak_heights = acf[peaks]
        regularity = np.mean(peak_heights[1:]) / peak_heights[0]
        
        return float(regularity)
    
    def _calculate_microtiming(
        self,
        onset_env: np.ndarray,
        beats: np.ndarray
    ) -> np.ndarray:
        """
        Calculate microtiming deviations from beats.
        
        Args:
            onset_env: Onset strength envelope
            beats: Beat positions
            
        Returns:
            Array of microtiming deviations
        """
        # Get local maxima in onset envelope
        peaks = signal.find_peaks(onset_env)[0]
        
        # Calculate distance to nearest beat
        microtiming = []
        for peak in peaks:
            nearest_beat = beats[np.argmin(np.abs(beats - peak))]
            deviation = (peak - nearest_beat) / self.hop_length
            microtiming.append(deviation)
        
        return np.array(microtiming)
    
    def _calculate_syncopation(
        self,
        onset_env: np.ndarray,
        beats: np.ndarray
    ) -> float:
        """
        Calculate syncopation level.
        
        Args:
            onset_env: Onset strength envelope
            beats: Beat positions
            
        Returns:
            Syncopation score (0-1)
        """
        if len(beats) < 2:
            return 0.0
        
        # Get onset strengths at and between beats
        beat_strengths = onset_env[beats]
        between_strengths = []
        
        for i in range(len(beats)-1):
            mid_point = (beats[i] + beats[i+1]) // 2
            between_strengths.append(onset_env[mid_point])
        
        between_strengths = np.array(between_strengths)
        
        # Calculate syncopation as ratio of between-beat to on-beat strengths
        syncopation = np.mean(between_strengths) / np.mean(beat_strengths)
        
        return float(syncopation)
    
    def _calculate_swing_ratio(
        self,
        onset_env: np.ndarray,
        beats: np.ndarray
    ) -> float:
        """
        Calculate swing ratio.
        
        Args:
            onset_env: Onset strength envelope
            beats: Beat positions
            
        Returns:
            Swing ratio (typically 1-3)
        """
        if len(beats) < 3:
            return 1.0
        
        # Get onset strengths at eighth note positions
        eighths = []
        for i in range(len(beats)-1):
            beat_start = beats[i]
            beat_end = beats[i+1]
            mid_point = (beat_start + beat_end) // 2
            
            # Get duration of first and second eighth notes
            first_eighth = mid_point - beat_start
            second_eighth = beat_end - mid_point
            
            eighths.append(first_eighth)
            eighths.append(second_eighth)
        
        eighths = np.array(eighths)
        
        # Calculate average ratio between odd and even eighth notes
        odd_eighths = eighths[::2]
        even_eighths = eighths[1::2]
        swing_ratio = np.mean(odd_eighths) / np.mean(even_eighths)
        
        return float(swing_ratio)
