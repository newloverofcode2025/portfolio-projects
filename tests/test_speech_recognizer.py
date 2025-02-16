import unittest
import numpy as np
import torch
from src.speech.speech_recognizer import SpeechRecognizer
import os

class TestSpeechRecognizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.recognizer = SpeechRecognizer()
        cls.sample_rate = 16000
        cls.duration = 3  # seconds
        
        # Create test audio (sine wave)
        t = np.linspace(0, cls.duration, cls.duration * cls.sample_rate)
        cls.test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
    def test_transcribe(self):
        """Test speech transcription."""
        # Save test audio temporarily
        import soundfile as sf
        temp_path = "temp_test.wav"
        sf.write(temp_path, self.test_audio, self.sample_rate)
        
        try:
            # Test transcription
            text = self.recognizer.transcribe(temp_path)
            self.assertIsInstance(text, str)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_detect_emotion(self):
        """Test emotion detection."""
        # Save test audio temporarily
        import soundfile as sf
        temp_path = "temp_test.wav"
        sf.write(temp_path, self.test_audio, self.sample_rate)
        
        try:
            # Test emotion detection
            emotions = self.recognizer.detect_emotion(temp_path)
            self.assertIsInstance(emotions, dict)
            self.assertTrue(all(isinstance(v, float) for v in emotions.values()))
            self.assertTrue(all(0 <= v <= 1 for v in emotions.values()))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_reduce_noise(self):
        """Test noise reduction."""
        # Add noise to test audio
        noise = np.random.normal(0, 0.1, len(self.test_audio))
        noisy_audio = self.test_audio + noise
        
        # Test noise reduction
        cleaned = self.recognizer.reduce_noise(noisy_audio, self.sample_rate)
        
        # Check output
        self.assertEqual(cleaned.shape, noisy_audio.shape)
        self.assertTrue(np.mean(np.abs(cleaned)) < np.mean(np.abs(noisy_audio)))
    
    def test_diarize(self):
        """Test speaker diarization."""
        # Save test audio temporarily
        import soundfile as sf
        temp_path = "temp_test.wav"
        sf.write(temp_path, self.test_audio, self.sample_rate)
        
        try:
            # Test diarization
            segments = self.recognizer.diarize(temp_path)
            self.assertIsInstance(segments, list)
            if segments:  # If any segments were detected
                self.assertTrue(all(isinstance(s, dict) for s in segments))
                self.assertTrue(all('speaker' in s for s in segments))
                self.assertTrue(all('start' in s for s in segments))
                self.assertTrue(all('end' in s for s in segments))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
