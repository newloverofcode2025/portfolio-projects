import unittest
import numpy as np
from src.effects.audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.processor = AudioProcessor()
        cls.sample_rate = 44100
        cls.duration = 2  # seconds
        
        # Create test audio (sine wave)
        t = np.linspace(0, cls.duration, cls.duration * cls.sample_rate)
        cls.test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    def test_apply_reverb(self):
        """Test reverb effect."""
        effects = ["Reverb"]
        parameters = {
            "reverb_room_size": 0.8,
            "reverb_damping": 0.5,
            "reverb_wet": 0.3,
            "reverb_dry": 0.7
        }
        
        # Apply effect
        processed = self.processor.apply_effects(
            self.test_audio,
            self.sample_rate,
            effects,
            parameters
        )
        
        # Check output
        self.assertEqual(processed.shape, self.test_audio.shape)
        self.assertFalse(np.array_equal(processed, self.test_audio))
    
    def test_apply_delay(self):
        """Test delay effect."""
        effects = ["Delay"]
        parameters = {
            "delay_time": 500,  # 500ms
            "delay_feedback": 0.3,
            "delay_mix": 0.5
        }
        
        # Apply effect
        processed = self.processor.apply_effects(
            self.test_audio,
            self.sample_rate,
            effects,
            parameters
        )
        
        # Check output
        self.assertEqual(processed.shape, self.test_audio.shape)
        self.assertFalse(np.array_equal(processed, self.test_audio))
    
    def test_apply_multiple_effects(self):
        """Test applying multiple effects."""
        effects = ["Reverb", "Delay", "Chorus"]
        parameters = {
            "reverb_room_size": 0.8,
            "delay_time": 500,
            "chorus_rate": 1.0,
            "chorus_depth": 0.5
        }
        
        # Apply effects
        processed = self.processor.apply_effects(
            self.test_audio,
            self.sample_rate,
            effects,
            parameters
        )
        
        # Check output
        self.assertEqual(processed.shape, self.test_audio.shape)
        self.assertFalse(np.array_equal(processed, self.test_audio))
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create test audio with varying amplitude
        audio = self.test_audio * 0.1  # Reduce volume
        
        # Normalize
        target_db = -20.0
        normalized = self.processor.normalize_audio(audio, target_db)
        
        # Check output
        self.assertEqual(normalized.shape, audio.shape)
        current_db = 20 * np.log10(np.sqrt(np.mean(normalized**2)))
        self.assertAlmostEqual(current_db, target_db, places=1)
    
    def test_time_stretch(self):
        """Test time stretching."""
        # Test various stretch rates
        rates = [0.5, 1.5, 2.0]
        
        for rate in rates:
            stretched = self.processor.time_stretch(self.test_audio, rate)
            
            # Check output
            expected_length = int(len(self.test_audio) / rate)
            self.assertAlmostEqual(
                len(stretched),
                expected_length,
                delta=self.sample_rate  # Allow for small differences
            )
    
    def test_pitch_shift(self):
        """Test pitch shifting."""
        # Test various semitone shifts
        semitones = [-12, -4, 4, 12]  # Octave down, major third down/up, octave up
        
        for n_steps in semitones:
            shifted = self.processor.pitch_shift(
                self.test_audio,
                self.sample_rate,
                n_steps
            )
            
            # Check output
            self.assertEqual(shifted.shape, self.test_audio.shape)
            self.assertFalse(np.array_equal(shifted, self.test_audio))

if __name__ == '__main__':
    unittest.main()
