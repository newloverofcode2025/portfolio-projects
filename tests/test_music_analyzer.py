import unittest
import numpy as np
from src.music.music_analyzer import MusicAnalyzer
import librosa

class TestMusicAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.analyzer = MusicAnalyzer()
        cls.sample_rate = 22050
        cls.duration = 5  # seconds
        
        # Create test audio (C major chord)
        t = np.linspace(0, cls.duration, cls.duration * cls.sample_rate)
        c4 = np.sin(2 * np.pi * 261.63 * t)  # C4
        e4 = np.sin(2 * np.pi * 329.63 * t)  # E4
        g4 = np.sin(2 * np.pi * 392.00 * t)  # G4
        cls.test_audio = (c4 + e4 + g4) / 3.0
    
    def test_detect_key(self):
        """Test musical key detection."""
        # Test key detection
        key = self.analyzer.detect_key(self.test_audio, self.sample_rate)
        
        # Check output
        self.assertIsInstance(key, str)
        self.assertTrue(any(note in key for note in ['C', 'D', 'E', 'F', 'G', 'A', 'B']))
        self.assertTrue(any(mode in key for mode in ['major', 'minor']))
    
    def test_detect_tempo(self):
        """Test tempo detection."""
        # Create rhythmic test audio
        duration = 10
        bpm = 120
        beat_length = int(60 / bpm * self.sample_rate)
        beats = np.zeros(duration * self.sample_rate)
        for i in range(0, len(beats), beat_length):
            beats[i:i+100] = 1.0
        
        # Test tempo detection
        tempo = self.analyzer.detect_tempo(beats, self.sample_rate)
        
        # Check output
        self.assertIsInstance(tempo, float)
        self.assertTrue(100 <= tempo <= 140)  # Should be close to 120 BPM
    
    def test_recognize_chords(self):
        """Test chord recognition."""
        # Test chord recognition
        chord_data = self.analyzer.recognize_chords(self.test_audio, self.sample_rate)
        
        # Check output
        self.assertIsInstance(chord_data, dict)
        self.assertIn('timestamps', chord_data)
        self.assertIn('chords', chord_data)
        self.assertTrue(len(chord_data['timestamps']) > 0)
        self.assertTrue(len(chord_data['chords']) > 0)
        
        # Check if C major chord is detected
        self.assertTrue(any('C' in chord for chord in chord_data['chords']))
    
    def test_key_profiles(self):
        """Test key profile generation."""
        profiles = self.analyzer._get_key_profiles()
        
        # Check output
        self.assertIsInstance(profiles, list)
        self.assertEqual(len(profiles), 24)  # 12 major + 12 minor
        self.assertTrue(all(isinstance(p, np.ndarray) for p in profiles))
        self.assertTrue(all(len(p) == 12 for p in profiles))
    
    def test_chord_templates(self):
        """Test chord template generation."""
        templates = self.analyzer._get_chord_templates()
        
        # Check output
        self.assertIsInstance(templates, list)
        self.assertEqual(len(templates), 36)  # 12 notes * 3 chord types
        self.assertTrue(all(isinstance(t, np.ndarray) for t in templates))
        self.assertTrue(all(len(t) == 12 for t in templates))

if __name__ == '__main__':
    unittest.main()
