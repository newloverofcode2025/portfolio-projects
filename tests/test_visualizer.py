import unittest
import numpy as np
import plotly.graph_objects as go
from src.visualization.audio_visualizer import AudioVisualizer

class TestAudioVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.visualizer = AudioVisualizer()
        cls.sample_rate = 22050
        cls.duration = 3  # seconds
        
        # Create test audio (mixture of frequencies)
        t = np.linspace(0, cls.duration, cls.duration * cls.sample_rate)
        f1, f2, f3 = 440, 880, 1320  # A4, A5, E6
        cls.test_audio = (np.sin(2 * np.pi * f1 * t) +
                         np.sin(2 * np.pi * f2 * t) * 0.5 +
                         np.sin(2 * np.pi * f3 * t) * 0.25)
    
    def test_plot_waveform(self):
        """Test waveform visualization."""
        # Generate plot
        fig = self.visualizer.plot_waveform(
            self.test_audio,
            self.sample_rate,
            "Test Waveform"
        )
        
        # Check output
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One trace
        self.assertEqual(fig.data[0].mode, 'lines')
        self.assertEqual(len(fig.data[0].x), len(self.test_audio))
    
    def test_plot_spectrogram(self):
        """Test spectrogram visualization."""
        # Generate plot
        fig = self.visualizer.plot_spectrogram(
            self.test_audio,
            self.sample_rate,
            "Test Spectrogram"
        )
        
        # Check output
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One heatmap
        self.assertIsInstance(fig.data[0], go.Heatmap)
    
    def test_plot_mfcc(self):
        """Test MFCC visualization."""
        # Generate plot
        fig = self.visualizer.plot_mfcc(
            self.test_audio,
            self.sample_rate,
            "Test MFCC"
        )
        
        # Check output
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One heatmap
        self.assertIsInstance(fig.data[0], go.Heatmap)
        self.assertEqual(fig.data[0].z.shape[0], 13)  # 13 MFCC coefficients
    
    def test_plot_chromagram(self):
        """Test chromagram visualization."""
        # Generate plot
        fig = self.visualizer.plot_chromagram(
            self.test_audio,
            self.sample_rate,
            "Test Chromagram"
        )
        
        # Check output
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One heatmap
        self.assertIsInstance(fig.data[0], go.Heatmap)
        self.assertEqual(fig.data[0].z.shape[0], 12)  # 12 pitch classes
    
    def test_plot_onset_strength(self):
        """Test onset strength visualization."""
        # Generate plot
        fig = self.visualizer.plot_onset_strength(
            self.test_audio,
            self.sample_rate,
            "Test Onset Strength"
        )
        
        # Check output
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # One trace
        self.assertEqual(fig.data[0].mode, 'lines')

if __name__ == '__main__':
    unittest.main()
