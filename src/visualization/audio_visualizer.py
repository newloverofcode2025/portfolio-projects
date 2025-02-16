import numpy as np
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List

class AudioVisualizer:
    def __init__(self):
        """Initialize audio visualization components."""
        self.color_scale = px.colors.sequential.Viridis
        
    def plot_waveform(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        title: str = "Waveform"
    ) -> go.Figure:
        """
        Create an interactive waveform plot.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            title: Plot title
            
        Returns:
            Plotly figure
        """
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=audio_data,
            mode='lines',
            name='Amplitude',
            line=dict(color='rgb(31, 119, 180)')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode='x',
            template='plotly_white'
        )
        
        return fig
    
    def plot_spectrogram(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        title: str = "Spectrogram"
    ) -> go.Figure:
        """
        Create an interactive spectrogram plot.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Compute spectrogram
        D = librosa.stft(audio_data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=S_db,
            colorscale=self.color_scale,
            colorbar=dict(title="dB")
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Frequency",
            template='plotly_white'
        )
        
        # Update axes
        time_coords = librosa.times_like(S_db)
        freq_coords = librosa.fft_frequencies(sr=sample_rate)
        
        fig.update_xaxes(ticktext=time_coords[::100], tickvals=list(range(0, S_db.shape[1], 100)))
        fig.update_yaxes(ticktext=freq_coords[::50], tickvals=list(range(0, S_db.shape[0], 50)))
        
        return fig
    
    def plot_mfcc(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        title: str = "MFCC"
    ) -> go.Figure:
        """
        Create an interactive MFCC plot.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Compute MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=mfccs,
            colorscale=self.color_scale,
            colorbar=dict(title="Magnitude")
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="MFCC Coefficient",
            template='plotly_white'
        )
        
        return fig
    
    def plot_chromagram(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        title: str = "Chromagram"
    ) -> go.Figure:
        """
        Create an interactive chromagram plot.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=chroma,
            colorscale=self.color_scale,
            colorbar=dict(title="Magnitude")
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Pitch Class",
            template='plotly_white'
        )
        
        # Update y-axis labels
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        fig.update_yaxes(ticktext=pitch_classes, tickvals=list(range(12)))
        
        return fig
    
    def plot_onset_strength(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        title: str = "Onset Strength"
    ) -> go.Figure:
        """
        Create an interactive onset strength plot.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        times = librosa.times_like(onset_env, sr=sample_rate)
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=onset_env,
            mode='lines',
            name='Onset Strength',
            line=dict(color='rgb(31, 119, 180)')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Strength",
            template='plotly_white'
        )
        
        return fig
