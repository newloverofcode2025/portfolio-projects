import streamlit as st
import os
import tempfile
from src.speech import SpeechRecognizer
from src.music import GenreClassifier, MusicAnalyzer
from src.effects import AudioProcessor
from src.visualization import AudioVisualizer
from src.utils.audio_utils import load_audio, save_audio

st.set_page_config(
    page_title="Advanced Audio Processing Tool",
    page_icon="🎵",
    layout="wide"
)

def main():
    st.title("🎵 Advanced Audio Processing Tool")
    
    # Sidebar for task selection
    task = st.sidebar.selectbox(
        "Select Task",
        ["Speech Processing", "Music Analysis", "Audio Effects", "Visualization"]
    )
    
    # File uploader
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "flac"])
    
    if audio_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            temp_path = tmp_file.name
        
        # Load audio for processing
        audio_data, sample_rate = load_audio(temp_path)
        
        try:
            if task == "Speech Processing":
                speech_processing(temp_path, audio_data, sample_rate)
            elif task == "Music Analysis":
                music_analysis(temp_path, audio_data, sample_rate)
            elif task == "Audio Effects":
                audio_effects(temp_path, audio_data, sample_rate)
            elif task == "Visualization":
                audio_visualization(audio_data, sample_rate)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

def speech_processing(file_path, audio_data, sample_rate):
    st.header("🎤 Speech Processing")
    
    # Initialize speech recognizer
    recognizer = SpeechRecognizer()
    
    # Task selection
    subtask = st.selectbox(
        "Select Speech Processing Task",
        ["Speech-to-Text", "Speaker Diarization", "Emotion Detection", "Noise Reduction"]
    )
    
    if subtask == "Speech-to-Text":
        with st.spinner("Transcribing audio..."):
            text = recognizer.transcribe(file_path)
            st.text_area("Transcription", text, height=200)
            
    elif subtask == "Speaker Diarization":
        with st.spinner("Analyzing speakers..."):
            segments = recognizer.diarize(file_path)
            for segment in segments:
                st.write(f"Speaker {segment['speaker']}: {segment['start']:.2f}s - {segment['end']:.2f}s")
                
    elif subtask == "Emotion Detection":
        with st.spinner("Detecting emotions..."):
            emotions = recognizer.detect_emotion(file_path)
            st.bar_chart(emotions)
            
    elif subtask == "Noise Reduction":
        with st.spinner("Reducing noise..."):
            cleaned_audio = recognizer.reduce_noise(audio_data, sample_rate)
            st.audio(cleaned_audio, sample_rate=sample_rate)

def music_analysis(file_path, audio_data, sample_rate):
    st.header("🎼 Music Analysis")
    
    # Initialize music analyzer
    analyzer = MusicAnalyzer()
    classifier = GenreClassifier()
    
    # Task selection
    subtask = st.selectbox(
        "Select Music Analysis Task",
        ["Genre Classification", "Key Detection", "Tempo Analysis", "Chord Recognition"]
    )
    
    if subtask == "Genre Classification":
        with st.spinner("Classifying genre..."):
            genre_probs = classifier.predict_proba(file_path)
            st.bar_chart(genre_probs)
            
    elif subtask == "Key Detection":
        with st.spinner("Detecting musical key..."):
            key = analyzer.detect_key(audio_data, sample_rate)
            st.write(f"Detected Key: {key}")
            
    elif subtask == "Tempo Analysis":
        with st.spinner("Analyzing tempo..."):
            tempo = analyzer.detect_tempo(audio_data, sample_rate)
            st.write(f"Tempo: {tempo:.1f} BPM")
            
    elif subtask == "Chord Recognition":
        with st.spinner("Recognizing chords..."):
            chords = analyzer.recognize_chords(audio_data, sample_rate)
            st.line_chart(chords)

def audio_effects(file_path, audio_data, sample_rate):
    st.header("🎛️ Audio Effects")
    
    # Initialize audio processor
    processor = AudioProcessor()
    
    # Effect selection
    effects = st.multiselect(
        "Select Effects",
        ["Reverb", "Delay", "Distortion", "Chorus", "EQ"]
    )
    
    # Effect parameters
    params = {}
    for effect in effects:
        st.subheader(effect + " Parameters")
        if effect == "Reverb":
            params["reverb_room_size"] = st.slider("Room Size", 0.0, 1.0, 0.5)
            params["reverb_damping"] = st.slider("Damping", 0.0, 1.0, 0.5)
        elif effect == "Delay":
            params["delay_time"] = st.slider("Delay Time (ms)", 0, 1000, 200)
            params["delay_feedback"] = st.slider("Feedback", 0.0, 1.0, 0.3)
    
    if st.button("Apply Effects"):
        with st.spinner("Applying effects..."):
            processed_audio = processor.apply_effects(
                audio_data,
                sample_rate,
                effects=effects,
                parameters=params
            )
            st.audio(processed_audio, sample_rate=sample_rate)

def audio_visualization(audio_data, sample_rate):
    st.header("📊 Audio Visualization")
    
    # Initialize visualizer
    visualizer = AudioVisualizer()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization",
        ["Waveform", "Spectrogram", "MFCC", "Chromagram"]
    )
    
    if viz_type == "Waveform":
        fig = visualizer.plot_waveform(audio_data, sample_rate)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Spectrogram":
        fig = visualizer.plot_spectrogram(audio_data, sample_rate)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "MFCC":
        fig = visualizer.plot_mfcc(audio_data, sample_rate)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Chromagram":
        fig = visualizer.plot_chromagram(audio_data, sample_rate)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
