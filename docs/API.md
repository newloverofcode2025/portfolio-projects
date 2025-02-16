# Audio Processing Tool API Documentation

## Table of Contents
1. [Speech Processing](#speech-processing)
2. [Music Analysis](#music-analysis)
3. [Audio Effects](#audio-effects)
4. [Visualization](#visualization)
5. [Utilities](#utilities)

## Speech Processing

### SpeechRecognizer

```python
from src.speech import SpeechRecognizer

recognizer = SpeechRecognizer()
```

#### Methods

##### transcribe
Convert speech to text.
```python
text = recognizer.transcribe("audio.wav")
```

##### detect_emotion
Detect emotions in speech.
```python
emotions = recognizer.detect_emotion("audio.wav")
# Returns: {"happy": 0.8, "sad": 0.1, ...}
```

##### diarize
Perform speaker diarization.
```python
segments = recognizer.diarize("audio.wav")
# Returns: [{"speaker": "A", "start": 0.0, "end": 1.5}, ...]
```

### VoiceDetector

```python
from src.speech import VoiceDetector

detector = VoiceDetector()
```

#### Methods

##### detect_voice_activity
Detect segments with voice activity.
```python
segments = detector.detect_voice_activity(audio_data, sample_rate)
# Returns: [{"start": 0.0, "end": 1.0, "has_voice": True, ...}, ...]
```

##### enroll_speaker
Enroll a new speaker in the database.
```python
detector.enroll_speaker(audio_data, sample_rate, "speaker_1")
```

##### verify_speaker
Verify if audio matches enrolled speaker.
```python
result = detector.verify_speaker(audio_data, sample_rate, "speaker_1")
# Returns: {"is_match": True, "similarity": 0.85}
```

## Music Analysis

### MusicAnalyzer

```python
from src.music import MusicAnalyzer

analyzer = MusicAnalyzer()
```

#### Methods

##### detect_key
Detect musical key of audio.
```python
key = analyzer.detect_key(audio_data, sample_rate)
# Returns: "C major"
```

##### detect_tempo
Detect tempo in BPM.
```python
tempo = analyzer.detect_tempo(audio_data, sample_rate)
# Returns: 120.5
```

##### recognize_chords
Recognize chord progressions.
```python
chords = analyzer.recognize_chords(audio_data, sample_rate)
# Returns: {"timestamps": [0.0, 1.0, ...], "chords": ["C", "Am", ...]}
```

### BeatTracker

```python
from src.music import BeatTracker

tracker = BeatTracker()
```

#### Methods

##### find_beats
Find beat positions and tempo.
```python
beats = tracker.find_beats(audio_data, sample_rate)
# Returns: {"tempo": 120.0, "beat_times": [0.0, 0.5, ...], ...}
```

##### analyze_rhythm
Analyze rhythmic properties.
```python
rhythm = tracker.analyze_rhythm(audio_data, sample_rate)
# Returns: {"main_tempo": 120.0, "rhythm_regularity": 0.85, ...}
```

##### find_groove
Analyze groove characteristics.
```python
groove = tracker.find_groove(audio_data, sample_rate)
# Returns: {"syncopation": 0.3, "swing_ratio": 1.5, ...}
```

## Audio Effects

### AudioProcessor

```python
from src.effects import AudioProcessor

processor = AudioProcessor()
```

#### Methods

##### apply_effects
Apply audio effects.
```python
processed = processor.apply_effects(
    audio_data,
    sample_rate,
    effects=["Reverb", "Delay"],
    parameters={"reverb_room_size": 0.8, "delay_time": 500}
)
```

##### normalize_audio
Normalize audio to target dB level.
```python
normalized = processor.normalize_audio(audio_data, target_db=-20.0)
```

##### time_stretch
Time stretch audio without changing pitch.
```python
stretched = processor.time_stretch(audio_data, rate=1.5)
```

##### pitch_shift
Shift pitch without changing tempo.
```python
shifted = processor.pitch_shift(audio_data, sample_rate, semitones=4)
```

## Visualization

### AudioVisualizer

```python
from src.visualization import AudioVisualizer

visualizer = AudioVisualizer()
```

#### Methods

##### plot_waveform
Create waveform visualization.
```python
fig = visualizer.plot_waveform(audio_data, sample_rate)
```

##### plot_spectrogram
Create spectrogram visualization.
```python
fig = visualizer.plot_spectrogram(audio_data, sample_rate)
```

##### plot_mfcc
Create MFCC visualization.
```python
fig = visualizer.plot_mfcc(audio_data, sample_rate)
```

##### plot_chromagram
Create chromagram visualization.
```python
fig = visualizer.plot_chromagram(audio_data, sample_rate)
```

## Utilities

### Audio Utilities

```python
from src.utils import audio_utils
```

#### Functions

##### load_audio
Load audio file.
```python
audio_data, sample_rate = audio_utils.load_audio("audio.wav")
```

##### save_audio
Save audio to file.
```python
audio_utils.save_audio("output.wav", audio_data, sample_rate)
```

##### convert_audio_format
Convert audio format.
```python
audio_utils.convert_audio_format("input.mp3", "output.wav")
```

##### split_audio
Split audio into segments.
```python
segments = audio_utils.split_audio(audio_data, sample_rate, segment_duration=5.0)
```

##### mix_audio
Mix two audio signals.
```python
mixed = audio_utils.mix_audio(audio1, audio2, weight1=0.6, weight2=0.4)
```

## Error Handling

All functions may raise the following exceptions:
- `ValueError`: Invalid parameter values
- `FileNotFoundError`: Audio file not found
- `RuntimeError`: Processing error

Example error handling:
```python
try:
    audio_data, sample_rate = audio_utils.load_audio("audio.wav")
except FileNotFoundError:
    print("Audio file not found")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Best Practices

1. Always check audio sample rate and format before processing
2. Use error handling for robust applications
3. Process audio in chunks for large files
4. Close file handles after use
5. Use GPU acceleration when available
6. Normalize audio before applying effects
7. Save intermediate results for long processing chains

## Contact & Support

For questions, bug reports, or feature requests, please contact:

- **Author**: Abhishek Banerjee
- **Email**: abhishekninja@yahoo.com
- **GitHub**: [newloverofcode2025](https://github.com/newloverofcode2025)

You can also:
1. Open an issue on [GitHub](https://github.com/newloverofcode2025/audio-processing-tool/issues)
2. Submit a pull request with improvements
3. Check the [documentation](https://github.com/newloverofcode2025/audio-processing-tool/docs) for updates
