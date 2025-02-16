# AI-Powered Image Analysis Tool 🔍

A sophisticated Python application that leverages state-of-the-art AI models for advanced image analysis, processing, and enhancement. This tool combines multiple AI capabilities into a user-friendly interface, making powerful computer vision technology accessible through a clean and intuitive Streamlit interface.

## ✨ Features

### 🎯 Object Detection & Recognition
- State-of-the-art YOLOv8 integration for accurate object detection
- Multiple model sizes (nano to extra large) for speed/accuracy tradeoffs
- GPU acceleration for real-time processing
- Support for custom object detection models

### 👤 Face Analysis
- Advanced facial landmark detection using MediaPipe
- Face recognition with FaceNet
- Custom face database support for identity matching
- Facial feature analysis and mesh visualization

### 🖼️ Image Enhancement
- AI-powered super-resolution using Real-ESRGAN
- Intelligent noise reduction with GPU acceleration
- Adaptive contrast enhancement
- Smart sharpening algorithms
- Background removal capabilities

### ⚡ Performance Features
- CUDA GPU acceleration for all major operations
- Batch processing with progress tracking
- Configurable processing pipeline
- Memory-efficient large image handling

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-analysis-tool.git
cd image-analysis-tool
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```

The application will open in your default web browser with an intuitive interface for:
- Uploading images
- Selecting analysis tasks
- Viewing results
- Downloading processed images

## 📖 Documentation

### Project Structure
```
image-analysis-tool/
├── app.py                  # Main Streamlit application
├── src/
│   ├── detection/         # Object detection modules
│   ├── face_analysis/     # Face analysis components
│   ├── enhancement/       # Image enhancement features
│   ├── utils/            # Utility functions
│   └── models/           # Model management
├── tests/                # Test suite
├── docs/                # Documentation
├── examples/            # Example notebooks
└── sample_images/       # Example images
```

### API Documentation
Detailed API documentation is available in the [docs/API.md](docs/API.md) file.

### Example Notebooks
- [Basic Usage](examples/1_basic_usage.ipynb)
- [Batch Processing](examples/2_batch_processing.ipynb)

## 🛠️ Advanced Usage

### GPU Acceleration
The tool automatically detects and utilizes available CUDA GPUs. Configure GPU settings in `src/config.py`:
```python
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Custom Models
1. Object Detection:
```python
from src.detection import ObjectDetector
detector = ObjectDetector(model_name="custom_model.pt", custom_model=True)
```

2. Face Recognition:
```python
from src.face_analysis import FaceAnalyzer
analyzer = FaceAnalyzer()
analyzer.load_face_database("path/to/custom_database.pt")
```

### Batch Processing
```python
from src.utils.batch_processor import BatchProcessor
processor = BatchProcessor(max_workers=4)
results = processor.process_batch(images, process_fn=detector.detect)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- MediaPipe by Google
- FaceNet PyTorch implementation
- Real-ESRGAN for super-resolution
- OpenCV community

## 📧 Contact

Abhishek Banerjee - [Your Email or Contact Information]

Project Link: [https://github.com/yourusername/image-analysis-tool](https://github.com/yourusername/image-analysis-tool)
