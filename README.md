# SignDecode: Real-Time Sign Language Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-red.svg)](https://mediapipe.dev/)

## Overview

SignDecode is a computer vision application that performs real-time American Sign Language (ASL) recognition using skeletal hand tracking and neural network classification. The system processes webcam input to detect hand landmarks and classify gestures into alphanumeric characters (A-Z, 0-9).

**Key Technical Achievement**: By using skeletal coordinate data instead of raw pixel analysis, the system achieves 99% reduction in input dimensionality (63 features vs 4,096 pixels for 64x64 images) while maintaining 92% classification accuracy.

## Architecture

### System Pipeline

```
Camera Input → MediaPipe Hand Detection → Landmark Extraction (21 points × 3 coords) 
→ MLP Classifier → Temporal Smoothing → Output Display
```

### Technology Stack

- **Computer Vision**: OpenCV, Google MediaPipe
- **Machine Learning**: TensorFlow/Keras (Multi-Layer Perceptron)
- **Backend**: Flask (REST API)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Data Processing**: NumPy, Pandas

### Model Architecture

```
Input Layer:    63 features (21 hand landmarks × xyz coordinates)
Hidden Layer 1: 128 neurons, ReLU activation, 30% dropout
Hidden Layer 2: 64 neurons, ReLU activation, 30% dropout  
Hidden Layer 3: 64 neurons, ReLU activation
Output Layer:   36 neurons (A-Z, 0-9), Softmax activation
```

**Training Configuration**:
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 50
- Batch Size: 16
- Train/Test Split: 80/20

**Performance Metrics**:
- Training Accuracy: 92%
- Validation Accuracy: 88%
- Inference Latency: <50ms per frame
- Model Size: ~500KB

## Project Structure

```
SignDecode/
├── src/                        # Application source code
│   ├── app.py                 # Flask server, API endpoints
│   ├── model.py               # Model wrapper class
│   ├── utils.py               # Keypoint extraction utilities
│   ├── labels.py              # Class label mappings
│   ├── text_to_speech.py      # Audio output module
│   ├── static/                # Frontend assets (CSS, JS)
│   └── templates/             # HTML templates
├── training/                   # Model training pipeline
│   ├── collect_data.py        # Data collection utility
│   ├── train_model.py         # Model training script
│   └── dataset/               # Training data storage
├── models/                     # Trained model artifacts
│   └── sign_language_model.h5
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Webcam
- Modern web browser (Chrome/Firefox recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/thesakshidigg/SignDecode-Sign-Language-Recognition-.git
cd SignDecode-Sign-Language-Recognition-

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python run.py
```

Navigate to `http://localhost:5000` in your browser.

## Usage

### Running the Application

```bash
python run.py
```

The Flask server starts on port 5000. The web interface provides:
- Real-time hand tracking visualization
- ASL character recognition
- Text-to-speech output
- Interactive sign language game

### Training a Custom Model

**Step 1: Collect Training Data**
```bash
cd training
python collect_data.py
```
Follow prompts to record hand gestures for each character. Data is saved to `training/dataset/sign_data.csv`.

**Step 2: Train Model**
```bash
python train_model.py
```
Trains the neural network and saves the model to `models/sign_language_model.h5`.

## Technical Implementation Details

### Hand Landmark Detection

MediaPipe detects 21 anatomical landmarks per hand:
- Wrist (1 point)
- Thumb (4 points)
- Index, Middle, Ring, Pinky fingers (4 points each)

Each landmark provides (x, y, z) coordinates, normalized to [0, 1] range.

### Classification Approach

The system uses a Multi-Layer Perceptron (MLP) rather than a Convolutional Neural Network (CNN) because:
1. **Input Type**: Structured coordinate data (not spatial image data)
2. **Efficiency**: 63 input features vs 4,096+ for image-based approaches
3. **Invariance**: Skeletal data is inherently invariant to lighting, background, and skin tone

### Temporal Smoothing

A 15-frame consistency filter prevents prediction flickering:
```python
if predicted_label == last_predicted_label:
    frame_count += 1
    if frame_count >= THRESHOLD_FRAMES:
        output_text += predicted_character
```

## API Endpoints

### POST /process_frame
Processes a single video frame for hand detection and classification.

**Request**:
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response**:
```json
{
  "prediction": "A",
  "image": "data:image/jpeg;base64,..."
}
```

### GET /status
Returns current recognized text.

### POST /clear_text
Clears the output text buffer.

## Performance Considerations

- **Latency**: Sub-50ms inference time enables real-time processing at 30 FPS
- **Accuracy Trade-off**: Temporal smoothing reduces false positives at the cost of ~500ms recognition delay
- **Scalability**: Lightweight model enables deployment on edge devices without GPU

## Future Enhancements

- Support for dynamic gestures (word-level recognition using LSTM/GRU)
- Multi-language sign language support (BSL, ISL, etc.)
- Mobile deployment using TensorFlow Lite
- Data augmentation for improved robustness
- REST API for third-party integration

## Contributing

Contributions are welcome. Please follow standard Git workflow:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Contact

**Sakshi Diggikar**  
GitHub: [@thesakshidigg](https://github.com/thesakshidigg)

## Acknowledgments

- Google MediaPipe for hand tracking framework
- TensorFlow team for ML infrastructure
- ASL dataset contributors
