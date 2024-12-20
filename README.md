# Face-Mask-Detection-System
# Face Mask Detection System

## Overview
This project demonstrates a Face Mask Detection System using Convolutional Neural Networks (CNN). The system detects whether a person is wearing a face mask or not in real-time using computer vision techniques. This application is especially relevant in promoting safety and enforcing mask compliance in public spaces.

## Features
- Real-time face detection and mask classification.
- High accuracy due to the use of CNN-based deep learning.
- Supports video stream processing (webcam or external video).
- Visual feedback with bounding boxes and labels indicating mask status.

## Technologies Used
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Programming Language**: Python
- **Libraries**: OpenCV, NumPy, Matplotlib, Scikit-learn
- **Dataset**: [Face Mask Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) from Kaggle

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/face-mask-detection.git
   cd face-mask-detection
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # For Windows, use `venv\Scripts\activate`
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
Download the [Face Mask Detection Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) and place it in the `dataset/` directory.

The dataset should have the following structure:
```
face-mask-detection/
├── dataset/
│   ├── with_mask/
│   ├── without_mask/
```

## Model Training
1. Preprocess the dataset:
   - Resize images to 224x224 pixels.
   - Normalize pixel values to [0, 1].
2. Train the model:
   ```bash
   python train_model.py
   ```
   This script:
   - Loads and preprocesses the dataset.
   - Trains a CNN model (e.g., MobileNetV2 or a custom architecture).
   - Saves the trained model as `mask_detector.model`.

## Running the Detection System
1. Run the detection script:
   ```bash
   python detect_mask_video.py
   ```
2. The system will:
   - Access the webcam (or process a video file if specified).
   - Detect faces using OpenCV's Haar Cascade or DNN-based face detector.
   - Classify each face as "Mask" or "No Mask".
   - Display the output with bounding boxes and labels.

## Results
The system achieves high accuracy on the test dataset, demonstrating robust performance in real-world scenarios.

## Future Enhancements
- Integrate with IoT devices for automated alerts.
- Optimize the model for edge devices.
- Add support for detecting improper mask usage.

## Contribution
Contributions are welcome! Please submit a pull request or open an issue for any bugs or improvements.

## License
This project is licensed under the [MIT License](LICENSE).

