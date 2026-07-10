# Driver Drowsiness Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://driver-drowsiness-detection-system.streamlit.app)
*(Add your live demo link above!)*

A real-time web application to detect driver drowsiness using a deep learning model and a webcam feed.

## How it Works

The application captures video frames from a live camera feed (or an uploaded video/image) and processes them in real-time. It uses OpenCV to detect faces and extract the eye region. This cropped region is then fed into a trained Convolutional Neural Network (CNN) which outputs a prediction score for drowsiness.

The application leverages Streamlit for the user interface and `streamlit-webrtc` to handle the live camera stream processing seamlessly in the browser. 

The live feed computes a rolling average over the most recent frames to smooth out minor glitches (like quick blinks) and ensures a reliable state evaluation. If the "Drowsy" prediction score surpasses the "Alert" score, the app displays a prominent warning to the user.

## Technology Stack
- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Live Video Streaming**: `streamlit-webrtc`
- **Image Processing**: OpenCV (Haar Cascades for face and eye detection)
- **Deep Learning Framework**: Keras / TensorFlow

## Model Details

The drowsiness detection model is based on the **MobileNetV2** architecture, which is highly efficient for real-time edge-device inference. 

It includes transfer learning from the base MobileNetV2 with custom Dense layers (1024 and 512 units) added on top for binary classification (`Drowsy` vs `Alert`).

### Dataset

The model was trained on the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) from Kaggle. 
The dataset contains a total of **41,793 images** split into two categories:
- **Drowsy**: Eyes closed or mostly closed.
- **Non Drowsy (Alert)**: Eyes open and focused.

### Accuracy
The model exhibits exceptional performance based on the training and evaluation splits:
- **Training Accuracy**: 99.99%
- **Test Accuracy**: 100.00%

> [!IMPORTANT]
> **Keep your face at an adequate distance from the camera and positioned near the center.** The application uses face and eye cascades to focus the model on the most critical regions, so having your face clearly visible guarantees the best accuracy.

> [!IMPORTANT]
> **Give the app a 1-2 second wait time after it starts detecting to test its results.** The system calculates a rolling average of consecutive frames to provide a stable prediction. It takes a brief moment for the initial buffer to fill up and provide an accurate state.

## Local Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
