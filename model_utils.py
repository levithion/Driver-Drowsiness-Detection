import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'numpy')

import io

import keras
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / 'drowsy_detect.keras'


@lru_cache(maxsize=1)
def get_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as exc:
        print(f'Error preprocessing image: {exc}')
        return None


def predict_image(image_bytes):
    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        raise ValueError('Could not process image')

    prediction = np.asarray(get_model().predict(processed_image, verbose=0))[0]

    if prediction.ndim == 0 or prediction.size == 1:
        score = float(prediction.reshape(-1)[0])
        label = 'Drowsy' if score > 0.5 else 'Alert'
        confidence = score if label == 'Drowsy' else 1 - score
        return label, confidence

    predicted_index = int(np.argmax(prediction))
    label = 'Drowsy' if predicted_index == 1 else 'Alert'
    confidence = float(prediction[predicted_index])
    return label, confidence