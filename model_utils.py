import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'numpy')

import io

import cv2
import keras
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / 'drowsy_detect.keras'
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
except Exception:
    FACE_CASCADE = None
    EYE_CASCADE = None


def crop_focus_region_from_array(image_array):
    if FACE_CASCADE is None:
        return image_array, 'full_frame'

    image_uint8 = image_array.astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return image_array, 'full_frame'

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    pad_x = max(int(w * 0.15), 10)
    pad_y = max(int(h * 0.2), 10)

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, image_array.shape[1])
    y2 = min(y + h + pad_y, image_array.shape[0])

    face_crop = image_uint8[y1:y2, x1:x2]

    if EYE_CASCADE is None or face_crop.size == 0:
        return face_crop if face_crop.size != 0 else image_uint8, 'face'

    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    face_gray = cv2.equalizeHist(face_gray)
    eyes = EYE_CASCADE.detectMultiScale(
        face_gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(20, 20),
    )

    if len(eyes) == 0:
        upper_height = max(int(face_crop.shape[0] * 0.7), 1)
        return face_crop[:upper_height, :], 'face'

    eye_boxes = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
    eye_x1 = min(box[0] for box in eye_boxes)
    eye_y1 = min(box[1] for box in eye_boxes)
    eye_x2 = max(box[0] + box[2] for box in eye_boxes)
    eye_y2 = max(box[1] + box[3] for box in eye_boxes)

    crop_x1 = max(eye_x1 - int(face_crop.shape[1] * 0.15), 0)
    crop_y1 = max(eye_y1 - int(face_crop.shape[0] * 0.15), 0)
    crop_x2 = min(eye_x2 + int(face_crop.shape[1] * 0.15), face_crop.shape[1])
    crop_y2 = min(eye_y2 + int(face_crop.shape[0] * 0.55), face_crop.shape[0])

    focus_crop = face_crop[crop_y1:crop_y2, crop_x1:crop_x2]
    return focus_crop if focus_crop.size != 0 else face_crop, 'eyes'


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


def preprocess_array(image_array):
    try:
        face_array, _crop_mode = crop_focus_region_from_array(image_array)
        img = Image.fromarray(face_array.astype(np.uint8)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as exc:
        print(f'Error preprocessing array: {exc}')
        return None


def predict_image(image_bytes):
    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        raise ValueError('Could not process image')

    prediction = np.asarray(get_model().predict(processed_image, verbose=0))[0]
    return prediction_from_raw(prediction)


def predict_array(image_array):
    processed_image = preprocess_array(image_array)
    if processed_image is None:
        raise ValueError('Could not process image')

    prediction = np.asarray(get_model().predict(processed_image, verbose=0))[0]
    return prediction_from_raw(prediction)


def prediction_from_raw(prediction):
    if prediction.ndim == 0 or prediction.size == 1:
        score = float(prediction.reshape(-1)[0])
        label = 'Alert' if score >= 0.5 else 'Drowsy'
        confidence = score if label == 'Alert' else 1 - score
        return label, confidence

    drowsy_score = float(prediction[0])
    alert_score = float(prediction[1])
    label = 'Alert' if alert_score >= drowsy_score else 'Drowsy'
    confidence = alert_score if label == 'Alert' else drowsy_score
    return label, confidence


def predict_scores(image_array):
    processed_image = preprocess_array(image_array)
    if processed_image is None:
        raise ValueError('Could not process image')

    prediction = np.asarray(get_model().predict(processed_image, verbose=0))[0]

    if prediction.ndim == 0 or prediction.size == 1:
        score = float(prediction.reshape(-1)[0])
        return {'alert': score, 'drowsy': 1 - score}

    return {'alert': float(prediction[1]), 'drowsy': float(prediction[0])}


def predict_frame(image_array):
    face_array, crop_mode = crop_focus_region_from_array(image_array)
    img = Image.fromarray(face_array.astype(np.uint8)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    processed_image = np.expand_dims(img_array, axis=0)

    prediction = np.asarray(get_model().predict(processed_image, verbose=0))[0]
    label, confidence = prediction_from_raw(prediction)

    if prediction.ndim == 0 or prediction.size == 1:
        score = float(prediction.reshape(-1)[0])
        scores = {'alert': score, 'drowsy': 1 - score}
    else:
        scores = {'alert': float(prediction[1]), 'drowsy': float(prediction[0])}

    return {
        'label': label,
        'confidence': confidence,
        'scores': scores,
        'crop_mode': crop_mode,
    }