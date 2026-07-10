import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model_utils import predict_image

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        label, confidence = predict_image(file.read())
        return jsonify({
            'prediction': label,
            'confidence': f'{confidence:.2f}'
        })
    except Exception as exc:
        print(f'Error during prediction: {exc}')
        return jsonify({'error': str(exc)}), 500