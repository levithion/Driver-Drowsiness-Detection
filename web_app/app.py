import sys
from pathlib import Path

from flask import Flask, request, jsonify, render_template

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

from model_utils import predict_image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            label, confidence = predict_image(file.read())

            return jsonify({
                'prediction': label,
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)