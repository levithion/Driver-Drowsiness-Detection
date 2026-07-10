from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from pathlib import Path

# Load the model
BASE_DIR = Path(__file__).resolve().parent
model = tf.keras.models.load_model(BASE_DIR.parent / 'drowsy_detect.keras')
print("Model loaded successfully!")

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))  # Adjust size according to your model's input requirements
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

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
            img_bytes = file.read()
            processed_image = preprocess_image(img_bytes)

            if processed_image is None:
                return jsonify({'error': 'Could not process image'}), 500

            prediction = model.predict(processed_image)
            score = float(prediction[0][0])
            label = "Drowsy" if score > 0.5 else "Alert"
            confidence = score if label == "Drowsy" else 1 - score

            return jsonify({
                'prediction': label,
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)