import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from jinja2.exceptions import TemplateNotFound

# Initialize Flask app with template_folder set to current directory
app = Flask(__name__, template_folder='.')

# Load the trained model
model_path_options = ['facial_emotion_detection_model_updated.h5', 'facial_emotion_detection_model.h5', 'best_model.h5']
model = None
for path in model_path_options:
    if os.path.exists(path):
        try:
            model = load_model(path)
            print(f"Model loaded successfully from {path}")
            break
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
    else:
        print(f"Model file '{path}' not found.")
if model is None:
    print("No valid model file found. Please run the training script to generate 'facial_emotion_detection_model.h5' or ensure a compatible model exists.")

# Define class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion detection function
def detect_emotion(img_path):
    if model is None:
        return None, "Model not loaded. Please ensure a valid model file exists in the project directory."
    try:
        img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = round(prediction[0][predicted_index] * 100, 2)

        return predicted_class, confidence
    except Exception as e:
        return None, f"Error processing image: {e}"

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', error='No file uploaded!', image_path=None, emotion=None, confidence=None)
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error='No file selected!', image_path=None, emotion=None, confidence=None)

            if file:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Detect emotion
                emotion, confidence = detect_emotion(file_path)
                if emotion is None:
                    return render_template('index.html', error=confidence, image_path=None, emotion=None, confidence=None)

                return render_template('index.html', image_path=file_path, emotion=emotion, confidence=confidence, error=None)

        return render_template('index.html', error=None, image_path=None, emotion=None, confidence=None)
    except TemplateNotFound:
        return "Template 'index.html' not found in the project directory. Please ensure it exists alongside 'main.py'.", 500

if __name__ == '__main__':
    app.run(debug=True)