
import cv2
import os
import numpy as np
import tensorflow as tf
import h5py
import json
import base64
from flask import Flask, render_template, request, jsonify
import io
from PIL import Image

app = Flask(__name__)

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Model and labels file paths - use absolute paths for deployment
model_path = os.path.join(os.path.dirname(__file__), 'Resources/Model/keras_model.h5')
labels_path = os.path.join(os.path.dirname(__file__), 'Resources/Model/labels.txt')

# Global variables
model = None
class_labels = []
dustbin_colors = {
    'Biodegradable waste': 'Green Dustbin',
    'Non bio degradable waste': 'Blue Dustbin',
    'Hazardeous Waste': 'Red Dustbin'
}

def initialize_model():
    global model, class_labels
    
    # File existence check
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return False

    # Load labels from the labels.txt file
    try:
        with open(labels_path, 'r') as f:
            class_labels = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_labels)} class labels: {class_labels}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return False

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Function to find the matching dustbin category
def get_dustbin_color(class_name):
    # Exact match
    if class_name in dustbin_colors:
        return dustbin_colors[class_name]
    
    # Case-insensitive match
    class_name_lower = class_name.lower()
    for key in dustbin_colors:
        if key.lower() == class_name_lower:
            return dustbin_colors[key]
    
    # Partial matches for safety
    if 'biodegrad' in class_name_lower and 'non' not in class_name_lower:
        return 'Green Dustbin'
    elif 'non' in class_name_lower and 'bio' in class_name_lower:
        return 'Blue Dustbin'
    elif 'hazard' in class_name_lower:
        return 'Red Dustbin'
    
    return 'Unknown'

def classify_image(image_data):
    try:
        # Convert base64 image to OpenCV format
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {
                'success': False,
                'message': 'Invalid image data'
            }
        
        # Prepare image for the model (resize, normalize)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        normalized_img_array = (img_array / 127.5) - 1

        # Get prediction from the model
        predictions = model.predict(normalized_img_array, verbose=0)
        index = np.argmax(predictions)
        confidence = predictions[0][index]
        
        if confidence > 0.5:  # Lowered threshold for better UX
            # Extract class name from label
            if ' ' in class_labels[index]:
                class_name = class_labels[index].split(' ', 1)[1]
            else:
                class_name = class_labels[index]
            
            # Get dustbin color
            dustbin_color = get_dustbin_color(class_name)
            
            return {
                'success': True,
                'category': class_name,
                'dustbin': dustbin_color,
                'confidence': float(confidence)
            }
        else:
            return {
                'success': False,
                'message': 'Low confidence. Please try with a clearer image.'
            }
            
    except Exception as e:
        print(f"Classification error: {e}")
        return {
            'success': False,
            'message': f'Error during classification: {str(e)}'
        }

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/voice-search')
def voice_search():
    return render_template('voice.html')

@app.route('/speech', methods=['GET'])
def speak():
    return render_template('voice.html')

@app.route('/classify', methods=['POST'])
def classify():
    if model is None:
        return jsonify({
            'success': False,
            'message': 'Model not loaded. Please try again later.'
        })
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'No image data received.'
            })
        
        image_data = data.get('image', '')
        result = classify_image(image_data)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in classify endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        })

@app.route('/upload', methods=['POST'])
def upload_classify():
    """Alternative endpoint for file uploads"""
    if model is None:
        return jsonify({
            'success': False,
            'message': 'Model not loaded.'
        })
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        # Convert uploaded file to base64
        image_data = base64.b64encode(file.read()).decode('utf-8')
        result = classify_image(image_data)
        return jsonify(result)
        
    except Exception as e:
        print(f"Upload classification error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        })

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'class_labels': class_labels,
        'status': 'ready'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# Initialize the model when the app starts
if initialize_model():
    print("Application initialized successfully.")
else:
    print("Application failed to initialize. Check model files.")

if __name__ == '__main__':
    # For production, use waitress or gunicorn
    app.run(debug=False)