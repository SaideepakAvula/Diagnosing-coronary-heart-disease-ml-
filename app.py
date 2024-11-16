from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from io import BytesIO
import joblib
import base64

app = Flask(__name__)

# Load SVM and Random Forest models
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Function to preprocess uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input shape
    img = np.array(img) / 255.0   # Normalize pixel values
    return img

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['image']
    
    try:
        img = Image.open(file)
        img = preprocess_image(img)
        
        # Reshape image to match model input shape (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        
        # Predict using both SVM and Random Forest models
        svm_pred = svm_model.predict(img.reshape(1, -1))[0]
        rf_pred = rf_model.predict(img.reshape(1, -1))[0]
        
        # Return the majority prediction
        if svm_pred == rf_pred:
            disease = svm_pred
        else:
            # Use SVM prediction by default if there's a tie
            disease = svm_pred
        
        return jsonify({'predicted_disease': int(disease)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to render upload form
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
