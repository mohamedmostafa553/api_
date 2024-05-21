import os
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# Set environment variable to disable oneDNN if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load the pre-trained model
model_path = "F:/project/saved models.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.form

        # Extract features from the request data
        features = np.array([data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], 
                             data['fbs'], data['restecg'], data['thalach'], data['exang'], 
                             data['oldpeak'], data['slope'], data['ca'], data['thal'],], dtype=float).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Interpret prediction
        if prediction[0][0] > 0.5:
            result = "you have heart disease"
        else:
            result = "you do not have heart disease"

        # Return prediction
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
