# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://your-frontend-domain.com"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
API_KEY = os.environ.get('API_KEY', '1234')  # Change in production

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract additional features from text"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = np.zeros((len(X), 5))
        
        for i, text in enumerate(X):
            features[i, 0] = len(str(text).split())  # word count
            features[i, 1] = bool(re.search(r'\d{1,2}/\d{1,2}|\d{4}', str(text)))  # has date
            features[i, 2] = bool(re.search(r'urgent|asap|immediate|critical', str(text).lower()))  # urgent words
            features[i, 3] = bool(re.search(r'meeting|call|conference', str(text).lower()))  # meeting words
            features[i, 4] = bool(re.search(r'deadline|due|by|until', str(text).lower()))  # deadline words
            
        return features

# Load the model
try:
    model = joblib.load('improved_priority_classifier.joblib')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        if api_key and api_key.startswith('Bearer '):
            api_key = api_key.split('Bearer ')[1]
        
        if not api_key or api_key != API_KEY:
            return jsonify({
                'error': 'Invalid or missing API key',
                'status': 'error'
            }), 401
        return f(*args, **kwargs)
    return decorated

def validate_request(data):
    """Validate request data"""
    if not isinstance(data, dict):
        return False, "Invalid request format"
    
    if 'task' not in data:
        return False, "Missing 'task' field"
    
    if not isinstance(data['task'], str):
        return False, "Task must be a string"
    
    if len(data['task'].strip()) == 0:
        return False, "Task cannot be empty"
    
    if len(data['task']) > 500:  # Limit task length
        return False, "Task description too long"
    
    return True, None

@app.route('/api/predict', methods=['POST'])
@require_api_key
def predict_priority():
    try:
        # Get request data
        data = request.get_json()
        
        # Validate request
        is_valid, error_message = validate_request(data)
        if not is_valid:
            return jsonify({
                'error': error_message,
                'status': 'error'
            }), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'status': 'error'
            }), 503
        
        # Make prediction
        task = data['task']
        prediction = model.predict([task])[0]
        probabilities = model.predict_proba([task])[0]
        confidence = float(max(probabilities) * 100)
        
        # Log prediction (you might want to store this in a database)
        logger.info(f"Prediction made - Task: {task}, Priority: {prediction}, Confidence: {confidence:.2f}%")
        
        # Return response
        return jsonify({
            'task': task,
            'priority': prediction,
            'confidence': f"{confidence:.2f}%",
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)