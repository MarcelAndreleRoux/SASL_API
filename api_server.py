from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)  # Allow requests from Next.js

# Load your model
print("Loading model...")
model = tf.keras.models.load_model("model_1.keras")
actions = ['hello', 'how', 'you', 'good']
threshold = 0.5
print(f"✅ Model loaded! Actions: {actions}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sequence = data.get('sequence', [])
        
        if len(sequence) != 30:
            return jsonify({'error': f'Expected 30 frames, got {len(sequence)}'}), 400
        
        # Convert to numpy array and make prediction
        sequence_array = np.array(sequence)
        
        # Make prediction
        res = model.predict(np.expand_dims(sequence_array, axis=0), verbose=0)[0]
        predicted_class = np.argmax(res)
        confidence = float(res[predicted_class])
        
        if confidence > threshold:
            predicted_action = actions[predicted_class]
        else:
            predicted_action = "uncertain"
        
        return jsonify({
            'prediction': predicted_action,
            'confidence': confidence,
            'all_predictions': {action: float(score) for action, score in zip(actions, res)}
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True, 
        'actions': actions
    })

if __name__ == '__main__':
    # Get port from environment variable (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)