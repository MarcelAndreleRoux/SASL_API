import gradio as gr
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import threading

# Create Flask app
flask_app = Flask(__name__)
CORS(flask_app)

# Load model
print("Loading model...")
model = tf.keras.models.load_model("model_1.keras")
actions = ['hello', 'how', 'you', 'good']
threshold = 0.5
print(f"✅ Model loaded! Actions: {actions}")

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sequence = data.get('sequence', [])
        
        if len(sequence) != 30:
            return jsonify({'error': f'Expected 30 frames, got {len(sequence)}'}), 400
        
        sequence_array = np.array(sequence)
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
        return jsonify({'error': str(e)}), 500

@flask_app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True, 
        'actions': actions
    })

# Gradio interface (required for Hugging Face Spaces)
def gradio_predict(sequence_text):
    """Simple Gradio interface for testing"""
    try:
        # This is just for testing in the Gradio UI
        return f"API is running! Use /predict endpoint for actual predictions.\nModel actions: {actions}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="Test Input", placeholder="API is ready"),
    outputs=gr.Textbox(label="Status"),
    title="Sign Language API",
    description="Your sign language prediction API is running. Use the /predict endpoint for actual predictions."
)

# Run Flask in background thread
def run_flask():
    flask_app.run(host="0.0.0.0", port=7860, debug=False)

if __name__ == "__main__":
    # Start Flask in background
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Launch Gradio interface
    iface.launch(server_port=7860, share=True)
