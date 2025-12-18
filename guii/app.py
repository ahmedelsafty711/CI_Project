
import sys
import os
import numpy as np
from flask import Flask, jsonify, render_template, request

# Add project root to Python path to import the 'lib' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lib.network import Network
from lib.layers import Dense
from lib.activations import Tanh, ReLU, Sigmoid

# --- Global variables ---
xor_net = None
autoencoder_net = None
XOR_MODEL_PATH = os.path.join(project_root, 'guii', 'models', 'xor_model.npz')
AUTOENCODER_MODEL_PATH = os.path.join(project_root, 'guii', 'models', 'autoencoder_model.npz')


# --- Flask App ---
app = Flask(__name__)

# --- Network Setup ---
def setup_xor_network():
    """Initializes and loads the XOR neural network."""
    global xor_net
    xor_net = Network()
    xor_net.add(Dense(2, 4))
    xor_net.add(Tanh())
    xor_net.add(Dense(4, 1))
    xor_net.add(Tanh())
    
    if os.path.exists(XOR_MODEL_PATH):
        xor_net.load(XOR_MODEL_PATH)
    else:
        print(f"WARNING: XOR model file not found at {XOR_MODEL_PATH}. GUI will use random weights.")

def setup_autoencoder_network():
    """Initializes and loads the Autoencoder network."""
    global autoencoder_net
    autoencoder_net = Network()
    autoencoder_net.add(Dense(784, 128))
    autoencoder_net.add(ReLU())
    autoencoder_net.add(Dense(128, 64))
    autoencoder_net.add(ReLU())
    autoencoder_net.add(Dense(64, 128))
    autoencoder_net.add(ReLU())
    autoencoder_net.add(Dense(128, 784))
    autoencoder_net.add(Sigmoid())
    
    if os.path.exists(AUTOENCODER_MODEL_PATH):
        autoencoder_net.load(AUTOENCODER_MODEL_PATH)
    else:
        print(f"WARNING: Autoencoder model file not found at {AUTOENCODER_MODEL_PATH}. GUI will use random weights.")


# --- HTML Routes ---
@app.route('/')
def index():
    """Serves the main XOR visualizer page."""
    return render_template('index.html')

@app.route('/autoencoder')
def autoencoder_page():
    """Serves the autoencoder visualizer page."""
    return render_template('autoencoder.html')


# --- API Routes for XOR Network ---
@app.route('/api/xor/network', methods=['GET'])
def get_xor_network_structure():
    """Returns the structure of the XOR neural network as JSON."""
    if not xor_net:
        return jsonify({"error": "XOR Network not initialized"}), 500

    layers_info = []
    for layer in xor_net.layers:
        info = {"type": layer.__class__.__name__}
        if isinstance(layer, Dense):
            info["input_size"] = layer.weights.shape[1]
            info["output_size"] = layer.weights.shape[0]
        layers_info.append(info)
    
    return jsonify({"layers": layers_info})

@app.route('/api/xor/predict', methods=['POST'])
def predict_xor():
    """Receives input data, performs a forward pass on XOR net, and returns the result."""
    if not xor_net:
        return jsonify({"error": "XOR Network not initialized"}), 500

    data = request.json
    if 'input' not in data:
        return jsonify({"error": "Invalid input"}), 400

    input_data = np.array(data['input']).reshape(2,)
    
    activations = []
    output = input_data
    for layer in xor_net.layers:
        output = layer.forward(output)
        activations.append(output.copy().tolist())

    prediction = activations[-1]
    
    return jsonify({
        "prediction": prediction,
        "activations": activations
    })

# --- API Routes for Autoencoder Network ---
@app.route('/api/autoencoder/structure', methods=['GET'])
def get_autoencoder_structure():
    """Returns the structure of the autoencoder network as JSON."""
    if not autoencoder_net:
        return jsonify({"error": "Autoencoder Network not initialized"}), 500

    layers_info = []
    for layer in autoencoder_net.layers:
        info = {"type": layer.__class__.__name__}
        if isinstance(layer, Dense):
            info["input_size"] = layer.weights.shape[1]
            info["output_size"] = layer.weights.shape[0]
        layers_info.append(info)
    
    return jsonify({"layers": layers_info})

@app.route('/api/autoencoder/predict', methods=['POST'])
def predict_autoencoder():
    """Receives image data and returns the reconstructed image."""
    if not autoencoder_net:
        return jsonify({"error": "Autoencoder Network not initialized"}), 500
    
    data = request.json
    if 'input' not in data or len(data['input']) != 784:
        return jsonify({"error": "Invalid input: requires a 784-element array."}), 400

    input_data = np.array(data['input'])
    
    # The predict method handles batching, so we wrap the input in a list
    reconstruction = autoencoder_net.predict([input_data])[0]
    
    return jsonify({"reconstruction": reconstruction.tolist()})


# --- Main Execution ---
if __name__ == '__main__':
    setup_xor_network()
    setup_autoencoder_network()
    # Note: use_reloader=False is important to prevent setup_network from being called twice
    app.run(debug=True, use_reloader=False)
