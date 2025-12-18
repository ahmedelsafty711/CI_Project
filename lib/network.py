# lib/network.py
import numpy as np

from .layers import Dense
from .optimizer import SGD

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            # Ensure input_data[i] is a numpy array
            output = np.array(input_data[i], ndmin=1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def save(self, file_path):
        """Saves the network's weights and biases to a file."""
        params = {}
        dense_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                params[f'w{dense_layer_idx}'] = layer.weights
                params[f'b{dense_layer_idx}'] = layer.biases
                dense_layer_idx += 1
        np.savez(file_path, **params)
        print(f"Network saved to {file_path}")

    def load(self, file_path):
        """Loads the network's weights and biases from a file."""
        params = np.load(file_path)
        dense_layer_idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                if f'w{dense_layer_idx}' in params:
                    layer.weights = params[f'w{dense_layer_idx}']
                    layer.biases = params[f'b{dense_layer_idx}']
                    dense_layer_idx += 1
                else:
                    print(f"Warning: No weights found for layer {dense_layer_idx} in {file_path}")
        print(f"Network loaded from {file_path}")


    def train(self, x_train, y_train, epochs, learning_rate):
        # Initialize optimizer
        optimizer = SGD(learning_rate=learning_rate)
        samples = len(x_train)

        for i in range(epochs):
            total_error = 0
            
            for j in range(samples):
                # Forward pass
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                # Track loss
                total_error += self.loss(y_train[j], output)

                # Backward pass
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    # Compute gradients without updating weights immediately
                    error = layer.backward(error, learning_rate=None)
                
                # Optimizer step: Update weights and biases
                for layer in self.layers:
                    optimizer.update(layer)

            # Log average error every 1000 epochs
            total_error /= samples
            if (i + 1) % 1000 == 0:
                print(f"Epoch {i + 1}/{epochs}   error={total_error:.6f}")