# lib/network.py
import numpy as np
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
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
            
        return result

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