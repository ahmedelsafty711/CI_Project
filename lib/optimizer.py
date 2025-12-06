# lib/optimizer.py
import numpy as np

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Updates weights and biases using:
        W = W - learning_rate * dW
        b = b - learning_rate * db
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """
        Perform one gradient descent step on a single layer.
        Assumes layer has:
            layer.weights
            layer.biases
            layer.grad_weights
            layer.grad_biases
        """
        if hasattr(layer, 'weights') and layer.grad_weights is not None:
            layer.weights -= self.learning_rate * layer.grad_weights
        if hasattr(layer, 'biases') and layer.grad_biases is not None:
            layer.biases -= self.learning_rate * layer.grad_biases