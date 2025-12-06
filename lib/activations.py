# -*- coding: utf-8 -*-
"""activations.py

Activation layers for neural networks.
Each activation function is implemented as a Layer with forward and backward methods.
"""

import numpy as np
from .layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    f'(x) = 1 if x > 0, else 0
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        """
        Apply ReLU activation: output = max(0, input)
        """
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backpropagate through ReLU.
        Gradient is 1 where input > 0, otherwise 0.
        """
        # Element-wise multiplication of output_error with derivative
        input_error = output_error * (self.input > 0)
        return input_error


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        """
        Apply sigmoid activation: output = 1 / (1 + exp(-input))
        """
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backpropagate through Sigmoid.
        Gradient is sigmoid(x) * (1 - sigmoid(x))
        """
        sigmoid_derivative = self.output * (1 - self.output)
        input_error = output_error * sigmoid_derivative
        return input_error


class Tanh(Layer):
    """
    Hyperbolic tangent activation function.
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    f'(x) = 1 - tanh^2(x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        """
        Apply tanh activation: output = tanh(input)
        """
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backpropagate through Tanh.
        Gradient is 1 - tanh^2(x)
        """
        tanh_derivative = 1 - self.output ** 2
        input_error = output_error * tanh_derivative
        return input_error


class Softmax(Layer):
    """
    Softmax activation function (typically used for multi-class classification).
    f(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    
    Note: Softmax backward pass is complex when used with cross-entropy loss.
    This implementation handles the general case.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        """
        Apply softmax activation: output_i = exp(x_i) / sum(exp(x_j))
        Uses numerical stability trick: subtract max before exp
        """
        self.input = input_data
        # Numerical stability: subtract max value
        exp_values = np.exp(input_data - np.max(input_data))
        self.output = exp_values / np.sum(exp_values)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backpropagate through Softmax.
        For a single sample, the Jacobian is:
        J_ij = s_i * (δ_ij - s_j) where s is softmax output, δ is Kronecker delta
        """
        n = self.output.shape[0]
        
        # Compute Jacobian matrix of softmax
        # J[i,j] = output[i] * (1 if i==j else 0) - output[i] * output[j]
        jacobian = np.diagflat(self.output) - np.outer(self.output, self.output)
        
        # Multiply Jacobian with output_error
        input_error = np.dot(jacobian, output_error)
        return input_error
