# -*- coding: utf-8 -*-
"""losses.py

Loss functions for neural network training.
Each loss function is implemented with both the loss calculation and its derivative.
"""

import numpy as np


def mse_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function.
    
    Formula: MSE = (1/n) * Σ(y_pred - y_true)^2
    
    Args:
        y_true: Ground truth values (target)
        y_pred: Predicted values from the network
    
    Returns:
        Scalar loss value
    """
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_prime(y_true, y_pred):
    """
    Derivative of Mean Squared Error loss with respect to predictions.
    
    Formula: dMSE/dy_pred = (2/n) * (y_pred - y_true)
    
    Args:
        y_true: Ground truth values (target)
        y_pred: Predicted values from the network
    
    Returns:
        Gradient of the loss with respect to predictions
    """
    return 2 * (y_pred - y_true) / y_true.size
