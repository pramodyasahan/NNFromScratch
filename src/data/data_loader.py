import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size):
        """TODO: Initialize weights and biases"""
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, X):
        """TODO: Compute Z = WX + b"""
        pass

    def backward(self, dZ):
        """TODO: Compute weight and bias gradients"""
        pass


class ActivationLayer:
    def __init__(self, activation_type):
        """TODO: Store activation function type (ReLU, Softmax)"""
        pass

    def forward(self, Z):
        """TODO: Apply activation function"""
        pass

    def backward(self, dA):
        """TODO: Compute derivative of activation function"""
        pass
