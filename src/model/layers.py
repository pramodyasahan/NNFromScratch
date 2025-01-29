import numpy as np


class DenseLayer:
    def __init__(self, input_size: int, output_size: int) -> None:
        """Initialize weights and biases"""
        self.Z = None
        self.input = None
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute Z = WX + b"""
        self.input = X  # Store input for backpropagation
        self.Z = np.dot(X, self.weights) + self.biases
        return self.Z

    def backward(self, dZ: np.ndarray) -> tuple:
        """Compute weight and bias gradients"""
        dW = np.dot(self.input.T, dZ) / self.input.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / self.input.shape[0]

        dX = np.dot(dZ, self.weights.T)
        return dW, db, dX
