import numpy as np

from src.utils.activations import ActivationFunctions


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


class ActivationLayer:
    def __init__(self, activation_type: str) -> None:
        """Initialize activation function based on type."""
        if activation_type == "relu":
            self.activation = ActivationFunctions.relu
            self.derivative = ActivationFunctions.relu_derivative
        elif activation_type == "softmax":
            self.activation = ActivationFunctions.softmax
            self.derivative = None  # Softmax derivative is handled in loss function
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        return self.activation(Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.derivative:
            return self.derivative(dA)
        else:
            raise ValueError("Softmax derivative is handled in the loss function.")
