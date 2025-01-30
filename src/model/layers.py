import numpy as np
import torch
from utils.activations import ActivationFunctions


class DenseLayer:
    def __init__(self, input_size: int, output_size: int):
        """Initialize weights and biases on CPU"""
        self.input_size = input_size
        self.output_size = output_size

        self.weights = torch.randn(input_size, output_size, dtype=torch.float32) * 0.01
        self.biases = torch.zeros((1, output_size), dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Z = WX + b"""
        self.input = X  # Store input for backpropagation
        self.Z = torch.matmul(X, self.weights) + self.biases
        return self.Z

    def backward(self, dZ: torch.Tensor) -> tuple:
        """Compute gradients for weight and bias updates"""
        batch_size = self.input.shape[0]

        dW = torch.matmul(self.input.T, dZ) / batch_size
        db = torch.sum(dZ, dim=0, keepdim=True) / batch_size

        dX = torch.matmul(dZ, self.weights.T)  # Gradient w.r.t. input for previous layer
        return dW, db, dX  # Return computed gradients


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
