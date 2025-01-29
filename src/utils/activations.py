import numpy as np


class ActivationFunctions:
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        """Applies the ReLU function element-wise."""
        return np.maximum(0, Z)

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        """Applies the Softmax function along the last axis, ensuring numerical stability."""
        Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Prevents overflow
        return Z_exp / np.sum(Z_exp, axis=1, keepdims=True)

    @staticmethod
    def relu_derivative(Z: np.ndarray) -> np.ndarray:
        """Computes the derivative of the ReLU function."""
        return (Z > 0).astype(float)
