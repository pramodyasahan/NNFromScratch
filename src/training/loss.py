import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        """Initialize loss function"""
        self.loss = None

    def compute_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        cross_entropy = -np.sum(labels * np.log(predictions + 1e-8)) / predictions.shape[0]  # Averaged over batch
        return cross_entropy

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute derivative of cross-entropy loss"""
        return predictions - labels  # dZ = A2 - Y
