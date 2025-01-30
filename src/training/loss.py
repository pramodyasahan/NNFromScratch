import torch
import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        """Initialize loss function"""
        self.loss = None

    def compute_loss(self, predictions, labels):
        """Compute cross-entropy loss using PyTorch operations"""
        # Ensure inputs are PyTorch tensors
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, dtype=torch.float32)

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float32)

        batch_size = predictions.shape[0]
        cross_entropy = -torch.sum(labels * torch.log(predictions + 1e-8)) / batch_size  # Now all are tensors
        return cross_entropy

    def backward(self, predictions, labels):
        """Compute derivative of cross-entropy loss"""
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions, dtype=torch.float32)

        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float32)

        return predictions - labels  # dZ = A2 - Y
