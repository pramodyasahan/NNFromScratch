import torch


class CrossEntropyLoss:
    def __init__(self):
        """Initialize loss function"""
        self.loss = None

    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss using PyTorch operations"""
        batch_size = predictions.shape[0]
        cross_entropy = -torch.sum(labels * torch.log(predictions + 1e-8)) / batch_size  # Use `torch.sum()`
        return cross_entropy

    def backward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute derivative of cross-entropy loss"""
        return predictions - labels  # dZ = A2 - Y
