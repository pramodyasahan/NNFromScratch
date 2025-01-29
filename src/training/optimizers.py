from src.model.layers import DenseLayer


class SGD:
    def __init__(self, learning_rate: float):
        """Initialize learning rate for SGD."""
        self.learning_rate = learning_rate

    def update(self, layer):
        """Update weights and biases of a DenseLayer using SGD."""
        if isinstance(layer, DenseLayer):  # Only update trainable layers
            layer.weights -= self.learning_rate * layer.dW  # Update weights
            layer.biases -= self.learning_rate * layer.db  # Update biases
