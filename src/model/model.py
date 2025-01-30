from model.layers import DenseLayer
from training.loss import CrossEntropyLoss


class NeuralNetwork:
    def __init__(self, layer_structure):
        """Initialize list of layers"""
        self.layers = layer_structure  # List of DenseLayer and ActivationLayer instances
        self.loss_fn = CrossEntropyLoss()

    def forward(self, X):
        """Perform forward propagation through all layers"""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, predictions, labels):
        """Perform backpropagation through all layers"""
        dZ = self.loss_fn.backward(predictions, labels)

        for layer in reversed(self.layers):  # Loop backward through layers
            if isinstance(layer, DenseLayer):  # Only DenseLayer has weights
                dW, db, dZ = layer.backward(dZ)  # Get weight, bias, and activation gradients
                layer.dW = dW  # Store computed gradients
                layer.db = db
            else:  # If it's an ActivationLayer, just pass gradients through
                dZ = layer.backward(dZ)

    def update_weights(self, optimizer):
        """Use optimizer to update weights"""
        for layer in self.layers:
            if isinstance(layer, DenseLayer):  # Only DenseLayer has weights to update
                optimizer.update(layer)
