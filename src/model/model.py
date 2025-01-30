from model.layers import ActivationLayer, DenseLayer
from training.loss import CrossEntropyLoss
from utils.activations import ActivationFunctions


class NeuralNetwork:
    def __init__(self, layer_structure):
        """Initialize list of layers"""
        self.layers = layer_structure
        self.loss_fn = CrossEntropyLoss()

    def forward(self, X):
        """Perform forward propagation through all layers"""
        for layer in self.layers:
            X = layer.forward(X)
        return X  # Final softmax output

    def backward(self, predictions, labels):
        """Perform backpropagation through all layers"""
        dZ = self.loss_fn.backward(predictions, labels)  # dZ = A2 - Y

        for layer in reversed(self.layers):
            # ✅ Skip Softmax Layer during backpropagation
            if isinstance(layer, ActivationLayer) and layer.activation == ActivationFunctions.softmax:
                continue  # Softmax derivative is already handled in loss function

            if isinstance(layer, DenseLayer):  # Backprop through DenseLayer
                dW, db, dZ = layer.backward(dZ)
                layer.dW = dW
                layer.db = db
            else:  # Activation layers (ReLU) pass gradients through
                dZ = layer.backward(dZ)
