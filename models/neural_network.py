import numpy as np
from utils import sigmoid, relu, softmax, cross_entropy_loss


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        self.cache = {}
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * 0.01,
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * 0.01,
            'b2': np.zeros((1, output_size))
        }

    def forward(self, X):
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = sigmoid(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = softmax(self.cache['Z2'])
        return self.cache['A2']

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.cache['A2'] - y
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.params['W2'].T) * (self.cache['A1'] * (1 - self.cache['A1']))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return gradients

    def update_parameters(self, grads, optimizer):
        self.params = optimizer.update(self.params, grads)
