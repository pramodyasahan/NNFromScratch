import numpy as np
from keras.datasets import mnist


def load_mnist():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize inputs
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Flatten images (28x28 -> 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Convert labels to one-hot encoding
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test
