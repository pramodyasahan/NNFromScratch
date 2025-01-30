import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist  # Can be replaced with another dataset


class DatasetLoader:
    def __init__(self, dataset_name: str, batch_size: int = 32):
        """Load dataset, preprocess it, and prepare for training."""
        self.batch_size = batch_size

        if dataset_name == "mnist":
            (X, y), (X_test, y_test) = mnist.load_data()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Normalize images (MNIST pixel values range from 0 to 255)
        X = X / 255.0
        X_test = X_test / 255.0

        # Flatten images (MNIST 28x28 -> 784)
        X = X.reshape(X.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Convert labels to one-hot encoding
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

        # Split training data into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        self.X_test, self.y_test = X_test, y_test

        # Shuffle training data for the first time
        self.shuffle_data()

    def shuffle_data(self):
        """Shuffle training data to avoid overfitting to data order."""
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)
        self.X_train = self.X_train[indices]
        self.y_train = self.y_train[indices]

    def get_train_data(self):
        """Yield training data in mini-batches."""
        for i in range(0, self.X_train.shape[0], self.batch_size):
            yield self.X_train[i:i + self.batch_size], self.y_train[i:i + self.batch_size]

    def get_test_data(self):
        """Return test data (full dataset, no batching needed)."""
        return self.X_test, self.y_test
