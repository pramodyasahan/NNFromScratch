from src.training.loss import CrossEntropyLoss


class Trainer:
    def __init__(self, model, dataset, optimizer, epochs):
        """Initialize the Trainer with a model, dataset, optimizer, and number of epochs."""
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.epochs = epochs
        self.loss_fn = CrossEntropyLoss()

    def train(self):
        """Train the neural network model."""
        X_train, Y_train = self.dataset.get_train_data()
        X_test, Y_test = self.dataset.get_test_data()

        for epoch in range(self.epochs):
            # Forward pass
            predictions = self.model.forward(X_train)

            # Compute loss
            loss = self.loss_fn.compute_loss(predictions, Y_train)

            # Backward pass
            self.model.backward(predictions, Y_train)

            # Update weights using optimizer
            self.model.update_weights(self.optimizer)

            # Evaluate on test set
            test_predictions = self.model.forward(X_test)
            test_accuracy = Metrics.accuracy(test_predictions, Y_test)

            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

        print("Training complete!")
