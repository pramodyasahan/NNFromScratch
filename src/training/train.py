import torch
import torch.nn.functional as F
from training.loss import CrossEntropyLoss


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
        train_loader = self.dataset.get_train_data()
        test_loader = self.dataset.get_test_data()

        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total_samples = 0

            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.view(X_batch.shape[0], -1), F.one_hot(Y_batch, num_classes=10).float()

                # Forward pass
                predictions = self.model.forward(X_batch)

                # Compute loss
                loss = self.loss_fn.compute_loss(predictions, Y_batch)
                total_loss += loss

                # Compute accuracy
                correct += (torch.argmax(predictions, dim=1) == torch.argmax(Y_batch, dim=1)).sum().item()
                total_samples += Y_batch.shape[0]

                # Backward pass
                self.model.backward(predictions, Y_batch)

                # Update weights using optimizer
                self.model.update_weights(self.optimizer)

            # Compute epoch loss and accuracy
            avg_loss = total_loss / len(train_loader)
            train_accuracy = correct / total_samples

            # Print progress
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

        print("Training complete!")
