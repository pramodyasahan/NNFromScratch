import numpy as np
import time
from tqdm import tqdm  # For progress bar
from datasets.mnist import load_mnist
from metrics.plot import plot_metrics
from models.neural_network import NeuralNetwork
from optimizers.sgd import SGD
from utils import cross_entropy_loss

# Load data
x_train, y_train, x_test, y_test = load_mnist()

# Initialize network and optimizer
nn = NeuralNetwork()
optimizer = SGD(learning_rate=0.01)

# Training settings
epochs = 10
batch_size = 64
num_batches = x_train.shape[0] // batch_size

# Store losses and accuracy for plotting
train_losses = []
test_losses = []
test_accuracies = []

start_time = time.time()
for epoch in range(epochs):
    epoch_loss = 0
    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

    for i in progress_bar:
        batch_X = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]

        # Forward pass
        predictions = nn.forward(batch_X)
        loss = cross_entropy_loss(predictions, batch_y)
        epoch_loss += loss

        # Backward pass
        gradients = nn.backward(batch_X, batch_y)

        # Update parameters
        nn.update_parameters(gradients, optimizer)

        # Update tqdm progress bar description
        progress_bar.set_postfix(loss=f"{loss:.4f}")

    # Compute test loss and accuracy after each epoch
    test_predictions = nn.forward(x_test)
    test_loss = cross_entropy_loss(test_predictions, y_test)
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))

    # Store metrics for plotting
    train_losses.append(epoch_loss / num_batches)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | Training Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Accuracy: {test_accuracies[-1] * 100:.2f}%")
    end_time = time.time() - start_time

end_time = time.time() - start_time
print(f"Time taken: {end_time:.2f}s")

# Call the function to plot results
plot_metrics(train_losses, test_losses, test_accuracies)
