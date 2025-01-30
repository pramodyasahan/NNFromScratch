import matplotlib.pyplot as plt

# Function to plot metrics
def plot_metrics(train_losses, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # Plot Training & Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, test_losses, label="Test Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Test Loss")
    plt.legend()

    # Plot Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in test_accuracies], label="Test Accuracy", color="green", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()