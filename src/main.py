import torch
from data.data_loader import DatasetLoader
from model.layers import DenseLayer, ActivationLayer
from model.model import NeuralNetwork
from training.loss import CrossEntropyLoss
from training.optimizers import SGD
from training.train import Trainer

# Load dataset
batch_size = 32
dataset = DatasetLoader(dataset_name="mnist", batch_size=batch_size)

train_loader = dataset.get_train_data()
test_loader = dataset.get_test_data()

# Define Neural Network architecture
model = NeuralNetwork([
    DenseLayer(input_size=28 * 28, output_size=128),  # Flatten MNIST images (28x28 -> 784)
    ActivationLayer("relu"),
    DenseLayer(input_size=128, output_size=10),  # 10 output classes (digits 0-9)
    ActivationLayer("softmax")
])

# Define optimizer and loss function
optimizer = SGD(learning_rate=0.01)
loss_fn = CrossEntropyLoss()

# Initialize Trainer
trainer = Trainer(model, dataset, optimizer, epochs=10)

# Train the model
trainer.train()

# Evaluate the trained model
X_test, Y_test = dataset.get_test_data()
X_test, Y_test = X_test, Y_test  # Move to GPU if available

with torch.no_grad():  # Disable gradient calculation for inference
    predictions = model.forward(X_test)
    test_accuracy = (torch.argmax(predictions, dim=1) == torch.argmax(Y_test, dim=1)).float().mean().item()

print(f"Final Test Accuracy: {test_accuracy:.4f}")
