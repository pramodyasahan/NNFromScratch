import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DatasetLoader:
    def __init__(self, dataset_name="mnist", batch_size=32):
        """Load and preprocess dataset for PyTorch."""
        if dataset_name.lower() != "mnist":
            raise ValueError("Only MNIST is supported for now.")

        # Define data transformations (Convert to tensor & Normalize)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean & std
        ])

        # Load MNIST dataset
        self.train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        self.test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

        self.batch_size = batch_size

        # Create DataLoader for batching
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_train_data(self):
        """Return training data loader (mini-batches)."""
        return self.train_loader

    def get_test_data(self):
        """Return test data loader (full dataset for evaluation)."""
        return self.test_loader
