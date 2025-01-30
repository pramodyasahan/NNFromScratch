import torch


class ActivationFunctions:
    @staticmethod
    def relu(Z: torch.Tensor) -> torch.Tensor:
        """Applies the ReLU function element-wise."""
        return torch.maximum(torch.tensor(0.0, dtype=Z.dtype), Z)

    @staticmethod
    def softmax(Z: torch.Tensor) -> torch.Tensor:
        """Applies the Softmax function with numerical stability."""
        Z_max = torch.max(Z, dim=1, keepdim=True)[0]
        Z_exp = torch.exp(Z - Z_max)
        return Z_exp / torch.sum(Z_exp, dim=1, keepdim=True)

    @staticmethod
    def relu_derivative(Z: torch.Tensor) -> torch.Tensor:
        """Computes the derivative of the ReLU function."""
        return (Z > 0).float()
