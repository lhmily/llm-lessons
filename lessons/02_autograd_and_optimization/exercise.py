import torch


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO")


def train_step(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, lr: float
) -> torch.Tensor:
    """Update leaf tensors in place and return the pre-update loss."""
    raise NotImplementedError("TODO")


def numerical_gradient(function, value: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    raise NotImplementedError("TODO")
