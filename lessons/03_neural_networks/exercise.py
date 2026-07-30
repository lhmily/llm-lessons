import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, classes: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO: use log_softmax")


def optimization_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer) -> float:
    raise NotImplementedError("TODO")
