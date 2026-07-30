import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, classes: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, dim=-1)[torch.arange(targets.numel()), targets].mean()


def optimization_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer) -> float:
    optimizer.zero_grad()
    loss = cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()
