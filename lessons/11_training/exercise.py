import torch
from torch import nn


def train_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer, max_norm: float = 1.0
) -> float:
    raise NotImplementedError("TODO")


def evaluate(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    raise NotImplementedError("TODO")
