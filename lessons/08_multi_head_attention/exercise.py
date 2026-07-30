import torch
from torch import nn


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    raise NotImplementedError("TODO")


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO")


class MultiHeadAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")
