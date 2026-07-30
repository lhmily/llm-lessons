import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")


class TransformerBlock(nn.Module):
    def __init__(self, width: int, attention: nn.Module) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attention = attention
        self.norm2 = RMSNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")
