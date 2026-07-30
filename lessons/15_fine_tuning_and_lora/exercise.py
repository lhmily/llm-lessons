import torch
from torch import nn


def response_loss(
    logits: torch.Tensor, targets: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    raise NotImplementedError("TODO")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x):
        raise NotImplementedError("TODO")

    def merged_weight(self):
        raise NotImplementedError("TODO")
