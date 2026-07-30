import torch
from torch import nn
from torch.nn import functional as F


def response_loss(logits, targets, response_mask):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.masked_fill(~response_mask, -100).reshape(-1),
        ignore_index=-100,
    )


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=1.0):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.a.T) @ self.b.T

    def merged_weight(self):
        return self.base.weight + self.scale * self.b @ self.a
