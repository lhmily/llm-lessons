import torch
from torch import nn
from torch.nn import functional as F


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, t, d = x.shape
    return x.view(b, t, heads, d // heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, t, d = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, h * d)


class MultiHeadAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (split_heads(t, self.heads) for t in self.qkv(x).chunk(3, -1))
        scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
        mask = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).tril()
        weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        return self.output(merge_heads(weights @ v))
