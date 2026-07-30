import torch
from torch.nn import functional as F


def causal_mask(length: int, device=None) -> torch.Tensor:
    return torch.ones(length, length, dtype=torch.bool, device=device).tril()


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if causal:
        scores = scores.masked_fill(~causal_mask(q.size(-2), q.device), float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights
