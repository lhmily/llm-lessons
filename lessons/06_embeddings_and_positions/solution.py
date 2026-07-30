import math

import torch
from torch import nn
from torch.nn import functional as F


def one_hot_lookup(tokens: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return F.one_hot(tokens, num_classes=table.size(0)).to(table.dtype) @ table


def sinusoidal_positions(length: int, width: int, device=None) -> torch.Tensor:
    positions = torch.arange(length, device=device).float().unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width)
    )
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        result[:, 1::2] = torch.cos(positions * frequencies[: result[:, 1::2].shape[1]])
    return result


def embed_sequence(
    tokens: torch.Tensor, token_embedding: nn.Embedding, position_embedding: nn.Embedding
) -> torch.Tensor:
    positions = torch.arange(tokens.size(1), device=tokens.device)
    return token_embedding(tokens) + position_embedding(positions)
