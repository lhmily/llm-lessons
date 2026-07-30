import torch
from torch import nn


def one_hot_lookup(tokens: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO")


def sinusoidal_positions(length: int, width: int, device=None) -> torch.Tensor:
    raise NotImplementedError("TODO")


def embed_sequence(
    tokens: torch.Tensor, token_embedding: nn.Embedding, position_embedding: nn.Embedding
) -> torch.Tensor:
    raise NotImplementedError("TODO")
