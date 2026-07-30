"""Small deterministic datasets; nothing in this module accesses the network."""

from __future__ import annotations

import torch

TINY_CORPUS = (
    "language models learn to predict the next token. "
    "attention lets every token gather useful context. "
) * 12


def byte_tokens(text: str = TINY_CORPUS) -> torch.Tensor:
    """Encode UTF-8 text as integer byte tokens."""
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def make_batches(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-token batches with shapes (batch, time)."""
    if tokens.numel() <= context_length:
        raise ValueError("tokens must be longer than context_length")
    starts = torch.randint(0, tokens.numel() - context_length, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + context_length] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1 : i + context_length + 1] for i in starts.tolist()])
    return x, y


def classification_data(n: int = 128, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a linearly separable two-dimensional dataset."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=generator)
    y = (x[:, 0] + 0.7 * x[:, 1] > 0).long()
    return x, y
