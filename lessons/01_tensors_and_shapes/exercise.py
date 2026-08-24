import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity along the last dimension."""

    raise NotImplementedError("TODO: normalize and take a dot product")


def batched_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Map (..., in_features) to (..., out_features)."""
    raise NotImplementedError("TODO: use matmul and broadcasting")


def standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standardize each row over its final dimension."""
    raise NotImplementedError("TODO: subtract mean and divide by standard deviation")
