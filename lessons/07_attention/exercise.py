import torch


def causal_mask(length: int, device=None) -> torch.Tensor:
    raise NotImplementedError("TODO")


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("TODO")
