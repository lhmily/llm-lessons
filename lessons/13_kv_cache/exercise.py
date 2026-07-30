import torch


def append_cache(cache: tuple[torch.Tensor, torch.Tensor] | None, k: torch.Tensor, v: torch.Tensor):
    raise NotImplementedError("TODO")


def offset_causal_mask(
    query_length: int, key_length: int, past_length: int, device=None
) -> torch.Tensor:
    raise NotImplementedError("TODO")


def cached_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache=None):
    raise NotImplementedError("TODO")
