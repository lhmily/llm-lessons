import torch


def filter_logits(
    logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    raise NotImplementedError("TODO")


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator=None,
) -> torch.Tensor:
    raise NotImplementedError("TODO")


def generate(model, tokens: torch.Tensor, max_new_tokens: int, **sampling) -> torch.Tensor:
    raise NotImplementedError("TODO")
