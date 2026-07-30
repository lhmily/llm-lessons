import torch


def make_examples(tokens: torch.Tensor, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("TODO")


def split_tokens(tokens: torch.Tensor, fraction: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("TODO")


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("TODO")
