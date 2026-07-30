import torch
from torch.nn import functional as F


def make_examples(tokens: torch.Tensor, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    windows = tokens.unfold(0, context_length + 1, 1)
    return windows[:, :-1].clone(), windows[:, 1:].clone()


def split_tokens(tokens: torch.Tensor, fraction: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    cut = int(tokens.numel() * fraction)
    return tokens[:cut], tokens[cut:]


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
