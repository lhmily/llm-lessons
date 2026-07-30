import torch


def normalize_answer(text: str) -> str:
    raise NotImplementedError("TODO")


def token_f1(prediction: str, reference: str) -> float:
    raise NotImplementedError("TODO")


def perplexity(losses: torch.Tensor, token_counts: torch.Tensor) -> float:
    raise NotImplementedError("TODO")


def bootstrap_interval(
    scores: torch.Tensor, samples: int = 1000, seed: int = 0
) -> tuple[float, float]:
    raise NotImplementedError("TODO")
