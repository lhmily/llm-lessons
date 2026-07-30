from torch import nn

from llm_lessons.tiny_gpt import TinyGPT, TinyGPTConfig


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["TinyGPT", "TinyGPTConfig", "parameter_count"]
