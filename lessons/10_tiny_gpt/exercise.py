import torch
from torch import nn

from llm_lessons.tiny_gpt import TinyGPTConfig


class TinyGPT(nn.Module):
    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        raise NotImplementedError("TODO: assemble the components from lessons 7-9")

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        raise NotImplementedError("TODO")


def parameter_count(model: nn.Module) -> int:
    raise NotImplementedError("TODO")
