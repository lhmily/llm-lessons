import torch
from torch import nn

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_norm_and_block():
    x = torch.randn(2, 4, 8, requires_grad=True)
    norm = m.RMSNorm(8)
    y = norm(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 4), atol=1e-4)
    block = m.TransformerBlock(8, nn.Linear(8, 8, bias=False))
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
