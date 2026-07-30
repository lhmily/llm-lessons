import torch
from torch import nn
from torch.nn import functional as F

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_lora():
    base = nn.Linear(5, 3)
    layer = m.LoRALinear(base, 2, 4)
    layer.b.data.normal_()
    x = torch.randn(4, 5)
    assert torch.allclose(layer(x), F.linear(x, layer.merged_weight(), base.bias), atol=1e-6)
    assert not base.weight.requires_grad
