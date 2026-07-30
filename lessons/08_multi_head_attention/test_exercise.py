import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_heads_and_causality():
    x = torch.randn(2, 5, 12)
    heads = m.split_heads(x, 3)
    assert heads.shape == (2, 3, 5, 4)
    assert torch.equal(m.merge_heads(heads), x)
    torch.manual_seed(0)
    layer = m.MultiHeadAttention(12, 3)
    original = layer(x)
    changed = x.clone()
    changed[:, 4] += 100
    altered = layer(changed)
    assert original.shape == x.shape
    assert torch.allclose(original[:, :4], altered[:, :4], atol=1e-5)
