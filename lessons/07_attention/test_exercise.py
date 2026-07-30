import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_causal_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 6)
    out, w = m.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 6)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4))
    assert torch.equal(w.triu(1), torch.zeros_like(w).triu(1))
    changed = v.clone()
    changed[:, 3] += 1000
    out2, _ = m.scaled_dot_product_attention(q, k, changed)
    assert torch.allclose(out[:, :3], out2[:, :3])
