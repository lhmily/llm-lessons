import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_tensor_operations():
    a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    assert torch.allclose(m.cosine_similarity(a, b), torch.tensor([1.0, 0.0]), atol=1e-6)
    x = torch.randn(2, 3, 4)
    w = torch.randn(4, 5)
    bias = torch.randn(5)
    assert torch.allclose(
        m.batched_linear(x, w, bias),
        torch.stack([row @ w + bias for batch in x for row in batch]).view(2, 3, 5),
    )
    z = m.standardize(torch.randn(4, 8))
    assert torch.allclose(z.mean(-1), torch.zeros(4), atol=1e-6)
    assert torch.allclose(z.var(-1, unbiased=False), torch.ones(4), atol=2e-4)
