import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_cache_equivalence():
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 2, 5, 4) for _ in range(3)]
    full, _ = m.cached_attention(q, k, v)
    cache = None
    parts = []
    for i in range(5):
        out, cache = m.cached_attention(
            q[:, :, i : i + 1], k[:, :, i : i + 1], v[:, :, i : i + 1], cache
        )
        parts.append(out)
    assert torch.allclose(full, torch.cat(parts, -2), atol=1e-6)
