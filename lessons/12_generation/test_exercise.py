import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_sampling():
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    assert (
        m.sample_token(logits, temperature=0).item() == 1
        and torch.isfinite(m.filter_logits(logits, top_k=2)).sum() == 2
    )
    g1 = torch.Generator().manual_seed(4)
    g2 = torch.Generator().manual_seed(4)
    assert torch.equal(m.sample_token(logits, generator=g1), m.sample_token(logits, generator=g2))
