import math

import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_metrics():
    assert (
        m.normalize_answer(" Hello, WORLD! ") == "hello world" and m.token_f1("a b", "a c") == 0.5
    )
    assert (
        abs(m.perplexity(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])) - math.exp(1.75))
        < 1e-6
    )
    low, high = m.bootstrap_interval(torch.tensor([0.0, 1.0, 1.0, 1.0]), 200, 2)
    assert 0 <= low <= high <= 1
