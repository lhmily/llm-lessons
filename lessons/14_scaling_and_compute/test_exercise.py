import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_scaling():
    assert m.parameter_memory(1000) == 16000 and m.training_flops(100, 50) == 30000
    x = torch.tensor([1.0, 4.0, 16.0, 64.0])
    c, e = m.fit_power_law(x, 1 + 2 * x.pow(-0.5), 1.0)
    assert abs(c - 2) < 1e-5 and abs(e + 0.5) < 1e-5
