import torch

from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_inference():
    weight = torch.randn(32, 16)
    q, s = m.quantize_int8(weight)
    restored = m.dequantize_int8(q, s)
    assert (restored - weight).abs().max() <= s / 2 + 1e-6 and m.storage_bytes(
        q, s
    ) < weight.numel() * weight.element_size()
    assert m.dynamic_batches([3, 5, 4, 8], 2, 9) == [[0, 1], [2], [3]]
