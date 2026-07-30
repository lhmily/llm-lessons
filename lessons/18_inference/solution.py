import torch


def quantize_int8(weight):
    scale = weight.abs().max().clamp_min(1e-8) / 127
    return (weight / scale).round().clamp(-127, 127).to(torch.int8), scale


def dequantize_int8(values, scale):
    return values.float() * scale


def storage_bytes(values, scale):
    return values.numel() * values.element_size() + scale.numel() * scale.element_size()


def dynamic_batches(lengths, max_requests, max_tokens):
    batches = []
    current = []
    tokens = 0
    for i, length in enumerate(lengths):
        if length > max_tokens:
            raise ValueError("request exceeds token budget")
        if current and (len(current) == max_requests or tokens + length > max_tokens):
            batches.append(current)
            current = []
            tokens = 0
        current.append(i)
        tokens += length
    if current:
        batches.append(current)
    return batches
