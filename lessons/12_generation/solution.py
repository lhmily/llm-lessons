import torch
from torch.nn import functional as F


def filter_logits(logits, top_k=None, top_p=None):
    result = logits.clone()
    if top_k is not None:
        result = result.masked_fill(
            result < result.topk(min(top_k, result.size(-1))).values[..., -1, None], float("-inf")
        )
    if top_p is not None:
        ordered, indices = result.sort(descending=True)
        remove = F.softmax(ordered, -1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        result = result.masked_fill(
            torch.zeros_like(remove).scatter(-1, indices, remove), float("-inf")
        )
    return result


def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    return torch.multinomial(
        F.softmax(filter_logits(logits / temperature, top_k, top_p), -1), 1, generator=generator
    )


def generate(model, tokens, max_new_tokens, **sampling):
    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -model.config.context_length :])
        tokens = torch.cat((tokens, sample_token(logits[:, -1], **sampling)), 1)
    return tokens
