import math
import re
import string
from collections import Counter

import torch


def normalize_answer(text):
    return " ".join(re.sub(f"[{re.escape(string.punctuation)}]", " ", text.lower()).split())


def token_f1(prediction, reference):
    p = normalize_answer(prediction).split()
    r = normalize_answer(reference).split()
    overlap = sum((Counter(p) & Counter(r)).values())
    if not p or not r:
        return float(p == r)
    return 2 * overlap / (len(p) + len(r))


def perplexity(losses, token_counts):
    return math.exp((losses * token_counts).sum().item() / token_counts.sum().item())


def bootstrap_interval(scores, samples=1000, seed=0):
    means = scores[
        torch.randint(
            0, len(scores), (samples, len(scores)), generator=torch.Generator().manual_seed(seed)
        )
    ].mean(1)
    return tuple(torch.quantile(means, torch.tensor([0.025, 0.975])).tolist())
