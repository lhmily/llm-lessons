from collections import Counter


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode_bytes(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8")


def pair_counts(tokens: list[int]):
    return Counter(zip(tokens, tokens[1:], strict=False))


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(text: str, merges: int) -> tuple[list[int], list[tuple[tuple[int, int], int]]]:
    tokens = encode_bytes(text)
    rules = []
    for new_id in range(256, 256 + merges):
        counts = pair_counts(tokens)
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        tokens = merge_pair(tokens, pair, new_id)
        rules.append((pair, new_id))
    return tokens, rules
