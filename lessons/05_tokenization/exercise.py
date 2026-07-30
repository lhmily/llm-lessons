def encode_bytes(text: str) -> list[int]:
    raise NotImplementedError("TODO")


def decode_bytes(tokens: list[int]) -> str:
    raise NotImplementedError("TODO")


def pair_counts(tokens: list[int]):
    raise NotImplementedError("TODO")


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    raise NotImplementedError("TODO")


def train_bpe(text: str, merges: int) -> tuple[list[int], list[tuple[tuple[int, int], int]]]:
    raise NotImplementedError("TODO")
