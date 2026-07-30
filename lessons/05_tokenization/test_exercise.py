from llm_lessons.testing import load_lesson

m = load_lesson(__file__)


def test_tokenizer():
    text = "LLMs 你好 🤖"
    assert m.decode_bytes(m.encode_bytes(text)) == text
    assert m.pair_counts([1, 2, 1, 2])[(1, 2)] == 2
    assert m.merge_pair([1, 2, 1, 2, 1], (1, 2), 9) == [9, 9, 1]
    a = m.train_bpe("banana banana", 4)
    b = m.train_bpe("banana banana", 4)
    assert a == b and len(a[0]) < len(b"banana banana")
