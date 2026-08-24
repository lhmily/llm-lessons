from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
LESSONS = sorted(
    path for path in (ROOT / "lessons").iterdir() if path.is_dir() and path.name[:2].isdigit()
)
DOCUMENTS = [ROOT / "README.md", *(path / "README.md" for path in LESSONS)]
REQUIRED_HEADINGS = {
    "## Learning objectives",
    "## Prerequisites",
    "## Mental model",
    "## Derivation and algorithm",
    "## Worked PyTorch example",
    "## Exercise",
    "## Expected shapes and invariants",
    "## Common mistakes",
    "## Further experiments",
    "## Summary",
}
KEYWORDS = {
    "01": ("broadcast", "cosine", "standard"),
    "02": ("autograd", "gradient", "finite difference"),
    "03": ("logits", "relu", "cross-entropy"),
    "04": ("next token", "target", "context"),
    "05": ("utf-8", "byte", "bpe"),
    "06": ("embedding", "one-hot", "position"),
    "07": ("query", "causal", "softmax"),
    "08": ("head", "split", "merge"),
    "09": ("residual", "rmsnorm", "pre-norm"),
    "10": ("tinygpt", "weight tying", "logits"),
    "11": ("adamw", "gradient clipping", "checkpoint"),
    "12": ("temperature", "top-k", "top-p"),
    "13": ("cache", "key", "value"),
    "14": ("flops", "memory", "power-law"),
    "15": ("lora", "frozen", "response"),
    "16": ("perplexity", "bootstrap", "f1"),
    "17": ("chosen", "rejected", "dpo"),
    "18": ("int8", "batch", "latency"),
}


def mermaid_blocks(text: str) -> list[str]:
    return re.findall(r"```mermaid\s*\n(.*?)```", text, flags=re.DOTALL)


def markdown_targets(text: str) -> list[str]:
    return re.findall(r"!?\[[^]]*]\(([^)]+)\)", text)


def test_visual_document_set_is_complete():
    assert len(LESSONS) == 18
    assert len(DOCUMENTS) == 19
    assert all(path.exists() for path in DOCUMENTS)


def test_every_document_has_valid_unique_mermaid():
    seen: set[str] = set()
    for path in DOCUMENTS:
        blocks = mermaid_blocks(path.read_text())
        assert blocks, f"missing Mermaid diagram: {path}"
        for block in blocks:
            body = block.strip()
            assert body.startswith(("flowchart ", "graph ", "sequenceDiagram", "stateDiagram"))
            assert body.count("[") == body.count("]")
            assert body.count("(") == body.count(")")
            normalized = re.sub(r"\s+", " ", body)
            assert normalized not in seen, f"duplicate diagram: {path}"
            seen.add(normalized)


def test_lessons_are_topic_specific_and_keep_the_contract():
    forbidden_boilerplate = (
        "This lesson isolates one production-relevant idea",
        "values = torch.tensor([1.0, 2.0, 3.0])",
        "sequence = torch.randn(2, 5, 16)",
    )
    for lesson in LESSONS:
        text = (lesson / "README.md").read_text()
        lower = text.lower()
        assert set(text.splitlines()) >= REQUIRED_HEADINGS
        assert len(text.split()) >= 300, lesson
        assert "```python" in text
        assert "**What to notice:**" in text
        assert not any(fragment in text for fragment in forbidden_boilerplate)
        assert all(keyword in lower for keyword in KEYWORDS[lesson.name[:2]]), lesson


def test_relative_links_and_images_resolve_locally():
    for document in DOCUMENTS:
        for raw_target in markdown_targets(document.read_text()):
            target = raw_target.split("#", 1)[0]
            if not target:
                continue
            if target.startswith(("http://", "https://")):
                if raw_target.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                    raise AssertionError(f"external image in {document}: {raw_target}")
                continue
            resolved = (document.parent / target).resolve()
            assert resolved.is_relative_to(ROOT.resolve()), (document, target)
            assert resolved.exists(), (document, target)
            if target.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                raise AssertionError(f"binary image in {document}: {target}")


def test_quantitative_assets_are_reproducible():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_documentation_assets.py"), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in (
        "llm-lessons-overview.svg",
        "scaling-power-law.svg",
        "bootstrap-confidence-interval.svg",
    ):
        text = (ROOT / "docs" / "assets" / name).read_text()
        assert "<title>" in text and "<desc>" in text


def test_root_links_to_all_lessons_in_order():
    text = (ROOT / "README.md").read_text()
    linked = re.findall(r"\(lessons/(\d{2})_[^/]+/README\.md\)", text)
    assert linked == [f"{number:02d}" for number in range(1, 19)]
