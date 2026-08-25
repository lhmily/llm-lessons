"""Prepare the existing course Markdown for the MkDocs site."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / ".site-docs"
SITE_URL = "https://lhmily.github.io/llm-lessons/"
GITHUB_URL = "https://github.com/lhmily/llm-lessons"


@dataclass(frozen=True)
class Lesson:
    number: int
    source: str
    slug: str
    title: str
    description: str

    @property
    def route(self) -> str:
        return f"lessons/{self.slug}/"


LESSONS = (
    Lesson(
        1,
        "01_tensors_and_shapes",
        "tensors-and-shapes",
        "Tensors and Shapes for LLMs",
        "Learn PyTorch tensor shapes, broadcasting, batched linear maps, cosine similarity, "
        "and standardization for language models.",
    ),
    Lesson(
        2,
        "02_autograd_and_optimization",
        "autograd-and-optimization",
        "Autograd and Optimization in PyTorch",
        "Understand gradients, finite differences, PyTorch autograd, and gradient descent by implementing the foundations of LLM training.",
    ),
    Lesson(
        3,
        "03_neural_networks",
        "neural-networks",
        "Neural Networks from First Principles",
        "Build neural-network layers, activations, cross-entropy loss, and a small multilayer perceptron with PyTorch.",
    ),
    Lesson(
        4,
        "04_next_token_prediction",
        "next-token-prediction",
        "Next-Token Prediction Explained",
        "Learn how context windows, shifted targets, logits, and cross-entropy turn text into the language-model training objective.",
    ),
    Lesson(
        5,
        "05_tokenization",
        "tokenization",
        "LLM Tokenization and Byte Pair Encoding",
        "Implement UTF-8 byte tokenization and deterministic byte pair encoding to understand how language models represent text.",
    ),
    Lesson(
        6,
        "06_embeddings_and_positions",
        "embeddings-and-positions",
        "Token Embeddings and Positional Encodings",
        "Implement token lookup tables and positional encodings, and trace the tensor shapes used as Transformer inputs.",
    ),
    Lesson(
        7,
        "07_attention",
        "attention",
        "Self-Attention from Scratch in PyTorch",
        "Derive and implement queries, keys, values, scaled dot-product attention, softmax, and causal masking in PyTorch.",
    ),
    Lesson(
        8,
        "08_multi_head_attention",
        "multi-head-attention",
        "Multi-Head Attention from Scratch",
        "Split, run, and merge parallel attention heads while tracking every PyTorch tensor shape in multi-head attention.",
    ),
    Lesson(
        9,
        "09_transformer_block",
        "transformer-block",
        "Build a Transformer Block",
        "Combine causal attention, RMS normalization, residual connections, and feed-forward layers into a pre-norm Transformer block.",
    ),
    Lesson(
        10,
        "10_tiny_gpt",
        "tiny-gpt",
        "Build a Tiny GPT from Scratch",
        "Assemble embeddings, causal Transformer blocks, weight tying, logits, and next-token loss into a complete tiny GPT model.",
    ),
    Lesson(
        11,
        "11_training",
        "training",
        "Train a Tiny GPT with PyTorch",
        "Train a tiny GPT with AdamW, gradient clipping, evaluation, deterministic batches, and checkpoint save and restore.",
    ),
    Lesson(
        12,
        "12_generation",
        "text-generation",
        "LLM Text Generation and Sampling",
        "Implement greedy decoding, temperature, top-k sampling, and top-p sampling for autoregressive language generation.",
    ),
    Lesson(
        13,
        "13_kv_cache",
        "kv-cache",
        "KV Cache for Faster LLM Decoding",
        "Understand and implement a key-value cache that avoids repeated attention computation during autoregressive decoding.",
    ),
    Lesson(
        14,
        "14_scaling_and_compute",
        "scaling-and-compute",
        "LLM Scaling, Memory, and Compute",
        "Estimate language-model parameters, memory, FLOPs, and synthetic scaling curves to reason about training compute.",
    ),
    Lesson(
        15,
        "15_fine_tuning_and_lora",
        "fine-tuning-and-lora",
        "Fine-Tuning and LoRA from Scratch",
        "Learn response masking and implement low-rank adaptation to fine-tune language models with fewer trainable parameters.",
    ),
    Lesson(
        16,
        "16_evaluation",
        "evaluation",
        "Evaluate Language Models",
        "Measure perplexity and task metrics, then use bootstrap confidence intervals to evaluate language-model quality responsibly.",
    ),
    Lesson(
        17,
        "17_alignment",
        "alignment-with-dpo",
        "LLM Alignment and DPO Fundamentals",
        "Understand preference pairs and implement the core Direct Preference Optimization loss for language-model alignment.",
    ),
    Lesson(
        18,
        "18_inference",
        "inference-and-serving",
        "LLM Inference and Serving",
        "Implement int8 quantization, storage accounting, and dynamic batching while studying inference latency and memory trade-offs.",
    ),
)

SOURCE_TO_ROUTE = {lesson.source: lesson.route for lesson in LESSONS}


def front_matter(title: str, description: str) -> str:
    return f"---\ntitle: {json.dumps(title)}\ndescription: {json.dumps(description)}\n---\n\n"


def rewrite_markdown(text: str, *, lesson: Lesson | None = None) -> str:
    for source, route in SOURCE_TO_ROUTE.items():
        target = f"../{route.removeprefix('lessons/')}index.md" if lesson else f"{route}index.md"
        text = re.sub(
            rf"(?:\.\./|lessons/){re.escape(source)}/README\.md",
            target,
            text,
        )
    text = text.replace("../../docs/assets/", "../../assets/")
    text = text.replace('src="docs/assets/', 'src="assets/')
    text = text.replace("(CONTRIBUTING.md)", f"({GITHUB_URL}/blob/main/CONTRIBUTING.md)")
    text = text.replace("(CHANGELOG.md)", f"({GITHUB_URL}/blob/main/CHANGELOG.md)")
    text = text.replace("(LICENSE)", f"({GITHUB_URL}/blob/main/LICENSE)")
    text = text.replace('href="LICENSE"', f'href="{GITHUB_URL}/blob/main/LICENSE"')

    if lesson is not None:
        index = lesson.number - 1
        links = ["[Course overview](../../index.md)"]
        if index > 0:
            previous = LESSONS[index - 1]
            links.append(
                f"[← Lesson {previous.number}: {previous.title}](../{previous.slug}/index.md)"
            )
        if index + 1 < len(LESSONS):
            following = LESSONS[index + 1]
            links.append(
                f"[Lesson {following.number}: {following.title} →](../{following.slug}/index.md)"
            )
        text += "\n\n---\n\n" + " · ".join(links) + "\n"
    return text


def prepare(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    homepage = (ROOT / "README.md").read_text()
    homepage = rewrite_markdown(homepage)
    description = "Build a tiny GPT from first principles with 18 illustrated, tested Python and PyTorch lessons—no GPU or API key required."
    (output / "index.md").write_text(
        front_matter("LLM Lessons: Build a GPT from Scratch", description) + homepage
    )

    for lesson in LESSONS:
        source = ROOT / "lessons" / lesson.source / "README.md"
        destination = output / "lessons" / lesson.slug / "index.md"
        destination.parent.mkdir(parents=True)
        body = rewrite_markdown(source.read_text(), lesson=lesson)
        destination.write_text(front_matter(lesson.title, lesson.description) + body)

    shutil.copytree(ROOT / "docs" / "assets", output / "assets")

    styles = output / "stylesheets" / "extra.css"
    styles.parent.mkdir()
    styles.write_text(
        ".md-typeset h1 { font-weight: 700; }\n.md-typeset img { border-radius: 0.4rem; }\n"
    )

    mathjax = output / "javascripts" / "mathjax.js"
    mathjax.parent.mkdir()
    mathjax.write_text(
        "window.MathJax = {tex: {inlineMath: [['\\\\(', '\\\\)']], "
        "displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]}, options: {ignoreHtmlClass: '.*|', "
        "processHtmlClass: 'arithmatex'}};\n"
    )

    (output / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.output.resolve())
    print(f"prepared {len(LESSONS)} lessons in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
