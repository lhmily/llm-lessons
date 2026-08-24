"""Generate deterministic SVG charts used by the illustrated lessons."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "docs" / "assets"


def svg_document(title: str, description: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 420" width="720" height="420" role="img">
  <title>{title}</title>
  <desc>{description}</desc>
  <rect width="720" height="420" fill="#ffffff"/>
  <g font-family="system-ui, sans-serif" fill="#172033">
{body}
  </g>
</svg>
"""


def axes(x_label: str, y_label: str) -> str:
    return f"""    <line x1="80" y1="350" x2="670" y2="350" stroke="#475569" stroke-width="2"/>
    <line x1="80" y1="350" x2="80" y2="45" stroke="#475569" stroke-width="2"/>
    <text x="375" y="400" text-anchor="middle" font-size="16">{x_label}</text>
    <text x="22" y="200" text-anchor="middle" font-size="16" transform="rotate(-90 22 200)">{y_label}</text>"""


def scaling_svg() -> str:
    compute = [1, 2, 4, 8, 16, 32, 64]
    losses = [1 + 2 * value**-0.5 for value in compute]
    x_values = [80 + math.log2(value) / 6 * 590 for value in compute]
    y_values = [350 - (loss - 1) / 2 * 270 for loss in losses]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_values, y_values, strict=True))
    dots = "\n".join(
        f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>'
        for x, y in zip(x_values, y_values, strict=True)
    )
    labels = "\n".join(
        f'    <text x="{x:.1f}" y="372" text-anchor="middle" font-size="12">{value}</text>'
        for x, value in zip(x_values, compute, strict=True)
    )
    body = f'''    <text x="360" y="28" text-anchor="middle" font-size="20" font-weight="700">Synthetic scaling curve</text>
{axes("Relative compute (log₂ spacing)", "Loss")}
    <line x1="80" y1="350" x2="670" y2="350" stroke="#94a3b8" stroke-dasharray="6 5"/>
    <text x="655" y="340" text-anchor="end" font-size="13">floor = 1</text>
    <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>
{dots}
{labels}
    <text x="110" y="70" font-size="14" fill="#2563eb">loss = 1 + 2 × compute⁻⁰·⁵</text>'''
    return svg_document(
        "Synthetic power-law scaling curve",
        "Loss decreases toward a floor of one as relative compute increases from one to sixty-four.",
        body,
    )


def bootstrap_svg() -> str:
    bins = [2, 8, 20, 42, 65, 82, 65, 42, 20, 8, 2]
    maximum = max(bins)
    bars = []
    for index, count in enumerate(bins):
        x = 105 + index * 48
        height = count / maximum * 250
        bars.append(
            f'    <rect x="{x}" y="{350 - height:.1f}" width="34" height="{height:.1f}" '
            'fill="#0f766e" opacity="0.85"/>'
        )
    body = f"""    <text x="360" y="28" text-anchor="middle" font-size="20" font-weight="700">Bootstrap means</text>
{axes("Resampled mean score", "Frequency")}
{chr(10).join(bars)}
    <line x1="170" y1="350" x2="170" y2="75" stroke="#dc2626" stroke-width="3" stroke-dasharray="6 4"/>
    <line x1="570" y1="350" x2="570" y2="75" stroke="#dc2626" stroke-width="3" stroke-dasharray="6 4"/>
    <text x="170" y="65" text-anchor="middle" font-size="13" fill="#dc2626">2.5%</text>
    <text x="570" y="65" text-anchor="middle" font-size="13" fill="#dc2626">97.5%</text>
    <path d="M170 385 L570 385" stroke="#dc2626" stroke-width="3"/>
    <text x="370" y="408" text-anchor="middle" font-size="14" fill="#dc2626">central 95% interval</text>"""
    return svg_document(
        "Bootstrap confidence interval",
        "A histogram of resampled mean scores with red boundaries marking the central ninety-five percent interval.",
        body,
    )


def overview_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 460" width="1200" height="460" role="img">
  <title>LLM Lessons course overview</title>
  <desc>An eighteen-lesson path from tensor foundations through a tiny GPT to training, alignment, and inference.</desc>
  <rect width="1200" height="460" rx="24" fill="#f8fafc"/>
  <text x="600" y="64" text-anchor="middle" font-family="system-ui, sans-serif" font-size="38" font-weight="700" fill="#172033">Build a tiny GPT from first principles</text>
  <text x="600" y="101" text-anchor="middle" font-family="system-ui, sans-serif" font-size="19" fill="#475569">18 illustrated, tested lessons · Python + PyTorch · runs on an ordinary CPU</text>
  <g font-family="system-ui, sans-serif">
    <rect x="55" y="150" width="230" height="156" rx="18" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
    <text x="170" y="190" text-anchor="middle" font-size="15" font-weight="700" fill="#1d4ed8">LESSONS 1–4</text>
    <text x="170" y="229" text-anchor="middle" font-size="25" font-weight="700" fill="#172033">Foundations</text>
    <text x="170" y="263" text-anchor="middle" font-size="16" fill="#475569">Tensors · gradients</text>
    <text x="170" y="287" text-anchor="middle" font-size="16" fill="#475569">next-token prediction</text>
    <path d="M295 228 H333" stroke="#64748b" stroke-width="4"/>
    <path d="M326 219 L337 228 L326 237" fill="none" stroke="#64748b" stroke-width="4"/>

    <rect x="347" y="150" width="230" height="156" rx="18" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>
    <text x="462" y="190" text-anchor="middle" font-size="15" font-weight="700" fill="#6d28d9">LESSONS 5–9</text>
    <text x="462" y="229" text-anchor="middle" font-size="25" font-weight="700" fill="#172033">Transformers</text>
    <text x="462" y="263" text-anchor="middle" font-size="16" fill="#475569">Tokens · embeddings</text>
    <text x="462" y="287" text-anchor="middle" font-size="16" fill="#475569">attention · blocks</text>
    <path d="M587 228 H625" stroke="#64748b" stroke-width="4"/>
    <path d="M618 219 L629 228 L618 237" fill="none" stroke="#64748b" stroke-width="4"/>

    <rect x="639" y="150" width="230" height="156" rx="18" fill="#ccfbf1" stroke="#0f766e" stroke-width="2"/>
    <text x="754" y="190" text-anchor="middle" font-size="15" font-weight="700" fill="#0f766e">LESSON 10</text>
    <text x="754" y="229" text-anchor="middle" font-size="25" font-weight="700" fill="#172033">TinyGPT</text>
    <text x="754" y="263" text-anchor="middle" font-size="16" fill="#475569">Assemble the complete</text>
    <text x="754" y="287" text-anchor="middle" font-size="16" fill="#475569">causal language model</text>
    <path d="M879 228 H917" stroke="#64748b" stroke-width="4"/>
    <path d="M910 219 L921 228 L910 237" fill="none" stroke="#64748b" stroke-width="4"/>

    <rect x="931" y="150" width="214" height="156" rx="18" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/>
    <text x="1038" y="190" text-anchor="middle" font-size="15" font-weight="700" fill="#c2410c">LESSONS 11–18</text>
    <text x="1038" y="229" text-anchor="middle" font-size="25" font-weight="700" fill="#172033">Apply it</text>
    <text x="1038" y="263" text-anchor="middle" font-size="16" fill="#475569">Train · evaluate · align</text>
    <text x="1038" y="287" text-anchor="middle" font-size="16" fill="#475569">fine-tune · serve</text>
  </g>
  <g font-family="system-ui, sans-serif" font-size="16" font-weight="600">
    <rect x="185" y="356" width="210" height="48" rx="24" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="290" y="386" text-anchor="middle" fill="#334155">No pretrained model</text>
    <rect x="425" y="356" width="160" height="48" rx="24" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="505" y="386" text-anchor="middle" fill="#334155">No API key</text>
    <rect x="615" y="356" width="160" height="48" rx="24" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="695" y="386" text-anchor="middle" fill="#334155">No GPU</text>
    <rect x="805" y="356" width="210" height="48" rx="24" fill="#ffffff" stroke="#cbd5e1"/>
    <text x="910" y="386" text-anchor="middle" fill="#334155">Executable exercises</text>
  </g>
</svg>
"""


def expected_assets() -> dict[Path, str]:
    return {
        ASSETS / "llm-lessons-overview.svg": overview_svg(),
        ASSETS / "scaling-power-law.svg": scaling_svg(),
        ASSETS / "bootstrap-confidence-interval.svg": bootstrap_svg(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify assets without writing")
    args = parser.parse_args()
    failures = []
    for path, expected in expected_assets().items():
        if args.check:
            if not path.exists() or path.read_text() != expected:
                failures.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
            print(f"wrote {path.relative_to(ROOT)}")
    if failures:
        print("out-of-date assets:", ", ".join(map(str, failures)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
