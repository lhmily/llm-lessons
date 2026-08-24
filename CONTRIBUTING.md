# Contributing to LLM Lessons

Thank you for helping make language-model internals easier to learn. Contributions that improve correctness, clarity, accessibility, or the exercise experience are welcome.

## Set up the project

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lhmily/llm-lessons.git
cd llm-lessons
uv sync --locked --dev
```

## Choose a contribution

Good contributions include:

- correcting an explanation, derivation, diagram, or test;
- adding a focused example or optional experiment;
- improving setup instructions on a supported operating system;
- making an exercise or error message easier to understand; and
- improving accessibility without adding external runtime dependencies.

For a substantial new lesson or a change to the curriculum structure, open an issue before investing in an implementation.

## Follow the lesson contract

Each numbered lesson contains:

- `README.md` with the standard teaching sections and at least one Mermaid diagram;
- `exercise.py` with typed starter functions and explicit `TODO`s;
- `solution.py` with a readable reference implementation; and
- `test_exercise.py` with behavior and mathematical-invariant checks.

Keep examples small, deterministic, CPU-friendly, and runnable without a network connection or API credentials. Prefer explanations and implementations that expose the underlying operation over wrappers that hide it.

## Validate your change

Run the full reference implementation and quality checks:

```bash
LESSON_IMPL=solution uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If you change a generated chart, update and verify the assets:

```bash
uv run python scripts/generate_documentation_assets.py
uv run python scripts/generate_documentation_assets.py --check
```

An unfinished `exercise.py` is expected to raise `NotImplementedError`; it must still import successfully.

## Open a pull request

Keep each pull request focused. In its description, explain:

1. what changed and why;
2. which lessons or learners are affected; and
3. which validation commands you ran.

By contributing, you agree that your contribution is licensed under the repository's [MIT License](LICENSE).
