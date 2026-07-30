from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
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


def lesson_directories() -> list[Path]:
    return sorted(
        path for path in (ROOT / "lessons").iterdir() if path.is_dir() and path.name[:2].isdigit()
    )


def public_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names - {"nn", "torch", "F", "Counter"}


def test_all_eighteen_lessons_follow_the_contract():
    lessons = lesson_directories()
    assert [int(path.name[:2]) for path in lessons] == list(range(1, 19))
    for lesson in lessons:
        assert {"README.md", "exercise.py", "solution.py", "test_exercise.py"} <= {
            path.name for path in lesson.iterdir()
        }
        document = (lesson / "README.md").read_text()
        assert set(document.splitlines()) >= REQUIRED_HEADINGS
        assert public_functions(lesson / "exercise.py") == public_functions(lesson / "solution.py")


def test_starters_import_without_executing_todos():
    for lesson in lesson_directories():
        path = lesson / "exercise.py"
        spec = importlib.util.spec_from_file_location(f"starter_{lesson.name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_lessons_do_not_require_network_or_api_credentials():
    forbidden = ("requests.", "urllib.", "http://", "https://api", "api_key")
    for lesson in lesson_directories():
        for path in lesson.glob("*.py"):
            text = path.read_text().lower()
            assert not any(term in text for term in forbidden), path
