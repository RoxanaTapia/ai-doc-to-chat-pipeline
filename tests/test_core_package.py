"""The installable core imports without pulling in the Streamlit demo."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_DIR / "ai_doc"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Top-level modules that live in apps/demo, not in the installable core.
_DEMO_MODULES = frozenset(
    {
        "app",
        "api_client",
        "ask_runtime",
        "page_copy",
        "page_preview",
        "session_upload",
        "sources_panel",
        "ui_theme",
    }
)


def _imported_modules(path: Path) -> list[str]:
    """Module names referenced by import statements in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_core_tree_does_not_import_streamlit() -> None:
    """Static guard: nothing under ai_doc imports Streamlit."""
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        modules = _imported_modules(path)
        if any(name == "streamlit" or name.startswith("streamlit.") for name in modules):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_core_tree_does_not_import_apps() -> None:
    """Static guard: the core does not import the Streamlit demo under apps/."""
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for name in _imported_modules(path):
            top = name.split(".", 1)[0]
            if top == "apps" or top in _DEMO_MODULES:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {name}")
    assert offenders == []


def test_core_imports_without_streamlit() -> None:
    """Import the public package in a fresh interpreter with Streamlit blocked."""
    code = """
import sys
sys.modules["streamlit"] = None
import ai_doc
import ai_doc.rag
import ai_doc.api
import ai_doc.rag.ingestion
from ai_doc import app, generate_answer, generate_answer_stream, resolve_llm_provider_name
assert ai_doc.__version__ == "0.10.0"
assert callable(generate_answer)
assert callable(generate_answer_stream)
assert callable(resolve_llm_provider_name)
assert app is ai_doc.api.app.app
demo = {
    "apps",
    "app",
    "api_client",
    "ask_runtime",
    "page_copy",
    "page_preview",
    "session_upload",
    "sources_panel",
    "ui_theme",
}
loaded = [
    name
    for name in sys.modules
    if name.split(".", 1)[0] in demo or name.startswith("apps.")
]
assert loaded == [], loaded
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env.pop("STREAMLIT_SERVER_PORT", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
