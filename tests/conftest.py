"""Put the core and the Streamlit demo on ``sys.path`` for the suite.

``PYTHONPATH=src`` still collects every test: ``ai_doc`` imports from ``src``,
and the demo modules import from ``apps/demo``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "apps" / "demo", _ROOT / "src"):
    _entry = str(_folder)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


@pytest.fixture(autouse=True)
def _isolate_demo_usage_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep question caps from a previous run out of this test.

    The public demo defaults are small on purpose. The suite asks many
    questions in one process, so each test gets its own count file and a
    high ceiling unless that test sets a lower one.
    """
    monkeypatch.setenv("DEMO_USAGE_STORE", str(tmp_path / "usage.sqlite"))
    monkeypatch.setenv("DEMO_VISITOR_HOURLY_CAP", "100000")
    monkeypatch.setenv("DEMO_DAILY_CAP", "100000")
    from usage_caps import reset_usage_store

    reset_usage_store()
