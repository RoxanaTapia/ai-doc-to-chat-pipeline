"""Put the core and the Streamlit demo on ``sys.path`` for the suite.

``PYTHONPATH=src`` still collects every test: ``ai_doc`` imports from ``src``,
and the demo modules import from ``apps/demo``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "apps" / "demo", _ROOT / "src"):
    _entry = str(_folder)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
