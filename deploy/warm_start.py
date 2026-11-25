#!/usr/bin/env python3
"""Warm the sample NDA index, then start Streamlit in this same process.

The index and embedding model live on this process. Starting Streamlit
in-process keeps them ready, so the first question does not build them.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("warm_start")


def prepare_import_path() -> Path:
    """Put the demo and the core on the import path before either is loaded."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    demo = root / "apps" / "demo"
    for entry in (str(demo), str(src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [str(src), str(demo)]
    if pythonpath:
        parts.append(pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return root


def serve_streamlit(script: Path, args: list[str]) -> None:
    """Hand off to Streamlit without leaving this process (the warm cache stays)."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(script), *args]
    stcli.main()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = list(sys.argv[1:] if argv is None else argv)
    root = prepare_import_path()
    os.chdir(root)
    from sample_warm import warm_sample_index

    started = time.perf_counter()
    bundle = warm_sample_index()
    if bundle is not None:
        logger.info(
            "Sample NDA index ready (%s chunks, %.2fs). Starting the demo.",
            len(bundle["chunks"]),
            time.perf_counter() - started,
        )
    serve_streamlit(root / "apps" / "demo" / "app.py", args)


if __name__ == "__main__":
    main()
