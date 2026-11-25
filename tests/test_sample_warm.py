"""Sample NDA index is built once per process and reused for the first question."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

import warm_start  # noqa: E402

import sample_warm  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_STUB_EMBEDDINGS", "1")
    sample_warm._EMBEDDINGS.clear()
    sample_warm._INDEXES.clear()


def test_warm_index_is_built_once_and_searchable() -> None:
    calls = {"n": 0}
    real = sample_warm._build_sample_index

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    sample_warm._build_sample_index = _wrapped
    try:
        first = sample_warm.warm_sample_index()
        second = sample_warm.warm_sample_index()
    finally:
        sample_warm._build_sample_index = real

    assert first is not None
    assert first is second
    assert calls["n"] == 1
    assert first["vector_store"].index.ntotal > 0
    hits = first["vector_store"].similarity_search("How long does this agreement last?", k=2)
    assert hits
    assert calls["n"] == 1


def test_embeddings_stay_in_the_process() -> None:
    model_name, _size, _overlap, _headers = sample_warm.index_settings()
    first = sample_warm.embeddings_for(model_name)
    second = sample_warm.embeddings_for(model_name)
    assert first is second


def test_stub_index_is_not_the_real_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    model_name, size, overlap, headers = sample_warm.index_settings()
    stub_key = sample_warm._index_key(model_name, size, overlap, headers)
    monkeypatch.setenv("USE_STUB_EMBEDDINGS", "0")
    real_key = sample_warm._index_key(model_name, size, overlap, headers)
    assert stub_key != real_key


def test_missing_sample_does_not_stop_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sample_warm, "sample_nda_path", lambda: tmp_path / "missing.pdf")
    assert sample_warm.warm_sample_index() is None


def test_server_starts_after_the_index_is_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _warm():
        calls.append("warm")
        return {"chunks": [object()]}

    def _serve(script: Path, args: list[str]) -> None:
        calls.append(("serve", script.name, args))

    monkeypatch.setattr(sample_warm, "warm_sample_index", _warm)
    monkeypatch.setattr(warm_start, "serve_streamlit", _serve)
    previous = Path.cwd()
    try:
        warm_start.main(["--server.port=9"])
    finally:
        os.chdir(previous)

    assert calls[0] == "warm"
    assert calls[1][0] == "serve"
    assert calls[1][1] == "app.py"
    assert calls[1][2] == ["--server.port=9"]
