"""Opt-in classifiers must fail open when torch is not installed.

``merken._shadow`` imports torch eagerly (load-order requirement, see
notes/nanogpt-training-log.md Mistake #10) whenever MERKEN_SHADOW /
MERKEN_PRIMARY / MERKEN_LABEL_LLM_MODEL is set. torch is not a declared
dependency, so in a plain install that import used to raise
ModuleNotFoundError at ``import merken`` time and take every write down.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import merken._shadow
from merken import HeuristicWriteDecider, Memory


@pytest.fixture
def no_torch(monkeypatch):
    # A None entry in sys.modules makes ``import torch`` raise ImportError
    # whether or not torch is actually installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    yield
    monkeypatch.undo()
    importlib.reload(merken._shadow)


@pytest.mark.parametrize(
    "env",
    [
        {"MERKEN_SHADOW": "nanogpt"},
        {"MERKEN_PRIMARY": "nanogpt"},
        {"MERKEN_SHADOW": "llm", "MERKEN_SHADOW_LLM_MODEL": "x/y"},
        {"MERKEN_LABEL_LLM_MODEL": "x/y"},
    ],
)
def test_shadow_module_loads_without_torch(no_torch, monkeypatch, env) -> None:
    for k in ("MERKEN_SHADOW", "MERKEN_PRIMARY", "MERKEN_LABEL_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    importlib.reload(merken._shadow)  # raised ModuleNotFoundError before


def test_missing_torch_surfaces_as_runtime_error(no_torch, monkeypatch, capsys) -> None:
    monkeypatch.delenv("MERKEN_PRIMARY", raising=False)
    monkeypatch.setenv("MERKEN_SHADOW", "nanogpt")
    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_CKPT", "/nope/ckpt.pt")
    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_META", "/nope/meta.pkl")

    importlib.reload(merken._shadow)
    assert "torch is not importable" in capsys.readouterr().err

    with pytest.raises(RuntimeError, match="requires torch"):
        merken._shadow.load_shadow_from_env()


def test_memory_falls_back_to_heuristic_without_torch(
    no_torch, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MERKEN_PRIMARY", "nanogpt")
    monkeypatch.setenv("MERKEN_PRIMARY_NANOGPT_CKPT", "/nope/ckpt.pt")
    monkeypatch.setenv("MERKEN_PRIMARY_NANOGPT_META", "/nope/meta.pkl")

    importlib.reload(merken._shadow)

    with Memory(project="t", db=tmp_path / "m.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider)


def test_missing_torch_does_not_block_jev(no_torch, monkeypatch) -> None:
    """jev needs no torch. A failed eager import (here triggered by
    MERKEN_LABEL_LLM_MODEL) must not disable a jev shadow decider."""
    from merken.classifiers.jev import JevWriteDecider

    monkeypatch.delenv("MERKEN_PRIMARY", raising=False)
    monkeypatch.delenv("MERKEN_JEV_LOCAL", raising=False)
    monkeypatch.setenv("MERKEN_LABEL_LLM_MODEL", "x/y")
    monkeypatch.setenv("MERKEN_SHADOW", "jev")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy-key")

    importlib.reload(merken._shadow)
    assert merken._shadow._TORCH_IMPORT_ERROR is not None

    assert isinstance(merken._shadow.load_shadow_from_env(), JevWriteDecider)
