"""``uv.lock`` is tracked and must stay in sync with ``pyproject.toml``.

Without a lockfile every fresh ``uv sync`` re-resolves open ranges
against today's PyPI. That is how mcp 2.x landed silently and broke
``merken-mcp`` (FastMCP was removed in 2.0). With the lock tracked, a
dependency moves only when someone runs ``uv lock`` and commits it.

``uv lock --check`` exits non-zero when the lockfile is missing or no
longer satisfies ``pyproject.toml``: after editing dependencies, run
``uv lock`` and commit the result.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_lockfile_is_present() -> None:
    assert (ROOT / "uv.lock").is_file(), "uv.lock is missing: run `uv lock` and commit it"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_lockfile_matches_pyproject() -> None:
    proc = subprocess.run(
        ["uv", "lock", "--check", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "uv.lock is out of sync with pyproject.toml; run `uv lock` and commit it.\n" + proc.stderr
    )
