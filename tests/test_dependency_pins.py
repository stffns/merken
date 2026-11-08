"""Guard rails on dependency ranges that silently break a surface.

``merken.mcp_server`` targets the mcp 2.x SDK (``mcp.server.mcpserver``).
mcp 1.x does not have that module, and the previous open ``mcp>=1.0``
range is how a fresh install ended up on a major the server did not
support. Keep both ends of the range explicit.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.requirements import Requirement

tomllib = pytest.importorskip("tomllib")  # stdlib on 3.11+

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared(name: str) -> Requirement:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    for raw in deps:
        req = Requirement(raw)
        if req.name == name:
            return req
    raise AssertionError(f"{name} not declared in pyproject.toml")


def test_mcp_range_is_v2_only() -> None:
    spec = _declared("mcp").specifier
    assert spec.contains("2.3.0"), f"mcp range {spec} must allow 2.x"
    assert not spec.contains("1.27.0"), (
        f"mcp range {spec} allows 1.x, which lacks mcp.server.mcpserver"
    )
    assert not spec.contains("3.0.0"), f"mcp range {spec} must cap the next major"


def test_installed_mcp_is_v2() -> None:
    assert int(version("mcp").split(".")[0]) == 2, version("mcp")
