"""Guard rails on dependency ranges that silently break a surface.

``merken.mcp_server`` is written against the mcp 1.x SDK
(``mcp.server.fastmcp.FastMCP``). mcp 2.0 renamed FastMCP to
``MCPServer`` and removed the old module, so an unpinned ``mcp>=1.0``
resolves to 2.x on a fresh install and ``merken-mcp`` dies on import.
Until the server is ported, the declared range must exclude 2.x.
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


def test_mcp_range_excludes_v2() -> None:
    spec = _declared("mcp").specifier
    assert spec.contains("1.27.0"), f"mcp range {spec} must still allow 1.x"
    assert not spec.contains("2.0.0"), (
        f"mcp range {spec} allows 2.x, where mcp.server.fastmcp no longer "
        "exists; merken-mcp would crash on import"
    )


def test_installed_mcp_is_v1() -> None:
    major = int(version("mcp").split(".")[0])
    assert major == 1, f"installed mcp {version('mcp')} is not 1.x"


def test_mcp_server_imports() -> None:
    import merken.mcp_server as srv

    assert srv.mcp is not None
