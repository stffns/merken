"""End-to-end: spawn ``python -m merken.mcp_server`` and talk MCP over stdio.

The rest of ``test_mcp_server.py`` calls the tool functions directly, so
it cannot see a broken SDK integration: the server module importing the
wrong SDK name, ``run()`` failing to start the stdio transport, or tool
results that no longer serialise. This test drives the real entry point
the way Claude Code does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "merken_remember",
    "merken_recall",
    "merken_consolidate",
    "merken_forget",
    "merken_audit",
    "merken_tombstones",
    "merken_status",
    "merken_stats",
}


def _payload(result) -> dict:
    assert not result.is_error, result
    if isinstance(result.structured_content, dict):
        sc = result.structured_content
        # Non-object return types get wrapped as {"result": ...}.
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


async def _session_roundtrip(db: Path) -> tuple[set[str], dict, dict]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "merken.mcp_server"],
        env={"ENGRAM_DB": str(db), "ENGRAM_PROJECT": "e2e"},
    )
    with anyio.fail_after(120):
        return await _drive(params)


async def _drive(params: StdioServerParameters) -> tuple[set[str], dict, dict]:
    async with stdio_client(params) as streams, ClientSession(*streams) as session:
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        remembered = _payload(
            await session.call_tool(
                "merken_remember",
                {"text": "Decidimos usar SQLite WAL para la memoria local de merken."},
            )
        )
        status = _payload(await session.call_tool("merken_status", {}))
        return names, remembered, status


def test_stdio_server_lists_tools_and_serves_calls(tmp_path: Path) -> None:
    db = tmp_path / "e2e.db"
    names, remembered, status = anyio.run(_session_roundtrip, db)

    assert names >= EXPECTED_TOOLS, f"missing tools: {EXPECTED_TOOLS - names}"
    assert remembered["written"] is True, remembered
    assert status["project"] == "e2e"
    assert status["db"] == str(db)
    assert status["total_events"] == 1
