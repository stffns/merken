"""Tests for the merken MCP server.

The MCP transport (stdio, SSE, etc.) is handled by the official
MCP Python SDK (``MCPServer``, mcp 2.x) — we don't test that here. What we test is that:

  1. Every tool is registered on the server with the expected name.
  2. Calling each tool function directly against a tmp_path DB
     produces the expected shape of output.
  3. Environment-variable config resolution works.
  4. The per-call ``project``/``db`` overrides take priority.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from merken.mcp_server import (
    _resolve_db,
    _resolve_project,
    merken_audit,
    merken_consolidate,
    merken_forget,
    merken_recall,
    merken_remember,
    merken_status,
    merken_tombstones,
    mcp,
)


# ------------------------------------------------------------------- registry


async def _list_tool_names() -> set[str]:
    tools = await mcp.list_tools()
    return {t.name for t in tools}


def test_server_registers_expected_tools() -> None:
    tool_names = asyncio.run(_list_tool_names())
    expected = {
        "merken_remember",
        "merken_recall",
        "merken_consolidate",
        "merken_forget",
        "merken_audit",
        "merken_tombstones",
        "merken_status",
        "merken_stats",
    }
    assert expected.issubset(tool_names), (
        f"missing tools: {expected - tool_names}"
    )


# ------------------------------------------------------------ config resolution


def test_resolve_project_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ENGRAM_PROJECT", raising=False)
    assert _resolve_project(None) == "default"


def test_resolve_project_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_PROJECT", "from_env")
    assert _resolve_project(None) == "from_env"


def test_resolve_project_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_PROJECT", "from_env")
    assert _resolve_project("explicit") == "explicit"


def test_resolve_db_from_arg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ENGRAM_DB", raising=False)
    db = tmp_path / "custom.db"
    resolved = _resolve_db(str(db), "any_project")
    assert resolved == db


def test_resolve_db_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ENGRAM_DB", raising=False)
    db = tmp_path / "env_db.db"
    monkeypatch.setenv("ENGRAM_DB", str(db))
    resolved = _resolve_db(None, "any_project")
    assert resolved == db


def test_resolve_db_default_isolated_from_vstash(monkeypatch) -> None:
    monkeypatch.delenv("ENGRAM_DB", raising=False)
    resolved = _resolve_db(None, "my_project")
    assert str(resolved).endswith("/.merken/my_project.db")
    assert ".vstash" not in str(resolved)


# ---------------------------------------------------------------- merken_remember


def test_remember_writes_and_returns_decision(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    result = merken_remember(
        text="an mcp test event long enough to pass vstash guardrails here",
        project="mcp_test",
        db=db,
    )
    assert result["written"] is True
    assert result["decision"]["reason"] == "novel"
    assert result["decision"]["policy"] == "HeuristicWriteDecider"


def test_remember_skips_duplicate_across_calls(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    text = "an mcp dedup test event that is definitely long enough for vstash"
    r1 = merken_remember(text=text, project="mcp_dedup", db=db)
    r2 = merken_remember(text=text, project="mcp_dedup", db=db)
    assert r1["written"] is True
    assert r2["written"] is False
    assert r2["decision"]["reason"] == "dup_exact"


def test_remember_surfaces_vstash_rejection(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    result = merken_remember(text="abcdefghi", project="mcp_short", db=db)
    assert result["written"] is False
    assert "vstash_rejected" in result["decision"]["reason"]


# ----------------------------------------------------------------- merken_recall


def test_recall_finds_remembered_event(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="the mcp kafka merchant pipeline discussion about backpressure",
        project="mcp_recall",
        db=db,
    )
    hits = merken_recall(
        query="kafka merchant pipeline", project="mcp_recall", db=db
    )
    assert hits
    assert any("kafka" in (h["text"] or "").lower() for h in hits)


def test_recall_empty_db_returns_empty_list(tmp_path: Path) -> None:
    hits = merken_recall(
        query="anything",
        project="mcp_empty",
        db=str(tmp_path / "e.db"),
    )
    assert hits == []


def test_recall_explicit_layer_passes_through(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="an event for the explicit layer recall test on the mcp server",
        project="mcp_layer",
        db=db,
        layer="episodic",
    )
    hits = merken_recall(
        query="explicit layer",
        project="mcp_layer",
        db=db,
        layer="episodic",
    )
    assert hits


# ------------------------------------------------------------- merken_consolidate


def test_consolidate_empty_db_skips(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    result = merken_consolidate(project="mcp_cons", db=db)
    assert result["skipped"] is True
    assert result["events_examined"] == 0


def test_consolidate_force_builds_fact(tmp_path: Path) -> None:
    """Uses a text pair pre-measured to cluster above 0.70 cosine in
    BOTH bge-small-en-v1.5 (0.903) and paraphrase-multilingual-MiniLM-L12-v2
    (0.841). Real vstash 0.27.0 sometimes resolves to multilingual despite
    the config default being bge-small, so test pairs must be model-
    independent to avoid flakes on different environments."""
    db = str(tmp_path / "e.db")
    merken_remember(
        text=(
            "PostgreSQL 16 was chosen over SQLite for the analytics warehouse "
            "because of concurrent write requirements."
        ),
        project="mcp_cons_force",
        db=db,
    )
    merken_remember(
        text=(
            "The analytics warehouse runs on PostgreSQL 16; SQLite was ruled "
            "out because of write concurrency concerns."
        ),
        project="mcp_cons_force",
        db=db,
    )
    result = merken_consolidate(
        project="mcp_cons_force",
        db=db,
        force=True,
    )
    assert result["skipped"] is False
    assert result["events_examined"] == 2
    assert result["facts_written"] == 1
    assert len(result["facts"]) == 1
    assert result["facts"][0]["cluster_size"] == 2


# ----------------------------------------------------------------- merken_forget


def test_forget_default_never_is_noop(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="an event that should not be forgotten by default never mode",
        project="mcp_forget_noop",
        db=db,
    )
    result = merken_forget(project="mcp_forget_noop", db=db)
    assert result["tombstoned"] == []
    assert result["decider"] == "NeverForget"


def test_forget_force_tombstones_everything(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="the first event about marine biology research for mcp forget force test",
        project="mcp_forget_force",
        db=db,
    )
    merken_remember(
        text="the second event about volcanic geology research for mcp forget force test",
        project="mcp_forget_force",
        db=db,
    )
    result = merken_forget(project="mcp_forget_force", db=db, force=True)
    assert len(result["tombstoned"]) == 2


def test_forget_consolidated_mode(tmp_path: Path) -> None:
    """Same model-independent text pair as test_consolidate_force_builds_fact."""
    db = str(tmp_path / "e.db")
    merken_remember(
        text=(
            "PostgreSQL 16 was chosen over SQLite for the analytics warehouse "
            "because of concurrent write requirements."
        ),
        project="mcp_forget_cons",
        db=db,
    )
    merken_remember(
        text=(
            "The analytics warehouse runs on PostgreSQL 16; SQLite was ruled "
            "out because of write concurrency concerns."
        ),
        project="mcp_forget_cons",
        db=db,
    )
    merken_consolidate(project="mcp_forget_cons", db=db, force=True)
    result = merken_forget(
        project="mcp_forget_cons",
        db=db,
        decider="consolidated",
    )
    assert len(result["tombstoned"]) == 2
    assert result["decider"] == "ForgetConsolidated"


# ------------------------------------------------------------------- merken_audit


def test_audit_returns_decision_rows(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="an audit tool test event about polymer chemistry for mcp server",
        project="mcp_audit",
        db=db,
    )
    rows = merken_audit(query="should_remember", project="mcp_audit", db=db)
    assert rows
    bodies = " ".join((r["text"] or "") for r in rows)
    assert "should_remember" in bodies


# ------------------------------------------------------------- merken_tombstones


def test_tombstones_after_forget(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="a distinctive event about quasars for the mcp tombstone test",
        project="mcp_tomb",
        db=db,
    )
    merken_forget(project="mcp_tomb", db=db, force=True)
    rows = merken_tombstones(query="quasars", project="mcp_tomb", db=db)
    assert rows
    assert any("quasar" in (r["text"] or "").lower() for r in rows)


# ------------------------------------------------------------------ merken_status


def test_status_reports_counts(tmp_path: Path) -> None:
    db = str(tmp_path / "e.db")
    merken_remember(
        text="the first status test event about analytics and reporting for mcp",
        project="mcp_status",
        db=db,
    )
    merken_remember(
        text="the second status test event about release planning for mcp server",
        project="mcp_status",
        db=db,
    )
    result = merken_status(project="mcp_status", db=db)
    assert result["project"] == "mcp_status"
    assert result["total_events"] == 2
    assert result["layers"]["episodic"] == 2


def test_status_honors_env_project(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_PROJECT", "from_env_mcp")
    merken_remember(
        text="an env-project test event for the mcp server resolver check",
        db=str(tmp_path / "e.db"),
    )
    result = merken_status(db=str(tmp_path / "e.db"))
    assert result["project"] == "from_env_mcp"
