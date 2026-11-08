"""merken MCP server — wraps the CLI commands as MCP tools.

Every tool maps 1:1 to a ``Memory`` method. No business logic
lives here; if a tool needs behavior the SDK doesn't have, the
method goes in ``Memory`` first.

Run with::

    python -m merken.mcp_server

or attach to Claude Code::

    claude mcp add merken -- python -m merken.mcp_server

Config, in priority order:

1. Per-tool-call argument (``project`` / ``db`` passed to the tool).
2. Environment variable (``ENGRAM_PROJECT`` / ``ENGRAM_DB``).
3. Built-in defaults (``"default"`` and ``~/.merken/default.db``).

**Note on the default DB path:** it is NOT ``~/.vstash/memory.db``
on purpose. The MCP server opens its own isolated merken store so
a buggy decider can't corrupt your main vstash. To attach merken
to your real vstash from Claude Code, pass
``db=~/.vstash/memory.db`` on the first tool call or set
``ENGRAM_DB=~/.vstash/memory.db`` before starting the server.

Silt's rule, carried across tools: *"before proposing an
algorithm, look at the distribution of the data."* This server
is the thinnest possible wrapper — no algorithms, no heuristics,
no hidden defaults. Every decision a tool call makes is
implemented in the merken SDK and audit-logged there.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from merken import (
    ForgetConsolidated,
    Memory,
    NeverForget,
)

_DEFAULT_PROJECT = "default"

mcp = MCPServer(
    "merken",
    instructions=(
        "merken is a decision-loop layer over vstash. Four primitives: "
        "remember (write), recall (read), consolidate (distill episodic "
        "into semantic facts), forget (tombstone consolidated events). "
        "Every decision is audit-logged. Use these tools the way you'd "
        "use a note-taking agent that can decide what's worth keeping."
    ),
)


# ---------------------------------------------------------- config resolution


def _default_db_path(project: str) -> Path:
    return Path.home() / ".merken" / f"{project}.db"


def _resolve_project(project: str | None) -> str:
    if project:
        return project
    return os.environ.get("ENGRAM_PROJECT", _DEFAULT_PROJECT)


def _resolve_db(db: str | None, project: str) -> Path:
    if db:
        return Path(db).expanduser()
    env_db = os.environ.get("ENGRAM_DB")
    if env_db:
        return Path(env_db).expanduser()
    path = _default_db_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _open_memory(
    project: str | None,
    db: str | None,
    *,
    forget_decider: Any = None,
) -> Memory:
    """Construct a Memory with resolved project + db.

    Each tool call opens its own Memory and closes it on return.
    The per-call overhead is a list() + get_document_chunks() for
    the write decider's hydration — acceptable for an interactive
    MCP flow where calls happen every few seconds, not per ms.
    Callers that want pooling can set ENGRAM_DB once and accept
    that a later call with a different project gets its own DB.
    """
    resolved_project = _resolve_project(project)
    resolved_db = _resolve_db(db, resolved_project)
    if forget_decider is not None:
        return Memory(
            project=resolved_project,
            db=resolved_db,
            forget_decider=forget_decider,
        )
    return Memory(project=resolved_project, db=resolved_db)


# ---------------------------------------------------------------------- tools


@mcp.tool(
    name="merken_remember",
    description=(
        "Write an event to memory. The configured should_remember decider "
        "may skip the write (empty, too short, too long, exact duplicate, "
        "or silently rejected by vstash). Returns the decision either way; "
        "`written` tells you whether the event actually landed in the "
        "vector index."
    ),
)
def merken_remember(
    text: str,
    project: str | None = None,
    db: str | None = None,
    layer: str = "episodic",
    title: str | None = None,
    tags: str | None = None,
) -> dict[str, Any]:
    with _open_memory(project, db) as mem:
        result = mem.remember(text, layer=layer, title=title, tags=tags)
    return {
        "written": result.written,
        "decision": {
            "write": result.decision.write,
            "reason": result.decision.reason,
            "policy": result.decision.policy,
            "confidence": result.decision.confidence,
        },
    }


@mcp.tool(
    name="merken_recall",
    description=(
        "Query memory through the should_recall decider. Default routing "
        "is semantic-first with episodic fallback via round-robin "
        "interleave. Pass an explicit `layer` to bypass the decider and "
        "query one layer only."
    ),
)
def merken_recall(
    query: str,
    project: str | None = None,
    db: str | None = None,
    top_k: int = 5,
    layer: str | None = None,
) -> list[dict[str, Any]]:
    with _open_memory(project, db) as mem:
        hits = mem.recall(query, top_k=top_k, layer=layer)
    return [
        {
            "title": getattr(h, "title", None),
            "path": getattr(h, "path", None),
            "text": getattr(h, "text", None),
            "score": getattr(h, "score", None),
            "chunk": getattr(h, "chunk", None),
        }
        for h in hits
    ]


@mcp.tool(
    name="merken_consolidate",
    description=(
        "Cluster episodic events into semantic facts. Default method is "
        "embedding_v1 with complete linkage at cosine threshold 0.70, "
        "calibrated via grid search on three loop_quality scenarios. "
        "Pass force=True to bypass the should_consolidate decider."
    ),
)
def merken_consolidate(
    project: str | None = None,
    db: str | None = None,
    method: str = "embedding_v1",
    threshold: float = 0.70,
    min_cluster: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    with _open_memory(project, db) as mem:
        result = mem.consolidate(
            method=method,
            embedding_threshold=threshold,
            min_cluster=min_cluster,
            force=force,
        )
    return {
        "events_examined": result.events_examined,
        "facts_written": result.facts_written,
        "skipped": result.skipped,
        "reason": result.reason,
        "decider": result.decider,
        "method": result.method,
        "facts": [
            {
                "text": f.text,
                "cluster_size": f.cluster_size,
                "method": f.method,
                "derived_from": f.derived_from,
            }
            for f in result.facts
        ],
    }


@mcp.tool(
    name="merken_forget",
    description=(
        "Tombstone episodic events. Reversible: the full text is preserved "
        "in the merken_tombstones collection. Deciders: 'never' (safe "
        "default, no-op unless force), 'consolidated' (tombstone events "
        "already in a fact). Pass force=True to tombstone everything "
        "regardless of decider."
    ),
)
def merken_forget(
    project: str | None = None,
    db: str | None = None,
    decider: str = "never",
    min_facts: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    if decider == "consolidated":
        forget_dec = ForgetConsolidated(min_facts=min_facts)
    else:
        forget_dec = NeverForget()

    with _open_memory(project, db, forget_decider=forget_dec) as mem:
        result = mem.forget(force=force)
    return {
        "tombstoned": result.tombstoned,
        "skipped": result.skipped,
        "events_examined": result.events_examined,
        "decider": result.decider,
    }


@mcp.tool(
    name="merken_audit",
    description=(
        "Query the decision audit log. Every should_remember / "
        "should_recall / should_consolidate / should_forget call writes "
        "an audit row with inputs, reason, policy. Use this to answer "
        "'why was this event kept or dropped?'."
    ),
)
def merken_audit(
    query: str = "should_",
    project: str | None = None,
    db: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    with _open_memory(project, db) as mem:
        rows = mem.audit(query=query, top_k=top_k)
    return [
        {
            "title": getattr(r, "title", None),
            "text": getattr(r, "text", None),
            "path": getattr(r, "path", None),
            "score": getattr(r, "score", None),
        }
        for r in rows
    ]


@mcp.tool(
    name="merken_tombstones",
    description=(
        "Query tombstoned (forgotten) events. Each row contains the full "
        "original text, title, layer, tags, and the semantic facts that "
        "preserve the event's content. Use this to answer 'what did I "
        "forget?' or to manually restore a specific event."
    ),
)
def merken_tombstones(
    query: str = "tombstone",
    project: str | None = None,
    db: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    with _open_memory(project, db) as mem:
        rows = mem.tombstones(query=query, top_k=top_k)
    return [
        {
            "title": getattr(r, "title", None),
            "text": getattr(r, "text", None),
            "path": getattr(r, "path", None),
            "score": getattr(r, "score", None),
        }
        for r in rows
    ]


@mcp.tool(
    name="merken_status",
    description=(
        "Project summary: project name, DB path, collection, total event "
        "count, and per-layer breakdown. Cheap — just a list() on the "
        "default collection. Call this first to see what's in the store."
    ),
)
def merken_status(
    project: str | None = None,
    db: str | None = None,
) -> dict[str, Any]:
    from collections import Counter

    resolved_project = _resolve_project(project)
    resolved_db = _resolve_db(db, resolved_project)

    with _open_memory(project, db) as mem:
        docs = mem._vstash.list(collection=mem.collection)
        layers = Counter(d.layer or "(none)" for d in docs)
    return {
        "project": resolved_project,
        "db": str(resolved_db),
        "collection": "default",
        "total_events": sum(layers.values()),
        "layers": dict(layers),
    }


@mcp.tool(
    name="merken_stats",
    description=(
        "Pass-through to vstash.Memory.stats. Reports total document "
        "count across ALL collections, chunk count, collection count, "
        "DB size in MB, and DB path. Useful for monitoring store growth."
    ),
)
def merken_stats(
    project: str | None = None,
    db: str | None = None,
) -> dict[str, Any]:
    with _open_memory(project, db) as mem:
        stats = mem._vstash.stats()
    out = {}
    for field in ("documents", "chunks", "collections", "db_size_mb", "db_path"):
        val = getattr(stats, field, None)
        if val is not None:
            out[field] = val
    return out


# --------------------------------------------------------------------- runner


def main() -> None:
    """Entry point for ``python -m merken.mcp_server``."""
    mcp.run()


if __name__ == "__main__":
    main()
