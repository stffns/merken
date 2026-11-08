# merken MCP server reference

The MCP server (`merken-mcp`) exposes merken's eight decision
primitives as Model Context Protocol tools. With it attached to
Claude Code (or any MCP client), Claude can call merken during a
conversation — you never have to manually type
`merken remember` or `merken recall`.

## Install

The MCP server ships with merken. After `pip install -e .`:

```bash
which merken-mcp
# /Users/you/.pyenv/shims/merken-mcp

merken-mcp --help
# MCPServer (mcp 2.x) entry point — takes no args, speaks MCP over stdio
```

Alternate invocation without the entry point:

```bash
python -m merken.mcp_server
```

## Attach to Claude Code

```bash
claude mcp add merken -- merken-mcp
```

Verify:

```bash
claude mcp list
# merken  enabled  stdio  merken-mcp
```

Then **in a Claude Code session**, Claude can call merken tools
whenever it thinks it should. You don't have to name them — you
just talk naturally:

> **You:** "Claude, remember that we decided to use Postgres 16
> for the analytics warehouse on 2026-04-08 because of write
> concurrency."
>
> **Claude:** *calls `merken_remember(text="we decided to use
> Postgres 16...", layer="episodic")`*
>
> **You:** "What did we decide about the analytics warehouse
> database?"
>
> **Claude:** *calls `merken_recall(query="analytics warehouse
> database decision")`, formats the hits in context*

## Configuration

merken-mcp has no command-line flags. Configuration is via
**environment variables** set before the process starts:

| Variable | Default | Meaning |
|---|---|---|
| `ENGRAM_PROJECT` | `default` | Default project name for every tool call that doesn't pass `project` explicitly. |
| `ENGRAM_DB` | `~/.merken/<project>.db` | Default DB path. Same isolation rule as the CLI — deliberately separate from `~/.vstash/memory.db`. |

Every tool also accepts `project` and `db` as optional
per-call parameters. The priority is **per-call > env > default**.

### Attaching to your real vstash

By default merken-mcp writes to a project-isolated DB. To
attach merken to your live vstash instead:

```bash
export ENGRAM_DB=~/.vstash/memory.db
claude mcp add merken -- merken-mcp
```

Or set it in Claude Code's settings.json under the merken
server's env block:

```json
{
  "mcpServers": {
    "merken": {
      "command": "merken-mcp",
      "env": {
        "ENGRAM_DB": "/Users/you/.vstash/memory.db",
        "ENGRAM_PROJECT": "main"
      }
    }
  }
}
```

**Be careful with this.** merken creates its own collections
(`merken_audit`, `merken_tombstones`) in any DB it opens. That's
safe, but if you share a DB with other consumers, they'll see
those collections. merken never touches vstash internals or
writes to collections it doesn't own.

## Tools

Eight tools total, one per merken primitive. Every tool returns
a JSON-serializable dict or list.

### `merken_remember`

Write an event to memory.

**Parameters:**

```
text     (required, string)  Event text
project  (string, optional)  Override default project
db       (string, optional)  Override default DB path
layer    (string, default "episodic")
title    (string, optional)  Document title
tags     (string, optional)  Comma-separated tags
```

**Returns:**

```json
{
  "written": true,
  "decision": {
    "write": true,
    "reason": "novel",
    "policy": "HeuristicWriteDecider",
    "confidence": 0.8
  }
}
```

**When Claude should call it:** when the user says "remember",
"note that", "save this", or when Claude wants to preserve a
decision or finding for later sessions.

**What the user feels:** nothing directly — merken's write is
silent unless Claude surfaces the result. The value shows up
later when `merken_recall` surfaces the memory.

### `merken_recall`

Query memory through the should_recall decider (or an explicit
layer).

**Parameters:**

```
query    (required, string)  Natural-language query
project  (string, optional)
db       (string, optional)
top_k    (int, default 5)
layer    (string, optional)  Restrict to one layer; bypasses
                             the should_recall decider
```

**Returns:**

```json
[
  {
    "title": "fact_a1b2c3d4",
    "path": "text://fact_a1b2c3d4",
    "text": "[observed 3×] The team picked Postgres 16 ...",
    "score": 0.0166,
    "chunk": 0
  },
  {
    "title": "the-team-picked-postgres-...",
    "path": "text://the-team-picked-postgres-...",
    "text": "The team picked Postgres for the analytics warehouse...",
    "score": 0.0161,
    "chunk": 0
  }
]
```

**When Claude should call it:** whenever answering a question
about previous sessions or decisions, whenever the user
references something said earlier, whenever context might help.

### `merken_consolidate`

Cluster episodic events into semantic facts.

**Parameters:**

```
project    (string, optional)
db         (string, optional)
method     (string, default "embedding_v1")
threshold  (float, default 0.70)
min_cluster (int, default 2)
force      (bool, default false)  Bypass should_consolidate decider
```

**Returns:**

```json
{
  "events_examined": 12,
  "facts_written": 3,
  "skipped": false,
  "reason": "clustered_embedding>=0.7_linkage=complete_mincluster=2",
  "decider": "PeriodicConsolidator",
  "method": "embedding_v1",
  "facts": [
    {
      "text": "[observed 4×] The team picked Postgres for ...",
      "cluster_size": 4,
      "method": "concat_v1",
      "derived_from": ["text://...", "text://...", "text://...", "text://..."]
    }
  ]
}
```

**When Claude should call it:** at the end of a long session,
when the user says "summarize what we discussed", or when
recall starts returning stale / scattered results. `force=true`
is the right choice for explicit summarization requests.

### `merken_forget`

Tombstone episodic events. Reversible via `merken_tombstones`.

**Parameters:**

```
project    (string, optional)
db         (string, optional)
decider    (string, default "never")  "never" or "consolidated"
min_facts  (int, default 1)            For "consolidated"
force      (bool, default false)       Tombstone everything
```

**Returns:**

```json
{
  "tombstoned": ["text://...", "text://..."],
  "skipped": [
    ["text://...", "not_consolidated:0<1"]
  ],
  "events_examined": 8,
  "decider": "ForgetConsolidated"
}
```

**When Claude should call it:** when the user says "forget
what we discussed about X" or "clean up after consolidation."
Default `decider="never"` means this tool is a no-op unless
Claude explicitly opts in.

**Safety:** tombstones preserve the full text. Claude can
recover a forgotten event via `merken_tombstones` and
`merken_remember` if the user asks.

### `merken_audit`

Query the decision audit log.

**Parameters:**

```
query    (string, default "should_")
project  (string, optional)
db       (string, optional)
top_k    (int, default 20)
```

**Returns:**

```json
[
  {
    "title": "audit:should_remember:novel:2026-04-09T11:04:33+00:00",
    "path": "text://audit:...",
    "text": "timestamp: 2026-04-09T11:04:33+00:00\ndecision: should_remember\n...",
    "score": 0.0166
  }
]
```

**When Claude should call it:** when the user asks "why did
merken keep / drop / skip X?" or when debugging unexpected
recall behavior.

### `merken_tombstones`

Query forgotten events.

**Parameters:**

```
query    (string, default "tombstone")
project  (string, optional)
db       (string, optional)
top_k    (int, default 20)
```

**Returns:** same shape as `merken_audit` — each row has a
title + full preserved text body.

**When Claude should call it:** when the user asks "what did I
forget?" or "recover the X from yesterday that I told you to
forget."

### `merken_status`

Project summary: DB path, total event count, layer breakdown.

**Parameters:**

```
project  (string, optional)
db       (string, optional)
```

**Returns:**

```json
{
  "project": "default",
  "db": "/Users/you/.merken/default.db",
  "collection": "default",
  "total_events": 47,
  "layers": {
    "episodic": 42,
    "semantic": 5
  }
}
```

**When Claude should call it:** first call of a new session
(to show what's in store), or when the user asks "what do we
have saved?"

### `merken_stats`

Pass-through to `vstash.Memory.stats()`. Reports document,
chunk, and collection counts across the whole DB.

**Parameters:**

```
project  (string, optional)
db       (string, optional)
```

**Returns:**

```json
{
  "documents": 52,
  "chunks": 134,
  "collections": 3,
  "db_size_mb": 2.04,
  "db_path": "/Users/you/.merken/default.db"
}
```

**When Claude should call it:** rarely — this is more operational
than conversational. Useful when the user asks "how big is my
memory?"

## A typical Claude Code flow

What the loop looks like when merken is attached:

```
session start
    │
    ▼
Claude calls merken_status ───► knows what's in memory
    │
    ▼
user asks a question
    │
    ▼
Claude calls merken_recall ────► gets relevant facts + events
    │
    ▼
Claude integrates the hits into its answer
    │
    ▼
user mentions a new decision
    │
    ▼
Claude calls merken_remember ──► audit row, layer=episodic
    │
    ▼
... many turns ...
    │
    ▼
end of session / before context compaction
    │
    ▼
Claude calls merken_consolidate ─► episodic → semantic facts
    │
    ▼
(optional) Claude calls merken_forget with decider="consolidated"
    │
    ▼
next session
    │
    ▼
Claude calls merken_recall — now sees consolidated facts first
```

This loop is what CLAUDE.md's "what IS next" section refers to
when it says "Claude Code hooks": two automatic triggers (save
on Stop, consolidate on PreCompact) that would close the loop
without Claude having to remember to call the tools manually.
Hooks aren't implemented yet but the MCP tools they'd call are.

## MCP tool naming convention

All eight tools are prefixed with `merken_`. This is
deliberate: if you attach multiple MCP servers to Claude Code,
each server's tools are namespaced so they coexist. Claude sees
`merken_remember` vs `other_server_remember` as distinct tools.

## Testing the server

merken's test suite imports each tool function directly and
exercises it against a tmp_path DB — see `tests/test_mcp_server.py`
for 22 tests that cover every tool, the config resolution chain, and
the dedup/guardrail invariants. `tests/test_mcp_stdio_e2e.py` spawns
`python -m merken.mcp_server` and drives it over stdio with the SDK's
`ClientSession` (initialize, list_tools, call_tool), so an SDK
integration break shows up in the suite instead of in Claude Code.

The server targets the mcp 2.x SDK (`mcp.server.mcpserver.MCPServer`;
FastMCP was renamed in 2.0). The dependency is pinned `mcp>=2.0,<3`.

Smoke test (start and signal):

```bash
(merken-mcp </dev/null & PID=$!; sleep 2; kill -INT $PID; wait $PID)
# should exit 0
```

If that doesn't work, check that:
1. `pip install -e .` was run in the merken repo
2. `which merken-mcp` returns a path under your pyenv shims
3. The `mcp` Python package is installed (`pip show mcp`)

## Config resolution in detail

When a tool call comes in, merken resolves project and DB path
in this order:

```
_resolve_project(project):
    1. If `project` argument passed to the tool, use it
    2. Else if $ENGRAM_PROJECT env var set, use it
    3. Else use "default"

_resolve_db(db, project):
    1. If `db` argument passed to the tool, expand ~ and use it
    2. Else if $ENGRAM_DB env var set, expand ~ and use it
    3. Else use ~/.merken/<project>.db
       (mkdir -p the parent directory if needed)
```

Each tool call opens a fresh `Memory` and closes it on return.
The per-call overhead is one `list()` + some
`get_document_chunks()` for the write decider's dedup hydration,
which is acceptable for interactive MCP flows (calls happen
every few seconds, not per millisecond). If a caller wants
connection pooling, set `$ENGRAM_DB` once and every call uses
the same DB — but the Memory object itself is still constructed
per call.

## Troubleshooting

**Claude Code doesn't see the server.** Run `claude mcp list`.
If `merken` isn't there, re-run `claude mcp add merken -- merken-mcp`.
If it *is* there but Claude can't call tools, check Claude
Code's MCP logs (usually in `~/.claude/logs/` or similar).

**The server starts but tools never fire.** Confirm the
server is actually running when Claude Code opens the session:

```bash
ps aux | grep merken-mcp
```

If nothing shows up, Claude Code may have failed to spawn the
server. Check stderr in Claude Code's MCP log.

**Tool calls return errors.** Check the merken audit log — even
failed operations write audit rows:

```bash
merken audit
merken audit error
```

If audit is empty, the server never got the call or the DB
path is wrong. Use `merken_status` as the first test — it
should always succeed and tell you the resolved DB path.

**Engram is using the wrong project.** Check `$ENGRAM_PROJECT`
in the environment Claude Code launched the server with. You
may need to set it in Claude Code's settings.json env block,
not just in your shell.

## Further reading

- [`cli.md`](cli.md) — same commands as a shell CLI
- [`primitives.md`](primitives.md) — what each tool decides
  under the hood
- [`architecture.md`](architecture.md) — the memory model
  every tool shares
- `merken/mcp_server.py` — the source of truth
- `tests/test_mcp_server.py` — how to call tools directly in
  tests
