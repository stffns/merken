# merken architecture

This document describes merken's memory model, the decision loop,
and how the code is organized. It is the "mental model" doc —
read it once before diving into
[`primitives.md`](primitives.md), [`cli.md`](cli.md), or
[`mcp-server.md`](mcp-server.md).

## The memory model

merken organizes memory into four **collections**, each with a
specific role. All four live in the same SQLite database via
vstash — merken does not create its own storage.

```
┌──────────────────────────────────────────────────────────┐
│ vstash SQLite (one DB per merken project)                 │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ default      │  │ merken_audit │  │ merken_        │  │
│  │ (user data)  │  │ (decisions)  │  │  tombstones    │  │
│  │              │  │              │  │ (forgotten)    │  │
│  │  layer:      │  │  layer:      │  │                │  │
│  │   episodic   │  │   audit      │  │  layer:        │  │
│  │   semantic   │  │              │  │   tombstone    │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### `default` collection — your data

This is where actual memory lives. Events land here when
`should_remember` says yes. Inside this collection, vstash
`layer` tags distinguish two kinds of memory:

- **`layer="episodic"`** — raw events the agent saw. One document
  per event. High volume, low information density. Think "what
  the user said on 2026-04-08 at 11:04".
- **`layer="semantic"`** — consolidated facts derived from one
  or more episodic events. Lower volume, higher density. Think
  "the user prefers Postgres as of 2026-04-08". Every semantic
  fact has `derived_from:<path1>,<path2>` in its vstash `tags`,
  so provenance is queryable.

merken does NOT currently use `layer="procedural"`. CONSTITUTION §5
describes procedural memory as a potential fourth memory layer
for captured-on-success task recipes ("how the agent solved X
last time"). It is not in v1, and **it is not on the near-term
roadmap** — there is no scheduled slice that implements it. The
layer is mentioned here only so a reader who reads CONSTITUTION
§5 knows why it's not represented in any diagram. Treat the
two-layer model (episodic + semantic) as merken's full memory
shape until a scenario demonstrably requires a third layer.

### `merken_audit` collection — every decision

When a decider runs (`should_remember`, `should_recall`,
`should_consolidate`, `should_forget`), merken writes one row here
with:

- The decision (`write: True/False` for remember; similar for
  others)
- The reason (e.g. `novel`, `dup_exact`, `too_short:<8`,
  `empty`, `never_auto`, `consolidated_in_2_facts`)
- The policy that fired (e.g. `HeuristicWriteDecider`)
- A preview of the event text (for should_remember)
- The layers/top_k chosen (for should_recall)

Layer is always `"audit"`. Collection is always
`merken_audit`, isolated from `default` so audit rows never
leak into normal recall.

You query it via `mem.audit(query, top_k)` or `merken audit`.

### `merken_tombstones` collection — what was forgotten

When `should_forget` decides to tombstone an event, merken:

1. Writes a row here with the **full text** of the original
   event plus metadata (title, layer, tags, derived_in_facts,
   reason, policy, timestamp)
2. Then calls `vstash.remove()` on the original path in the
   `default` collection

The tombstone **is** the forgetting record. It's not a soft
delete flag on the original row (vstash doesn't support that),
it's a copy in a separate collection that the user can query
and, in principle, restore from.

Layer is `"tombstone"`. Collection is `merken_tombstones`.
Query via `mem.tombstones(query)` or `merken tombstones`.

## The decision loop

Every `Memory` operation runs through a decision primitive that
writes an audit row, even if the decision is "do nothing."

```
                 ┌──────────────────────────────┐
   user ─────►   │  Memory.remember(text)       │
                 └────────┬─────────────────────┘
                          │
                          ▼
                 ┌──────────────────────────────┐
                 │  WriteDecider.decide(event)  │ ── audit row
                 └────────┬─────────────────────┘
                          │
              ┌───────────┴────────────┐
              │                        │
         write=False              write=True
              │                        │
              ▼                        ▼
    return RememberResult      vstash.remember(text, layer, ...)
    (written=False)                    │
                                       ▼
                              ┌──────────────────────────┐
                              │  Was vstash status=ok?   │
                              └────────┬─────────────────┘
                                       │
                           ┌───────────┴────────────┐
                           │                        │
                         no                        yes
                           │                        │
                           ▼                        ▼
              override decision =        return RememberResult
              vstash_rejected:<status>   (written=True)
              audit row, return
              written=False
```

Two non-obvious things in this flow:

1. **vstash has its own guardrails.** Specifically, it rejects
   text shorter than ~20 chars with `status="empty"`, no
   exception. merken surfaces this as a second decision
   (`vstash_rejected:empty`) and writes a corrective audit row
   so `RememberResult.written` never lies.
2. **The write decider hydrates lazily from vstash.** The first
   `decide()` call after `Memory` construction pulls every
   existing episodic doc's text into the decider's in-process
   dedup set. This is how cross-invocation dedup works for the
   CLI and MCP server — every time you run `merken remember`,
   the fresh decider sees everything already in the store.

   **Scaling caveat.** Current hydration is O(N) memory and
   O(N) SQLite round-trips per `Memory` construction: one
   `list()` + N `get_document_chunks()` calls, then every text
   joined into a `set[str]`. At ~10k episodic events this costs
   roughly 5 MB of in-process strings and 15-25 seconds of
   startup latency per CLI invocation; at 100k it becomes
   minutes of startup and tens of MB of memory. **The current
   implementation is expected to break usefully around
   10k–50k events**, depending on hardware.

   This is a known gap, not a hidden footgun. Two concrete
   fixes, neither implemented in v1:

   - **Hash-only hydration** — store SHA-1 of normalized text
     instead of the text itself (~8 bytes per entry, 60× memory
     reduction, same startup cost). ~20 lines of code; backward
     compatible with `HeuristicWriteDecider` semantics.
   - **Persistent dedup sidecar** — move `_seen` into a dedicated
     `merken_dedup` collection with one row per hash. One query
     at startup, incremental writes on each `remember`. Scales
     by design. ~50 lines of code.

   Neither fix ships until a loop-quality scenario exercises a
   large store. Building for a problem that no scenario
   demonstrates would be speculative work, and we'd have no way
   to validate the fix doesn't regress. Track under
   `loop_quality/scenarios/` — a "10k events" scenario is the
   gating requirement.

Similar flow diagrams exist for `recall`, `consolidate`, and
`forget`. All four end in an audit row.

### `recall` — layered interleave

```
   ┌────────────────────────────┐
   │  Memory.recall(query)      │
   └──────────┬─────────────────┘
              │
              │ (if layer is None)
              ▼
   ┌────────────────────────────┐
   │  RecallDecider.decide()    │ ── audit row
   │  returns RecallPlan:       │
   │    [sem:top_k=5,           │
   │     epi:top_k=3]           │
   └──────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │  Fetch ALL layers first    │ ← sequential drain was a bug,
   │  (not sequential drain)    │   see notes/silt.md
   └──────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │  Round-robin interleave    │
   │  dedup by path             │
   │  truncate to user's top_k  │
   └──────────┬─────────────────┘
              │
              ▼
          SearchResult[]
```

If the caller passes an explicit `layer=`, the decider is
bypassed and the call is a direct `vstash.search` with that
layer. This is the escape hatch for benchmarks and for callers
who already know exactly which layer they want.

### `consolidate` — episodic → semantic via clustering

```
   ┌────────────────────────────┐
   │  Memory.consolidate()      │
   └──────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │  List episodic docs        │
   │  Reassemble text from      │
   │  chunks                    │
   └──────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │  ConsolidateDecider.decide │ ── audit row
   │  (n_events, ctx)           │
   └──────────┬─────────────────┘
              │
      ┌───────┴────────┐
      │                │
   proceed=False   proceed=True  (or force=True)
      │                │
      ▼                ▼
   skip         cluster_by_embedding(events, embed_fn, threshold,
                                     linkage="complete")
                       │
                       ▼
              For each cluster ≥ min_cluster:
                  materialize_fact(cluster)
                  fact_path = f"text://fact_{fact_fingerprint(fact)}"
                  vstash.remember(
                      fact.text,
                      title=f"fact_{fp}",
                      layer="semantic",
                      tags=f"derived_from:{','.join(fact.derived_from)}",
                  )
                       │
                       ▼
              return ConsolidationResult
```

The clustering uses vstash's embedder (resolved from vstash's
`store_meta.embedding_model` at runtime, falling back to config
default) with **complete linkage at cosine threshold 0.70**.
Both knobs were picked via grid search on three loop-quality
scenarios — see
[`experiments/loop_quality/RESULTS.md`](../experiments/loop_quality/RESULTS.md).

**Why complete linkage and not average:** a 2026-04-09 grid
compared `complete` and `average` linkage across the three
scenarios at thresholds 0.65 – 0.82. Headline findings:

- **Complete @ 0.70 is the unique Pareto point** — the lowest
  threshold where all three scenarios hit 100% pass rate AND
  100% purity simultaneously, with the highest mean
  topic_coverage (71%).
- **Average linkage matches complete at threshold 0.72+** and
  loses to complete at 0.70 specifically on
  `jay_vstash_2026_04_09_snapshot`: average pulls the
  `agent-memory-use-cases` event into the `vstash_notes`
  cluster because the mean cross-pair stays above 0.70 even
  though one cross-pair is below. Complete rejects that outlier
  and keeps the cluster pure.
- **Neither linkage strictly dominates the other.** They are
  equivalent at 0.72+; complete wins at 0.70 because the
  outlier-rejection behavior matters at the lowest viable
  threshold.
- **Average linkage is a real option.** It is implemented,
  tested, and passable via `Memory.consolidate(
  embedding_linkage="average")`. A user whose content
  distribution differs from the three scenarios (e.g. denser
  clusters with deliberate inclusion of borderline members)
  should try average — it may Pareto-dominate complete on
  different content. The default is complete because **these
  specific scenarios** prefer it.

The trade-off in one line: **complete is conservative about
the identity of a cluster, average is generous to cluster
growth**. Neither is objectively correct; the grid picked the
one that Pareto-wins on merken's own design bar.

Fact identity is a stable SHA-1 of the sorted `derived_from`
list, so re-running consolidation on the same episodic set is
idempotent: the fact gets the same path and vstash overwrites
it rather than duplicating.

### `forget` — tombstone, not delete

```
   ┌────────────────────────────┐
   │  Memory.forget()           │
   └──────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │  Build derived_in map:     │
   │    event_path → [fact      │
   │      paths citing it]      │
   │  (from semantic layer's    │
   │   `tags` field)            │
   └──────────┬─────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │  For each episodic event:  │
   │   ForgetDecider.decide()   │ ── audit row
   └──────────┬─────────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
  skip              tombstone
    │                    │
    │                    ▼
    │            ┌──────────────────────┐
    │            │ Write tombstone      │
    │            │ row to merken_       │
    │            │ tombstones with      │
    │            │ full text            │
    │            └──────┬───────────────┘
    │                   │
    │                   ▼
    │            ┌──────────────────────┐
    │            │ vstash.remove(path)  │
    │            └──────────────────────┘
    │                   │
    └───────────────────┴─► ForgetResult
```

The tombstone is written **before** the removal. If the
tombstone write fails, we raise — we must not remove the
original without a backup. This is the one fail-closed path in
the loop; every other audit write is fail-open.

## Code map

```
merken/
├── __init__.py            ← public surface. Exports Memory +
│                            every decider + Fact/RememberResult/etc.
├── memory.py              ← Memory class. Wires vstash + deciders,
│                            implements remember/recall/consolidate/
│                            forget/audit/tombstones. Resolves the
│                            embed model at runtime.
├── consolidation.py       ← Fact dataclass, cluster_by_{jaccard,
│                            recall, embedding}, materialize_fact,
│                            fact_fingerprint. Pure functions —
│                            Memory is the only thing that calls
│                            them.
├── audit.py               ← format_{audit,recall,consolidate,
│                            forget}_row helpers. Collection/layer
│                            constants.
├── cli.py                 ← argparse CLI. Eight subcommands, each
│                            a thin wrapper around a Memory method.
│                            No business logic.
├── mcp_server.py          ← MCPServer (mcp 2.x). Same eight
│                            primitives as the CLI, exposed as MCP
│                            tools.
└── policies/
    ├── __init__.py        ← re-exports every public policy
    ├── types.py           ← Event, Decision, WriteContext,
    │                        WriteDecider Protocol
    ├── should_remember.py ← AlwaysWrite, HeuristicWriteDecider
    ├── should_recall.py   ← SemanticOnlyRecaller, LayeredRecaller,
    │                        RecallPlan, LayerRequest, RecallContext
    ├── should_consolidate.py  ← NeverConsolidate, PeriodicConsolidator
    │                            ConsolidationDecision, ConsolidateContext
    └── should_forget.py   ← NeverForget, ForgetConsolidated,
                              ForgetContext, ForgetDecision
```

**Invariants maintained by the code map:**

1. `memory.py` is the only module that talks to vstash. Every
   decision primitive and every helper function takes pure
   data in and returns pure data out — they never open a
   vstash connection.
2. `policies/` is the only place a user subclasses or extends.
   Writing a custom decider never requires touching `memory.py`.
3. `cli.py` and `mcp_server.py` are thin wrappers. If either
   needs behavior that isn't in `memory.py`, the method goes
   in `memory.py` first.

## Interaction with vstash

merken is a **strict consumer** of vstash's public API:

- `vstash.Memory.remember(text, title, collection, layer, tags)`
  — every write goes through here.
- `vstash.Memory.search(query, top_k, collection, layer)` — every
  read (except direct list/get_chunks) goes through here.
- `vstash.Memory.list(collection, layer)` — used by consolidate
  and forget to enumerate episodic/semantic docs.
- `vstash.Memory.get_document_chunks(path, collection)` — used
  by consolidate to re-assemble event text from stored chunks.
- `vstash.Memory.remove(path)` — used by forget to remove the
  original event after writing its tombstone.
- `vstash.Memory.stats()` — passed through by the `stats` CLI
  command.
- `vstash.embed.embed_texts(texts, model_name, backend)` — used
  by `cluster_by_embedding` to compute raw cosine for
  consolidation clustering.

merken **does not** touch:

- `vstash._store` internals, except to read `_store.db_path`
  for the embed-model resolver
- SQLite tables directly (except in probe scripts under
  `notes/`)
- `vstash.Memory.ask` (the RAG-style chat method)
- `vstash` journal API (`journal_log`, `journal_save`, etc.)

When a vstash feature is missing or surprising, the
resolution is an upstream issue, not a workaround. Filed
issues so far: #165 (closed), #172, #173.

## What this architecture is NOT

- **Not a graph database.** There is no edge table, no
  Neo4j-style traversal, no "tunnels between rooms" like
  mempalace. Engram represents relationships as `derived_from`
  tags on semantic facts and nothing else. If a knowledge
  graph is needed, it goes in a separate layer that also wraps
  vstash, not inside merken.
- **Not a multi-agent system.** One merken Memory = one agent.
  Multiple agents mean multiple DBs. Shared memory between
  agents is a separate problem not addressed here.
- **Not a framework.** There is no plugin registry, no DSL, no
  YAML config. Deciders are plain Python objects that match a
  Protocol. Configure via constructor kwargs.
- **Not RAG-as-a-service.** Recall returns `SearchResult`
  objects from vstash; it does not call an LLM to generate
  answers. What you do with the hits is your problem.

See [`CONSTITUTION.md`](../CONSTITUTION.md) §3 for the full
non-goals list.

## Further reading

- [`primitives.md`](primitives.md) — each decision primitive in depth
- [`cli.md`](cli.md) — CLI command reference
- [`mcp-server.md`](mcp-server.md) — MCP tools + Claude Code setup
- [`extending.md`](extending.md) — write your own decider
- [`../CONSTITUTION.md`](../CONSTITUTION.md) — principles
- [`../experiments/BENCHMARK_STRATEGY.md`](../experiments/BENCHMARK_STRATEGY.md) — measurement doctrine
- [`../notes/silt.md`](../notes/silt.md) — design-rule provenance
