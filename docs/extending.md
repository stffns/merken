# Extending merken — write your own decider

Every decision primitive in merken is a **Protocol**. Extending
merken means implementing the Protocol and passing your
instance via the `Memory` constructor. You don't subclass,
you don't touch the existing deciders, you don't modify
`Memory.py`.

This doc walks through writing a custom decider for each of
the four primitives, with working code examples and test
patterns.

## The general pattern

1. **Import the Protocol and the Decision type** for the
   primitive you're extending.
2. **Write a class** with a `name` attribute and a `decide()`
   method that matches the Protocol.
3. **Construct an instance** and pass it to `Memory` via the
   matching `*_decider` kwarg.
4. **Write unit tests** using fake contexts so the decider
   can be exercised without vstash.
5. **Validate on a loop-quality scenario** before committing
   it as merken's default.

If your decider needs state (a cache, a counter, a model), the
state lives on the instance. Each `Memory` gets its own
decider instance, so state is per-Memory-lifetime.

## 1. Write a custom `should_remember` decider

### The Protocol

```python
# merken/policies/types.py
class WriteDecider(Protocol):
    name: str
    def decide(self, event: Event, ctx: WriteContext) -> Decision: ...
```

Where:

```python
@dataclass
class Event:
    text: str
    layer: str = "episodic"
    title: str | None = None
    tags: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class WriteContext:
    project: str
    # Intentionally small. See "State management" section below
    # for why there is no recall callable.

@dataclass(frozen=True)
class Decision:
    write: bool
    reason: str
    confidence: float
    policy: str
```

**Note:** an earlier version of `WriteContext` exposed a
`recall: Callable` field that was backed by `vstash.Memory.search`.
It was removed 2026-04-09 because (a) nothing used it after the
O(N²) dedup refactor, and (b) its presence contradicted the
"never call vstash from decide()" rule below. If a future
similarity-based decider genuinely needs read access to vstash at
decision time, the right move is to re-add it with an explicit
docstring warning about cost, not to leave a vague hook around
"for later."

### Example: content-type prior decider

Suppose you tag events with a `type:<kind>` tag, and you want
to accept "decision"-type events aggressively but reject
"ambient_chat" events unless they're novel.

```python
# my_deciders.py
from merken.policies.types import Decision, Event, WriteContext
from merken.policies.should_remember import HeuristicWriteDecider


class ContentTypePriorDecider:
    """Extends HeuristicWriteDecider with an A-MAC-style content
    type prior. Gates acceptance on a tag-based prior before
    falling through to the heuristic rules.

    Expected tag format: `type:<kind>`, e.g. `type:decision`.
    Events without a type tag get the default prior (1.0 — pass
    through to the heuristic rules unchanged).
    """

    name = "ContentTypePriorDecider"

    _DEFAULT_PRIORS = {
        "decision": 1.0,       # always pass through
        "fact": 0.9,
        "observation": 0.7,
        "question": 0.6,
        "ambient_chat": 0.2,
        "tool_echo": 0.1,
    }

    def __init__(self, *, priors: dict[str, float] | None = None):
        self._priors = priors or dict(self._DEFAULT_PRIORS)
        self._heuristic = HeuristicWriteDecider()

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        # Extract type tag
        content_type = self._extract_type(event.tags)
        prior = self._priors.get(content_type, 1.0)

        # Very low prior: skip without running the heuristic
        if prior < 0.25:
            return Decision(
                write=False,
                reason=f"low_prior:{content_type}:{prior:.2f}",
                confidence=1.0 - prior,
                policy=self.name,
            )

        # Otherwise delegate to the heuristic decider
        base = self._heuristic.decide(event, ctx)

        # If the heuristic decided to write, fold the prior into confidence
        if base.write:
            return Decision(
                write=True,
                reason=f"novel+prior:{content_type}:{prior:.2f}",
                confidence=base.confidence * prior,
                policy=self.name,
            )
        return base

    def _extract_type(self, tags: str | None) -> str:
        if not tags:
            return "unknown"
        for tag in tags.split(","):
            tag = tag.strip()
            if tag.startswith("type:"):
                return tag.split(":", 1)[1]
        return "unknown"
```

### Using it

```python
from merken import Memory
from my_deciders import ContentTypePriorDecider

with Memory(
    project="my_agent",
    write_decider=ContentTypePriorDecider(),
) as mem:
    # High-prior: always written
    mem.remember("We decided to use Postgres 16", tags="type:decision")

    # Low-prior: skipped silently
    mem.remember("ok thanks", tags="type:ambient_chat")

    # No type tag: falls through to heuristic (default prior = 1.0)
    mem.remember("a long enough observation about the codebase")
```

### Custom priors per project

```python
mem = Memory(
    project="medlocal",
    write_decider=ContentTypePriorDecider(priors={
        "diagnosis": 1.0,       # keep everything medical
        "demo_case": 1.0,
        "ambient_chat": 0.1,    # even more aggressive skip
        "tool_echo": 0.05,
    }),
)
```

### Testing your custom write decider

```python
# tests/test_my_deciders.py
from merken.policies.types import Event, WriteContext
from my_deciders import ContentTypePriorDecider


def _ctx() -> WriteContext:
    return WriteContext(project="unit", recall=lambda q, k, l: [])


def test_decision_tag_writes_novel() -> None:
    d = ContentTypePriorDecider()
    result = d.decide(
        Event(
            text="we chose Postgres 16 for the analytics warehouse",
            tags="type:decision",
        ),
        _ctx(),
    )
    assert result.write
    assert "novel+prior" in result.reason
    assert "decision" in result.reason


def test_ambient_tag_skipped_without_heuristic() -> None:
    d = ContentTypePriorDecider()
    result = d.decide(
        Event(
            text="ok thanks for the explanation about postgres",
            tags="type:ambient_chat",
        ),
        _ctx(),
    )
    assert not result.write
    assert "low_prior" in result.reason
    assert "ambient_chat" in result.reason


def test_no_tag_falls_through_to_heuristic() -> None:
    """An event without a type tag should hit the heuristic
    rules unchanged — no prior is applied."""
    d = ContentTypePriorDecider()
    result = d.decide(
        Event(text="a sufficiently long event with no type tag"),
        _ctx(),
    )
    assert result.write
    assert "novel" in result.reason


def test_custom_priors_override_defaults() -> None:
    d = ContentTypePriorDecider(priors={"banter": 0.05})
    result = d.decide(
        Event(text="something said in passing", tags="type:banter"),
        _ctx(),
    )
    assert not result.write
    assert "low_prior:banter" in result.reason
```

Run with:

```bash
pytest tests/test_my_deciders.py -v
```

### Validating on a loop-quality scenario

Before committing your decider as the default for merken,
verify it doesn't regress the scenarios:

```bash
# Point the runner at a specific scenario
python -m experiments.loop_quality.runner \
    --scenario experiments/loop_quality/scenarios/jay_vstash_2026_04_09_snapshot.json
```

If the scenario's pass_rate or purity drops, the decider isn't
strictly better — it's a trade-off, and the trade-off has to
be documented in the commit message. See
[`primitives.md`](primitives.md) for the rules merken's own
deciders were held to.

## 2. Custom `should_recall`

### The Protocol

```python
class RecallDecider(Protocol):
    name: str
    def decide(self, query: str, ctx: RecallContext) -> RecallPlan: ...
```

Where:

```python
@dataclass
class RecallContext:
    project: str
    top_k: int

@dataclass(frozen=True)
class LayerRequest:
    layer: str
    top_k: int

@dataclass(frozen=True)
class RecallPlan:
    layers: list[LayerRequest]
    reason: str
    policy: str
```

### Example: query-type-aware router

```python
from merken.policies.should_recall import (
    LayerRequest,
    RecallContext,
    RecallPlan,
)


class QueryTypeRouter:
    """Routes entity/specific queries to episodic first, theme
    queries to semantic first. A tiny regex-based classifier,
    no LLM.

    'What did X say about Y?' — entity query, episodic first
    'Summarize our decisions' — theme query, semantic first
    """

    name = "QueryTypeRouter"

    # Naive classifier: if the query mentions proper nouns or
    # asks about a specific time / person, it's an entity
    # query. Otherwise theme.
    _ENTITY_SIGNALS = (
        "who ", "when ", "where ", "last ", "yesterday", "earlier",
        "on 2026", "said", "decided", "meeting",
    )

    def __init__(self, *, top_k_per_layer: int = 5):
        self.top_k_per_layer = top_k_per_layer

    def decide(self, query: str, ctx: RecallContext) -> RecallPlan:
        is_entity = self._classify(query)

        if is_entity:
            order = [("episodic", self.top_k_per_layer),
                     ("semantic", self.top_k_per_layer)]
            reason = "entity_first_episodic"
        else:
            order = [("semantic", self.top_k_per_layer),
                     ("episodic", self.top_k_per_layer)]
            reason = "theme_first_semantic"

        return RecallPlan(
            layers=[LayerRequest(layer=l, top_k=k) for l, k in order],
            reason=reason,
            policy=self.name,
        )

    def _classify(self, query: str) -> bool:
        q = query.lower()
        return any(sig in q for sig in self._ENTITY_SIGNALS)
```

### Using it

```python
from merken import Memory
from my_deciders import QueryTypeRouter

with Memory(
    project="my_agent",
    recall_decider=QueryTypeRouter(top_k_per_layer=4),
) as mem:
    # Classifies as entity query → episodic first
    mem.recall("what did the user say on Monday?")

    # Classifies as theme query → semantic first
    mem.recall("summarize the architecture decisions")
```

### Testing

```python
def test_entity_query_routes_episodic_first() -> None:
    d = QueryTypeRouter()
    plan = d.decide(
        "what did we decide yesterday?",
        RecallContext(project="unit", top_k=5),
    )
    assert plan.layers[0].layer == "episodic"
    assert plan.reason == "entity_first_episodic"


def test_theme_query_routes_semantic_first() -> None:
    d = QueryTypeRouter()
    plan = d.decide(
        "summarize everything about databases",
        RecallContext(project="unit", top_k=5),
    )
    assert plan.layers[0].layer == "semantic"
    assert plan.reason == "theme_first_semantic"
```

## 3. Custom `should_consolidate`

### The Protocol

```python
class ConsolidateDecider(Protocol):
    name: str
    def decide(
        self,
        n_events: int,
        ctx: ConsolidateContext,
    ) -> ConsolidationDecision: ...
```

### Example: time-based consolidator

```python
import time
from merken.policies.should_consolidate import (
    ConsolidateContext,
    ConsolidationDecision,
)


class TimeBasedConsolidator:
    """Fire consolidation when either N events accumulate OR
    T seconds have passed since the last run."""

    name = "TimeBasedConsolidator"

    def __init__(
        self,
        *,
        min_events: int = 10,
        min_seconds_since_last: int = 3600,
    ):
        self.min_events = min_events
        self.min_seconds = min_seconds_since_last
        self._last_run_at = 0.0

    def decide(
        self,
        n_events: int,
        ctx: ConsolidateContext,
    ) -> ConsolidationDecision:
        now = time.time()
        elapsed = now - self._last_run_at

        if n_events >= self.min_events:
            self._last_run_at = now
            return ConsolidationDecision(
                proceed=True,
                reason=f"enough_events:{n_events}",
                policy=self.name,
            )

        if elapsed >= self.min_seconds and n_events > 0:
            self._last_run_at = now
            return ConsolidationDecision(
                proceed=True,
                reason=f"time_elapsed:{elapsed:.0f}s_n={n_events}",
                policy=self.name,
            )

        return ConsolidationDecision(
            proceed=False,
            reason=f"skip:n={n_events}_elapsed={elapsed:.0f}s",
            policy=self.name,
        )
```

**Gotcha:** the decider's state (`_last_run_at`) is per-instance
and reset on `Memory` construction. If you want
cross-invocation persistence (like `HeuristicWriteDecider`'s
hydration), you need to pull state from somewhere vstash-side
yourself. There's no built-in hydration hook for
`ConsolidateDecider` or `ForgetDecider` in v1 — only
`should_remember` has `set_hydrate_fn`.

## 4. Custom `should_forget`

### The Protocol

```python
class ForgetDecider(Protocol):
    name: str
    def decide(
        self,
        event_path: str,
        event_text: str,
        ctx: ForgetContext,
    ) -> ForgetDecision: ...
```

### Example: strict consolidation forget

A stacked-gate decider that composes two checks, both
implementable with what `ForgetContext` actually exposes. It
tombstones only events that have been consolidated into at
least `min_facts` **and** pass a minimum event-text length
sanity check — the idea being that very short events are
either noise or near-empty, and either way don't deserve the
same forgetting threshold as substantive events.

```python
from merken.policies.should_forget import ForgetContext, ForgetDecision


class StrictConsolidationForget:
    """Two-gate forget: consolidated AND non-trivial.

    Events that are in >= min_facts semantic facts get
    tombstoned, but only if their text is at least
    min_text_chars long. Very short events skip the forget
    pipeline on the theory that they're noise or placeholder,
    and the audit row is cheap enough to preserve.
    """

    name = "StrictConsolidationForget"

    def __init__(
        self,
        *,
        min_facts: int = 2,
        min_text_chars: int = 40,
    ):
        self.min_facts = min_facts
        self.min_text_chars = min_text_chars

    def decide(
        self,
        event_path: str,
        event_text: str,
        ctx: ForgetContext,
    ) -> ForgetDecision:
        # Gate 1: must be consolidated with at least min_facts
        n = len(ctx.derived_in_facts)
        if n < self.min_facts:
            return ForgetDecision(
                tombstone=False,
                reason=f"not_consolidated:{n}<{self.min_facts}",
                confidence=1.0,
                policy=self.name,
            )

        # Gate 2: text must be substantive
        text_len = len(event_text.strip())
        if text_len < self.min_text_chars:
            return ForgetDecision(
                tombstone=False,
                reason=f"text_too_short:{text_len}<{self.min_text_chars}",
                confidence=1.0,
                policy=self.name,
            )

        return ForgetDecision(
            tombstone=True,
            reason=f"consolidated_and_substantive:n_facts={n}_chars={text_len}",
            confidence=1.0,
            policy=self.name,
        )
```

Both gates use only what `ForgetContext` and the function args
already provide (`event_text` is passed by `Memory.forget`,
`derived_in_facts` is in the context). No `added_at`, no
`access_count`, no vstash lookups — the decider is pure.

### What if `ForgetContext` doesn't expose what I need?

This is the honest v1 limitation. `ForgetContext` currently
only has `project` and `derived_in_facts`. If you want to
decide based on the event's age, access count, source tags,
or any other metadata, you have three options, each with a
real trade-off:

1. **Contribute an extension to merken core.** Add the field
   to `ForgetContext` in a PR, with a loop-quality scenario
   that demonstrates why it's needed. This is the clean move
   but has a high bar (scenario + no regression + docs).

2. **Open the vstash directly from your decider.** Your
   decider can hold a reference to the `Memory` (or a
   closure that opens its own vstash connection) and query
   `added_at` via `vstash.Memory.list()` on each `decide()`
   call. **This violates the "no vstash from decide()" rule
   below** and will be O(N) per forget run. Don't do it
   unless you've measured the cost and accepted it.

3. **Pre-compute the info once before calling `forget()`.**
   Wrap `Memory.forget()` in your own function that first
   queries vstash for age/access metadata, builds a mapping,
   and either (a) passes it to your decider via a class attr
   or (b) uses it to filter which events your custom
   `forget()` wrapper passes to the underlying Memory
   operation.

Option 3 is the cleanest workaround because it keeps the
decider pure while still getting the extra info. Example
sketch:

```python
class AgeAwareForgetWrapper:
    def __init__(self, mem, max_age_days=30, min_facts=1):
        self.mem = mem
        self.max_age_days = max_age_days
        self.min_facts = min_facts

    def forget_old_consolidated(self):
        from datetime import datetime, timezone
        # Pre-query age info for every episodic doc
        now = datetime.now(timezone.utc)
        age_by_path = {
            d.path: (now - datetime.fromisoformat(d.added_at)).days
            for d in self.mem._vstash.list(
                collection=self.mem.collection,
                layer="episodic",
            )
        }

        class _AgedForget:
            name = "_AgedForget"
            def decide(inner_self, path, text, ctx):
                if len(ctx.derived_in_facts) < self.min_facts:
                    return ForgetDecision(False, "not_consolidated", 1.0, inner_self.name)
                age = age_by_path.get(path, 0)
                if age < self.max_age_days:
                    return ForgetDecision(False, f"too_recent:{age}d", 1.0, inner_self.name)
                return ForgetDecision(True, f"old_and_consolidated:age={age}d", 1.0, inner_self.name)

        # Swap the decider for this forget call
        self.mem._forget_decider = _AgedForget()
        return self.mem.forget()
```

This reaches into `Memory._forget_decider` which is private —
a real contribution would add a `forget(decider=...)` override
param to `Memory.forget` in merken core instead. But as a
workaround for a specific user's decider, the pattern works
and keeps the core `ForgetContext` model clean.

## 5. Plug a reranker into `recall`, or a materializer into `consolidate`

Two more hooks landed 2026-09-18. Neither is a decision primitive
(four is enough — CLAUDE.md "What's NOT next"); they let a decider
act *inside* recall and consolidation without touching `Memory`.

### `Reranker` — reorder recall candidates

```python
# merken/reranking.py
class Reranker(Protocol):
    name: str
    def rerank(self, query: str, hits: list[SearchResult]) -> list[SearchResult]: ...
```

Pass it as `Memory(reranker=my_reranker, recall_overfetch=4)`. With a
reranker set, every layer in the `RecallPlan` fetches
`top_k * recall_overfetch` candidates, the reranker sees the merged,
deduped list, and its ordering (it may also drop hits) is truncated to
the caller's `top_k`. The recall audit row records `reranker: <name>`.
Explicit `layer=` calls bypass it. Return the hits unchanged on any
internal error — never lose a retrieval result to a reranker bug.

### `materialize_fn` — decide what a semantic fact *is*

`Memory.consolidate(materialize_fn=fn)` where `fn(cluster) -> Fact` and
`cluster` is `list[(doc_path, text)]`. Unlike `synthesize_fn` (text
only), the materializer controls `derived_from` too, so it can expel
members that do not belong in the fact's provenance. Return
`merken.consolidation.materialize_fact(cluster)` as your fallback.

### Reference implementation

`merken.classifiers.jev` ships one of each on TypeSafe's Jev
(network, opt-in): `JevReranker` and `JevMaterializer`, plus a
`JevWriteDecider`. `experiments/loop_quality/jev_probe.py` is the
ablation that justified them; `experiments/loop_quality/RESULTS.md`
has the numbers.

## Per-decider state management

Different primitives have different state needs. Here's the
current state management model:

| Primitive | State lives where | Cross-instance? |
|---|---|---|
| `should_remember` | `self._seen` on the decider instance, hydrated lazily from vstash via `hydrate_fn` on first `decide()` | Yes — every new Memory rehydrates |
| `should_recall` | Stateless by default. A custom decider can hold state on the instance. | No — per-Memory-lifetime |
| `should_consolidate` | Stateless by default. | No |
| `should_forget` | Stateless by default. | No |

If you need cross-invocation state for a non-remember decider,
the options are:

1. **Store it in vstash yourself** under a dedicated collection
   (e.g. `my_decider_state`). Look it up at the start of
   `decide()`.
2. **Use the audit log** — your previous decisions are already
   persistent there. You can query them via `Memory.audit()`
   from inside your decider if you hold a reference to the
   Memory.
3. **Hold Memory via context** — extend the context dataclass
   (in your own code, not in merken core) to include a
   callable that opens the parent Memory, and use it to query.

For v1 all four primitives in merken core are either stateless
or use `should_remember`'s `hydrate_fn` pattern. Custom
extensions are welcome to do more.

## Testing your custom deciders

The test patterns are:

### Pure unit tests (no vstash)

Fake the context object (pass a lambda for `recall`, an empty
list for `derived_in_facts`, etc.) and assert on the returned
decision. Fastest feedback loop.

### Integration tests (with vstash, tmp_path DB)

Construct `Memory` with your decider, call the real methods,
and assert on behavior. Slower but catches wiring bugs.

```python
from pathlib import Path
from merken import Memory
from my_deciders import ContentTypePriorDecider


def test_content_type_prior_integrated(tmp_path: Path) -> None:
    with Memory(
        project="integration_test",
        db=tmp_path / "e.db",
        write_decider=ContentTypePriorDecider(),
    ) as mem:
        r1 = mem.remember(
            "we decided postgres 16 for the analytics warehouse",
            tags="type:decision",
        )
        r2 = mem.remember(
            "ok thanks for the explanation about postgres",
            tags="type:ambient_chat",
        )

    assert r1.written
    assert not r2.written
    assert "low_prior" in r2.decision.reason
```

### Loop-quality scenario runs

The strictest bar. Run your decider against a real-content
scenario and compare pass_rate / purity / coverage to the
baseline (merken's default decider).

**Programmatic API** (since 2026-04-09) — `run_scenario` now
accepts decider overrides:

```python
from pathlib import Path
from merken import NeverConsolidate
from experiments.loop_quality.runner import run_scenario
from experiments.loop_quality.scenario import load_scenario
from my_package import MyCustomConsolidator

scenario = load_scenario(
    Path("experiments/loop_quality/scenarios/jay_vstash_2026_04_09_snapshot.json")
)

# Baseline: merken defaults
baseline = run_scenario(scenario, db=Path("/tmp/baseline.db"))

# With your custom consolidator
mine = run_scenario(
    scenario,
    db=Path("/tmp/mine.db"),
    consolidate_decider=MyCustomConsolidator(),
)

print(f"baseline pass={baseline.query_pass_rate:.2%} purity={baseline.cluster_purity:.2%}")
print(f"mine     pass={mine.query_pass_rate:.2%} purity={mine.cluster_purity:.2%}")
```

**CLI** (same date) — any of the four deciders is overridable
via a dotted import path, default-constructed:

```bash
# Baseline — default deciders, every scenario in fixtures/
python -m experiments.loop_quality.runner

# With your custom decider on all scenarios
python -m experiments.loop_quality.runner \
    --consolidate-decider my_package.MyCustomConsolidator

# Multiple overrides
python -m experiments.loop_quality.runner \
    --write-decider my_package.MyWriteDecider \
    --consolidate-decider my_package.MyConsolidator \
    --embedding-linkage average

# One scenario only
python -m experiments.loop_quality.runner \
    --scenario experiments/loop_quality/scenarios/jay_vstash_2026_04_09_snapshot.json \
    --consolidate-decider my_package.MyConsolidator
```

The dotted path must resolve to a **default-constructible**
class — no required constructor args. For parameterized
deciders, subclass and hard-code the params:

```python
# my_package.py
from merken import PeriodicConsolidator

class AggressiveConsolidator(PeriodicConsolidator):
    def __init__(self):
        super().__init__(min_events=2)
```

Then `--consolidate-decider my_package.AggressiveConsolidator`.

This closes the "validate before landing" loop — you can run
your decider against every scenario in the safety net with one
command, compare deltas against the baseline, and only open a
PR when the data supports it.

## Design principles for custom deciders

Adapted from the patterns the four default deciders follow:

### Keep decide() pure

`decide()` should be a function of `(input, ctx)` with minimal
side effects. If you need to update state (like
`HeuristicWriteDecider` updating `_seen`), do it explicitly
and document it in the docstring.

### Never call vstash from decide() in the hot path

Decision primitives should not trigger vstash searches per
`decide()` call. If you need similarity information,
pre-compute it (via `hydrate_fn` for writes, or via a pre-pass
that populates instance state for other primitives) or accept
the limitation. A decider that is O(1) in the hot path is
almost always better than one that is O(log N) by consulting
vstash.

The reason is cost-at-scale. An merken `Memory.remember` call
runs the write decider once. A `forget` call runs the forget
decider N times (once per episodic event). If `decide()` calls
`vstash.search` internally, the forget run becomes O(N log N)
or worse. On a 10k-event store that's the difference between
milliseconds and tens of seconds per forget pass.

**Historical note:** an earlier version of `WriteContext`
exposed a `recall: Callable` field. It was removed 2026-04-09
both because nothing used it after the dedup refactor and
because its presence made this rule ambiguous. See the
"Note" near the top of this doc. If a similarity-based decider
is the right move for your use case, add the callable back to
the relevant context with a clear docstring about cost — don't
sneak it in behind a generic name.

### Reason strings should be grep-able

Engram's audit log is queryable via vstash hybrid search. A
good `reason` string is:

- **Machine-parseable** — e.g. `too_short:<8`, not "text is
  too short"
- **Uniquely grep-able** — e.g. `low_prior:ambient_chat:0.20`,
  not just "skipped"
- **Informative without the source** — a reader of
  `merken audit low_prior` should understand what happened
  without reading the decider code

### Fail open, unless it's a safety-critical write

`HeuristicWriteDecider` swallows hydration failures — if vstash
is temporarily unreachable, dedup degrades but writes keep
happening. This is fail-open and it's correct for decision
primitives.

The one exception is `Memory.forget()` — it raises if the
tombstone write fails, because we must not remove the original
without a backup. That's fail-closed and also correct.

For your custom decider, default to fail-open. Make
exceptions explicit.

### Version your `name`

If you iterate on a decider's logic, append a version
suffix to the `name` attribute:

```python
class ContentTypePriorDecider:
    name = "ContentTypePriorDecider_v2"
```

The audit log preserves the `policy` field — in a year's time,
you'll be able to see which version of your decider made which
decisions and correlate with behavior changes.

## Extending merken core vs extending in your own code

**Extend in your own code** if:

- The custom decider is specific to your agent / project
- It depends on state or context outside of merken's general
  model
- You want to experiment before upstreaming

**Contribute to merken core** if:

- The decider represents a general-interest improvement (e.g.
  ContentTypePrior, TemporalRecaller, EbbinghausDecayForget
  — all discussed in [`../notes/research-2026-04-09.md`](../notes/research-2026-04-09.md))
- It comes with a loop-quality scenario that demonstrates the
  value
- It doesn't regress any existing scenario
- The `decide()` function is pure and testable

**How to check "doesn't regress any existing scenario":**

```bash
# Record the baseline (merken defaults)
python -m experiments.loop_quality.runner > /tmp/baseline.txt

# Run your decider
python -m experiments.loop_quality.runner \
    --consolidate-decider my_package.MyConsolidator \
    > /tmp/mine.txt

# Compare
diff /tmp/baseline.txt /tmp/mine.txt
```

If `diff` shows that any scenario's `query_pass_rate` or
`cluster_purity` dropped, your decider is a trade-off not a
strict improvement. That doesn't disqualify it — some
trade-offs are worth making — but the PR description has to
name the trade-off, and ideally add a new scenario that
makes the trade-off visible (the failing case on the default
decider and the passing case on yours).

The general-interest bar is high on purpose — merken aims for
a small, stable default set of deciders, with extensions
living in user code until empirically justified. But the bar
is now achievable with one command, not "requires modifying
the runner" as an earlier version of this doc said.

## Marking authoritative content -- Type A vs Type B memory

merken stores two ontologies in the same vstash:

- **Type A (derived):** episodic events, briefs, traces. Mutable;
  consolidate / forget / future decay can transform or remove them.
- **Type B (authoritative):** clinical protocols, specs, laws, fixed
  corpora. Immutable; mutative ops MUST skip them. Updated only via
  versioned replacement.

The distinction is a tag convention, not a separate layer or
collection (real decisions integrate both).

### Marking events as Type B

CLI:

```bash
merken --project medlocal remember "WHO clause: amoxicillin 50 mg/kg/day for pneumonia" \
    --title proto_who_amox_pneumonia \
    --tags "topic:pneumonia,protocol:who" \
    --immutable
```

The `--immutable` flag adds `source:authoritative` to the tags. If
your `--tags` already include a `source:<value>` your value wins
(no double-tagging).

Python SDK:

```python
mem.remember(
    "WHO clause...",
    title="proto_who_amox_pneumonia",
    tags="source:authoritative,topic:pneumonia,protocol:who",
)
```

### Fail-closed semantics

The predicate `merken.sourcing.is_safely_mutable(tags)` decides
whether `consolidate()` and `forget()` may touch an event:

| tags                                  | mutable? | reason |
|---------------------------------------|---------|--------|
| `None` / empty                        | yes | legacy untagged |
| `source:session` (KNOWN_DERIVED)      | yes | explicitly derived |
| `source:authoritative`                | no  | explicitly authoritative |
| `source:autoritative` (typo)          | no  | unknown source -> fail closed |
| `source:foo_unknown`                  | no  | unknown source -> fail closed |
| no `source:` tag at all               | yes | legacy default |

The asymmetry is intentional: skipping a mutable-but-misspelled
event is recoverable (re-tag, re-run); mutating a Type B event is
not (lost protocol). When in doubt: do not mutate.

### Adding new derived sources

If a new ingest path produces events that should be Type A, register
its source value in `merken.sourcing.KNOWN_DERIVED`. Otherwise the
filter will treat events from that path as immutable and they'll
never get consolidated.

### Citation preference (separate from mutability)

`is_authoritative(tags)` returns True only for explicitly
authoritative sources. Useful when ranking search results: a
matched protocol clause should typically outrank a derived note,
even if both are equally relevant by embedding distance. Note this
is *distinct* from `is_safely_mutable`: an unknown source value is
NOT mutable AND NOT authoritative -- the unknown case is
conservative for mutation but does not get citation preference.

## Further reading

- [`primitives.md`](primitives.md) — the four default deciders
  in depth, with audit formats and trade-offs
- [`architecture.md`](architecture.md) — the memory model your
  custom decider will run against
- [`../notes/research-2026-04-09.md`](../notes/research-2026-04-09.md)
  — 6 papers with concrete extension ideas (ContentTypePrior
  from A-MAC, TemporalRecaller from CMA, EbbinghausDecay from
  SuperLocalMemory)
- `merken/policies/*.py` — the Protocol definitions and
  reference implementations
- `tests/test_should_*.py` — reference test patterns
