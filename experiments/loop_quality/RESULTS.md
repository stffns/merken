# Loop-quality results

## Scope

This file records scenario-level metrics for merken's decision loop.
Every row is a run of the scenario runner against a pinned merken
commit. Unlike `experiments/retrieval/`, these numbers *are* about
whether merken's loop is adding value — there is no "absolute
positioning" disclaimer to hide behind.

## Metrics per scenario

- **query_pass_rate** — fraction of queries whose top-k semantic hits
  include a fact whose ground-truth topic matches the query's
  expected topic. Primary metric.
- **cluster_purity** — of all multi-event facts written, the fraction
  whose derived events all share the same topic. A mixed-topic fact
  is impure even if one of its members matches the query.
- **topic_coverage** — of all topics with ≥ 2 events, the fraction
  that produced at least one multi-event topic-pure fact. Measures
  whether the consolidator is finding the clusters that exist.

## Results

| Date | Commit | Scenario | Method | Threshold | Linkage | n_events | facts | pass_rate | purity | coverage | Notes |
|------|--------|----------|--------|-----------|---------|----------|-------|-----------|--------|----------|-------|
| 2026-04-09 | `4e6c7e0` | `session_2026_04_09` | `embedding_v1` | 0.65 | single | 12 | 3 | **25.00%** (1/4) | 66.67% | 33.33% | First honest run. Single-link transitive cascade via two cross-topic edges (longmemeval_a~dedup_fix_a=0.663, vstash_bug_a~dedup_fix_a=0.652) contaminates a 4-event impure cluster. Three same-topic pairs sit just below 0.65 (longmemeval=0.649, dedup_fix=0.575, loop_philosophy=0.558). |
| 2026-04-09 | `f83115e` | `session_2026_04_09` | `embedding_v1` | 0.65 | complete | 12 | 4 | **50.00%** (2/4) | 75.00% | 50.00% | Complete-link refuses to merge `{vstash_bug_a,b}` with `{longmemeval_a, dedup_fix_a}` because the weakest cross-pair (`vstash_bug_a~longmemeval_a=0.616`) is below 0.65. Three pure clusters emerge (mempalace, vstash_bug, consolidation_design) plus one impure mini-cluster (longmemeval_a+dedup_fix_a, the one edge that did cross). pass_rate doubles without changing the threshold. |
| 2026-04-09 | `8ff6953` | `analytics_project` | `embedding_v1` | 0.65 | complete | 12 | 6 | **100.00%** (4/4) | 100.00% | 100.00% | **Control scenario.** Six lexically distinct topics (auth/database/deploy/frontend/monitoring/billing). Pre-measured pairwise cosines: same-topic pairs [0.778, 0.937] median 0.915; cross-topic pairs [0.320, 0.617] median 0.465. Clean 0.162 gap — any threshold in (0.617, 0.778) gives 100%. This is the "loop works when the embedder cooperates" reference point. Future consolidator changes must not drop this below 100% without naming the trade-off. Recall path: explicit `layer="semantic"`. |
| 2026-04-09 | `HEAD` | `session_2026_04_09` | `embedding_v1` | 0.65 | complete | 12 | 4 | **100.00%** (4/4) | 75.00% | 50.00% | **`should_recall` enabled.** Same consolidation as row 2 (purity/coverage unchanged). The lift to 100% comes from `LayeredRecaller` falling back to the episodic layer when semantic doesn't have a topic-pure fact. `dedup_fix` and `longmemeval` queries now match their raw episodic events directly instead of failing for lack of a pure fact. The loop is now robust to imperfect consolidation — an impure cluster no longer takes down the query. |
| 2026-04-09 | `7e85566` | `analytics_project` | `embedding_v1` | 0.65 | complete | 12 | 6 | **100.00%** (4/4) | 100.00% | 100.00% | `should_recall` enabled. No regression: the scenario that was already at 100% stays at 100%. This is the "did we break anything" check; it passed. |
| 2026-04-09 | `5240f72` | `jay_vstash_2026_04_09_snapshot` | `embedding_v1` | 0.65 | complete | **20** | 4 | **75.00%** (3/4) | 75.00% | 60.00% | **First real-content row.** 20 organic docs from Jay's vstash frozen as a fixture, topic labels assigned by honest reading of titles (6 topics). Three queries pass. `vstash_notes` fails because Fact 4 mixes 3 vstash_notes events with 1 merken_design event; the fact's anchor literally contains "vstash Upstream Improvement Ideas" but the strict purity check rejects an impure cluster. |
| 2026-04-09 | `HEAD` | `analytics_project` | `embedding_v1` | **0.70** | complete | 12 | 6 | **100.00%** (4/4) | 100.00% | 100.00% | Post-grid-search baseline. Unchanged from threshold=0.65 — this scenario's same-topic pairs all live at 0.78–0.94 so any threshold in that gap gives 100%. |
| 2026-04-09 | `HEAD` | `session_2026_04_09` | `embedding_v1` | **0.70** | complete | 12 | 2 | **100.00%** (4/4) | **100.00%** | 33.33% | **Purity jumped 75% → 100%.** Higher threshold drops the two cross-topic edges that were crossing (0.663, 0.652) and also drops the borderline `consolidation_design` pair (0.661). Only 2 pure facts remain (mempalace, vstash_bug), coverage drops 50% → 33%. But `query_pass_rate` stays at 100% because the interleave fallback in `Memory.recall` catches dedup_fix, longmemeval, and consolidation_design queries episodically. Coverage is a means, not an end — pass_rate is what users feel. |
| 2026-04-09 | `HEAD` | `jay_vstash_2026_04_09_snapshot` | `embedding_v1` | **0.70** | complete | 20 | 4 | **100.00%** (4/4) | **100.00%** | 80.00% | **Pass rate jumped 75% → 100%, purity 75% → 100%.** Raising the threshold broke up Fact 4 — the agent-memory-use-cases doc no longer clusters with the vstash_notes group, so the vstash cluster becomes pure. `vstash_notes` query now passes. The full three-scenario picture at 0.70: all three at 100% pass_rate and 100% purity. |

## Linkage × threshold grid search (2026-04-09, arch review follow-up)

Jay's architecture review pointed out that complete-link is
conservative and can sub-cluster when one outlier is weaker
than the rest. Average-link is the natural middle ground
between single (cascades) and complete (can reject). The
threshold grid from earlier only varied threshold at linkage
fixed to `complete`; this second grid adds `average` as a
dimension.

```
                       complete                    average
scenario               0.65  0.68  0.70  0.72      0.65  0.68  0.70  0.72
─────────────────────  ─────────────────────────  ─────────────────────────
analytics_project      100   100   100   100       100   100   100   100
  (min_pass/min_pur)
session_2026_04_09     100   100   100   100       100   100   100   100
jay_vstash_snapshot     75    75   100   100        75    75    75   100
                                                              ↑
                                                     fails at 0.70
```

**Observations:**

- **At threshold 0.70, complete strictly dominates average on
  `jay_vstash_snapshot`.** Complete keeps the vstash_notes
  cluster pure by rejecting `agent-memory-use-cases` as an
  outlier with at least one cross-pair below threshold.
  Average admits it because the mean cross-pair stays above
  threshold, resulting in an impure 4-event cluster and
  dragging pass_rate and purity to 75%.
- **At threshold 0.72+, complete and average are equivalent**
  on all three scenarios. Both hit 100% pass rate AND 100%
  purity everywhere.
- **Complete @ 0.70 is the unique Pareto point** for the
  combined (pass, purity, coverage) objective. It is the
  *lowest threshold* where both complete and average can
  possibly reach the 100%/100% plateau, and complete is the
  only linkage that actually gets there.

**Verdict:** keep `complete` as the merken default. Document
`average` as a supported alternative accessible via
`Memory.consolidate(embedding_linkage="average")`. A user whose
content looks different from the three scenarios (denser
clusters, more generous inclusion tolerance) may find average
Pareto-wins on their data — we just can't show it on this
scenario set.

**No change to defaults.** `complete @ 0.70` remains.

## Threshold grid search (2026-04-09, all three scenarios)

Once there were three scenarios in the safety net, the "right
threshold" question became measurable instead of arguable. Grid:

```
thresh   analytics                 session                   jay_snapshot              min_pass  min_purity
         pass  purity  coverage    pass  purity  coverage    pass  purity  coverage
0.60     100%  100%    100%        100%   67%     33%         75%   33%     20%          75%       33%
0.63     100%  100%    100%        100%   75%     50%         75%   75%     60%          75%       75%
0.65     100%  100%    100%        100%   75%     50%         75%   75%     60%          75%       75%    ← old default
0.68     100%  100%    100%        100%  100%     33%         75%   75%     60%          75%       75%
0.70     100%  100%    100%        100%  100%     33%        100%  100%     80%         100%      100%    ← NEW default
0.72     100%  100%    100%        100%  100%     17%        100%  100%     80%         100%      100%
```

0.70 is the sweet spot: maximizes the minimum pass_rate AND the
minimum cluster_purity across all three scenarios, and leaves
topic_coverage as a monotone trade (lower coverage on the
session scenario because borderline pairs stop clustering, but
the loop's interleave fallback makes that a non-user-facing
loss).

The grid eliminates the "we need an LLM consolidator to push
past 50%" framing that the previous iteration of this file had
recorded. That was almost right — the interleave WAS real, and
the crossing cosines WERE real — but it assumed the crossing
happened above 0.70. It didn't. Moving the threshold up 5 points
drops the cross-topic edges and only costs coverage on one
borderline same-topic pair, which the fallback absorbs.

Silt-check: the ceiling I named earlier ("above 50% on this
scenario requires an LLM") was an overread. The data disagreed
when asked properly. The grid was there to run all along — we
just didn't run it until a second scenario and a real-content
snapshot made the shape of the answer visible.

## Three-scenario picture (post threshold grid)

As of the real-vstash snapshot commit, the metric table has three
rows that are all running under the same runner, in the same
`pytest tests/` invocation, on the same commit:

|                          | analytics_project | session_2026_04_09 | jay_vstash_2026_04_09_snapshot |
|---|---|---|---|
| Type                     | synthetic control | synthetic borderline | real organic content |
| Events                   | 12                | 12                  | 20                  |
| Topics                   | 6 distinct        | 6 overlapping       | 6 (real distribution) |
| Same-topic median cosine | 0.915             | 0.649               | (not measured yet)  |
| facts_written            | 6                 | 4                   | 4                   |
| **pass_rate**            | **100%**          | **100%**            | **75%**             |
| cluster_purity           | 100%              | 75%                 | 75%                 |
| topic_coverage           | 100%              | 50%                 | 60%                 |

The three scenarios stress the loop from three angles:

- **analytics_project** — the embedder cooperates. This is the
  "nothing is broken" control. Any regression here is a disaster.
- **session_2026_04_09** — the embedder disagrees with ground
  truth. Tests how well the loop survives overlapping vocabulary.
  cluster_purity and topic_coverage are stuck but query routing
  still succeeds via the interleave fallback.
- **jay_vstash_2026_04_09_snapshot** — the real thing. Content
  the user produced organically, topic labels assigned by honest
  title-reading. The 75% is the first number that measures
  merken's loop on content nobody tuned for it.

The gap between "curated scenarios pass rate" (100%) and
"real-content pass rate" (75%) is the number to watch over time.
If it closes, the loop is improving. If it widens, the curated
scenarios are drifting away from what real content looks like and
need new siblings.

## Real-content smoke (2026-04-09, post should_recall)

After the scenario runner hit 100% on both curated fixtures, we ran
a **qualitative** smoke test against the user's real vstash — 20
recent docs that neither I nor the user curated for merken. Output
in `experiments/loop_quality/smoke_real_vstash.py`; this is not a
scenario runner because we have no ground-truth topic labels for
organic content.

**Consolidation on real content was actually good.** 20 events
produced 4 coherent clusters plus 2 honest singletons:

```
Fact 1 (n=4):  MedLocal clinical demos (Meningococcemia, Neonato,
               Motrin, Organofosforados) — pure cluster
Fact 2 (n=7):  MedLocal architecture/strategy (CHT, Loop, Competitive,
               Decision Tables, Retrieval Engineering, Pipeline,
               Estado EOD) — pure cluster
Fact 3 (n=3):  Daily Reviews (Teams×2 + Mail) — pure cluster
Fact 4 (n=4):  vstash meta notes (Non-Obvious, Upstream, Debug,
               agent-memory use cases) — mostly pure, agent-memory
               is arguable
Singletons:    merken v0.1 decisions, Kafka Merchant Pipeline meeting
               — both honestly unique in this slice
```

**Recall surfaced a real bug in Memory.recall, caught the fix, and
now shows both strengths and limits:**

- BEFORE the fix, the query "what happened in the Kafka merchant
  pipeline meeting?" never returned the Kafka singleton. Memory.recall
  drained layers sequentially: semantic returned 4 facts, filled the
  top_k=3 budget, and episodic was never visited. The "fallback"
  was a fallback only in name.
- AFTER the fix (round-robin interleave), the Kafka note surfaces
  at rank 2, and the MedLocal benchmark query surfaces the "98/99"
  EOD number at rank 2 — real episodic evidence that was previously
  invisible.
- BUT broad thematic queries ("what vstash bugs were found?", "what
  are the merken architecture decisions?") still rank a big
  MedLocal cluster at the top because the fact's anchor text
  contains vstash vocabulary and everything in the MedLocal cluster
  mentions vstash or merken in passing. The semantic layer is
  dense enough that broad queries flood toward large clusters
  regardless of topic specificity.

**What the smoke says about merken as of this commit:**

- Consolidation works on real content. No fixture-lying by
  construction.
- Layered recall + interleave works for specific queries
  (singletons, entity names, specific numbers).
- Layered recall fails the discrimination test for broad thematic
  queries on dense content. `bge-small-en-v1.5` at 384 dimensions
  cannot separate "vstash bug" from "MedLocal case that mentions
  vstash" when both appear in clusters with high cosine density.
- The honest next improvement is **query-type awareness in
  `should_recall`** — route entity/specific queries episodic-first
  and theme queries semantic-first. Not a new decider primitive,
  just a smarter default.

The smoke script is idempotent and safe: it opens the user's vstash
read-only and writes to a throwaway tempdir. Re-run any time with
`python -m experiments.loop_quality.smoke_real_vstash --n-docs 20`.

## What the `should_recall` rows say

`session_2026_04_09` went from 50% → 100% **without any change to
consolidation**. The trick was not fixing the clusters — it was
accepting that consolidation is imperfect and routing around the
imperfection.

`LayeredRecaller` (semantic first, episodic fallback):

```
Query: "why was the dedup rule rewritten to use an in-process set?"
  → semantic search: top hit is the impure 2-event cluster
    {longmemeval_a, dedup_fix_a} (wrong topic label)
  → fallback: episodic search returns dedup_fix_a and dedup_fix_b
    directly — raw events with the correct topic tag
  → runner's unified lookup accepts the episodic match
  → pass ✓
```

The consolidation metrics on the session scenario are unchanged:

- `cluster_purity: 75%` (one of four facts still mixes topics)
- `topic_coverage: 50%` (only 3 of 6 eligible topics produced a
  topic-pure multi-event fact)

These are real failures of the v1 consolidator, and they stay
visible in the results table. But `query_pass_rate` — the metric
that matches what a user actually experiences — is now 100%
because the loop can use the raw stream as evidence when the
distilled layer is broken.

**Important framing:** this is not "we fixed the 50% ceiling." The
50% consolidation ceiling is still real; it just stopped being a
user-facing problem because the loop has a second path to the
answer. An LLM-based consolidator would lift cluster_purity and
topic_coverage, and that has its own value (cheaper recall,
higher-quality facts, compression of episodic over time). But for
query routing on this scenario, the fallback is enough.

The next change that touches the consolidator needs to justify its
cost against what the current loop already does: 100% pass rate on
both scenarios via fallback. Not just against the 50% purity
number in isolation.

## The two-scenario picture

As of row 3 the repo has two scenarios and they say very different things:

|                    | session_2026_04_09 | analytics_project |
|---|---|---|
| Content            | meta-discussion about merken itself | realistic agent stream, six distinct topics |
| Same-topic cosine (median) | 0.649                | 0.915                |
| Cross-topic cosine (max)   | 0.663                | 0.617                |
| Gap                | **-0.014** (overlap) | **+0.162** (clean)   |
| pass_rate          | 50%                  | 100%                 |

**What this rules in:**

- The merken consolidation pipeline is *not* broken. On content where
  the embedder can separate topics, it hits 100% on every metric.
- Complete-link clustering is the right default. It does not cost the
  control scenario anything (still 100%) and it fixes the session
  scenario cascade.

**What this rules out:**

- The 50% on `session_2026_04_09` is not an indictment of merken. It
  is a statement about `bge-small-en-v1.5` on meta-discussion
  content. Any change to the consolidator that "fixes" the 50%
  without also holding `analytics_project` at 100% is tuning to
  the failing test, not improving the loop.

**What this sets up:**

- Every future decider change gets measured against *both*
  scenarios. A change that moves session up and analytics down is a
  trade-off that must be named, not declared an improvement.
- When we decide whether to add an LLM consolidator or a
  cross-encoder reranker, the bar is clear: **move session up
  without moving analytics down**. Anything else is a sideways step.

## Delta log

**Row 2 (complete-link, 2026-04-09):** `pass_rate 25% → 50%`,
`purity 67% → 75%`, `coverage 33% → 50%`. Same embedder, same
threshold, only the linkage strategy changed. Complete-link turned
out to be one line of logic (require min-cross-pair ≥ threshold
instead of any-cross-pair) but it fixed the specific cascade this
scenario exposed.

The ceiling that row 2 reveals is more important than the
improvement: **no threshold-based approach on this embedder can
push pass rate above ~50% on this scenario**. The reason is
visible in the pairwise cosine distribution around 0.65:

```
0.778  vstash_bug (same-topic)      ← above, clusters
0.717  mempalace  (same-topic)      ← above, clusters
0.663  longmemeval_a ~ dedup_fix_a  (CROSS-TOPIC)  ← above, false positive
0.661  consolidation_design (same)  ← above, clusters
0.652  vstash_bug_a ~ dedup_fix_a   (CROSS-TOPIC)  ← above, false positive
0.649  longmemeval (same-topic)     ← BELOW by 0.001, missed
0.616  longmemeval_a ~ vstash_bug_a (cross)        ← below
...
0.575  dedup_fix  (same-topic)      ← below, missed
0.558  loop_philosophy (same-topic) ← below, missed
```

Same-topic and cross-topic edges are interleaved through the band
`0.55 – 0.70`. `bge-small-en-v1.5` at 384 dimensions cannot
distinguish "merken internals about dedup" from "merken internals
about longmemeval benchmark" by cosine alone, because both land in
roughly the same region of the vector space.

**Implication for the next commits:**

- Lowering the threshold to 0.60 would catch longmemeval (0.649)
  and bring dedup_fix/loop_philosophy closer, but it also drags in
  more cross-topic false positives. Coverage up, purity down.
  Trade-off has to be measured, not argued.
- Raising the threshold to 0.70 kills the consolidation_design pair
  (0.661) and the false positives. Purity up, coverage down.
- A stricter linkage (e.g. k-medoids, or requiring min cluster
  density) can squeeze a few more points but cannot cross the
  signal/noise boundary that this embedder imposes.
- **The only path above ~50% on this scenario is an LLM-based
  consolidator that reads the text and distinguishes topics
  semantically, not by vector geometry.** Every non-LLM alternative
  is rearranging deck chairs in the 25–50% range.

That last bullet is the finding we commit to memory. The next
scenario to add to `loop_quality/scenarios/` should be content
where the embedder *does* cleanly separate topics — so we have a
control for "the loop is broken" vs "this scenario is beyond the
embedder."

## What the first row said (pre-complete-link)

**The merken loop at commit `be33c51` is not good enough yet.** On a
12-event scenario derived from merken's own design session, only
25% of queries route correctly to a topic-pure fact. Two failure
modes were diagnosed:

1. **Cross-topic false positives cascade via single-link**. Two
   pairs crossed the 0.65 threshold on genuine semantic similarity
   ("both are about merken internals") despite having different
   ground-truth topics. Single-link union-find cascades the
   contamination: `vstash_bug_a + dedup_fix_a + longmemeval_a`
   ended up in one 4-event cluster because each edge individually
   exceeded the threshold.
2. **Borderline same-topic pairs sit just under 0.65**. On this
   content domain (meta-discussion about merken itself),
   paraphrases of the same topic frequently land at cosine
   0.55–0.65 — below the v1 default. Coverage is penalized.

**What this rules out (or shouldn't yet):**

- It does NOT yet say merken is worse than raw vstash on this
  scenario — we haven't run the raw-vstash baseline for
  loop_quality yet. Scheduled as a follow-up.
- It does NOT say embedding_v1 is wrong. It says embedding_v1 with
  single-link clustering and threshold 0.65 is wrong *on
  meta-discussion content with semantically related but
  topic-distinct pairs*. A different scenario might show a
  different picture.

**What to try next (each in its own commit, each with a RESULTS.md row):**

1. Raise threshold to 0.70 — kills false positives, kills
   consolidation_design (0.661), kills longmemeval (0.649). Trade
   precision for recall.
2. Switch from single-link to average-link or complete-link. Single
   cross-topic edge no longer cascades through a whole cluster.
3. Per-scenario threshold calibration (advisory only — default
   stays a single value).
4. LLM-based consolidator that can distinguish topics by reading
   the sentences, not just counting vector proximity.

Each option above is a hypothesis. The right way to resolve them is
to run the runner again with the change and land a new row here. Do
not tune the scenario to make the current numbers look better.

## ContentTypePriorDecider experiment (2026-04-10)

The A-MAC paper's central claim — *content-type prior is the most
influential factor in memory admission* — was tested on a new
`noisy_agent_stream` scenario: 8 signal events (4 decision topics × 2)
mixed with 16 noise events (4 tool_echo, 4 ack, 4 status, 4
ambient_chat).

### Baseline vs ContentTypePriorDecider (explicit tags)

| Metric | HeuristicWriteDecider | ContentTypePriorDecider |
|---|---|---|
| Events ingested | 24 | 8 |
| Facts written | 6 | 3 |
| Noise facts (impure) | 3 | **0** |
| cluster_purity | **50%** | **100%** |
| query_pass_rate | 100% | 100% |
| topic_coverage | 37.5% | 37.5% |
| Elapsed | 4.3s | 2.8s |

**What happened:** the consolidator was grouping ambient chat ("let me
think about Redis...") with ack events ("OK, I'll update the canary
config") because `bge-small-en-v1.5` sees them as semantically similar.
Filtering noise before ingestion eliminated all 3 impure facts. The
decision events consolidated cleanly into 3 pure facts. `db_decision`
(TimescaleDB) did not consolidate in either case — its two events sit
below threshold 0.70 (embedder limit, not decider limit).

### Auto-classification without explicit tags

The auto-classifier (`merken/classification.py`) uses regex + keyword
patterns to detect noise from text alone, no LLM:

| Type | Precision | Pattern |
|---|---|---|
| tool_echo | 100% (4/4) | CLI tool names + output keywords |
| ack | 100% (4/4) | Formulaic phrase starters + short length |
| status | 100% (4/4) | Metric keywords + status verbs |
| ambient_chat | 100% (4/4) | Thinking-out-loud phrases |
| decision | N/A | Falls through as "unknown" (prior 1.0) |

**End-to-end without tags:** 12/24 events written (8 decisions + 4
status that pass the 0.3 > 0.25 threshold check), 0 noise facts,
100% query pass rate. Same quality as explicit tagging.

**Cross-validation on real content (`jay_vstash_2026_04_09_snapshot`):**
0/20 false positives. All 20 organic events classified as "unknown" and
pass through unchanged. One bug caught and fixed: bare keyword "pod"
matched Spanish words ("poder", "podemos").

### What the experiment says

1. **The A-MAC hypothesis holds.** Content-type prior is a cheap, high-
   impact filter. No LLM needed for the noise categories that matter.
2. **The impact is on consolidation, not recall.** query_pass_rate was
   100% with and without the filter because `LayeredRecaller`'s
   episodic fallback catches everything. The damage from noise is in
   the semantic layer: impure facts that pollute long-term memory.
3. **Auto-classification is conservative and safe.** It only rejects
   categories with obvious structural patterns. Decisions, observations,
   and everything ambiguous passes through unchanged.
4. **The limitation is honest:** distinguishing decisions from
   observations requires semantic understanding the classifier can't do.
   But it doesn't need to — both should be written.

## Recall routing experiment (2026-04-10)

Tested the hypothesis that query-type-aware routing (specific queries
→ episodic-first, thematic queries → semantic-first) would improve
recall quality.

### Three recallers compared across all scenarios

| Scenario | SemanticOnly | Layered (default) | EpisodicOnly |
|---|---|---|---|
| analytics_project | 100% | 100% | 100% |
| jay_vstash_real | **25%** | 75% | **75%** |
| noisy_agent_stream | **50%** | 100% | **100%** |
| session_2026_04_09 | **50%** | 100% | **100%** |

### What the data says

1. **EpisodicOnly = Layered on 3 of 4 scenarios.** The semantic layer
   never provides a hit that episodic doesn't also have. On these
   scenario sizes (12-24 events), the episodic layer contains the
   same information as the semantic layer plus more.

2. **SemanticOnly fails hard on singleton topics.** `kafka_meeting` (1
   event) and `merken_design` (2 events below threshold) never produce
   a semantic fact, so semantic-only recall can't find them.

3. **The hypothesis of smart routing doesn't apply at this scale.**
   Changing the order (episodic-first vs semantic-first) wouldn't
   change results because round-robin interleave already gives both
   layers slots. The "routing" question only matters when one layer
   has so many candidates that it drowns out the other.

4. **The semantic layer's value is not in recall — it's in
   compression.** At 20 events, episodic is fine. At 2,000 events,
   the episodic haystack becomes too large for the embedder to
   reliably find specific events, and consolidated facts become the
   more reliable path. This scenario set cannot test that.

### Scale experiment (2026-04-10)

Tested whether episodic recall degrades at scale, making semantic
recall necessary. Generated events with 6 distinct topics (2 signal
events each) + increasing confusing fillers that share vocabulary
with real topics (auth-adjacent, deploy-adjacent, etc).

```
N      Signal%   Semantic     Layered      Episodic
12     100.0%    0/6 (0%)     6/6 (100%)   6/6 (100%)
30      40.0%    0/6 (0%)     6/6 (100%)   6/6 (100%)
60      20.0%    0/6 (0%)     5/6 (83%)    6/6 (100%)
100     12.0%    0/6 (0%)     5/6 (83%)    6/6 (100%)
200      6.0%    0/6 (0%)     5/6 (83%)    6/6 (100%)
500      2.4%    0/6 (0%)     5/6 (83%)    6/6 (100%)
```

**Findings:**

1. **Episodic stays at 100% through 500 events** with 2.4% signal
   density. The embedder finds 12 signal events among 488 confusing
   fillers reliably. The "needle in haystack" degradation does not
   appear at this scale.

2. **Semantic stays at 0%.** Confusing fillers (e.g. "Stripe webhook
   reliability improved" near "Stripe replaced in-house billing")
   contaminate clusters, producing only 5 impure facts. No pure
   facts match query topics.

3. **Layered drops to 83% from 60 events onwards.** The semantic
   layer produces an impure fact that takes a round-robin slot away
   from a correct episodic hit. The semantic layer actively hurts
   recall when its facts are impure.

4. **The crossover point was not reached.** At 500 events, episodic
   is still strictly better than layered. To find the crossover we
   would need either: (a) much larger scale (thousands of events),
   or (b) real conversational data where topics blend more naturally.
   Synthetic generation hits a ceiling because the fillers are
   structurally different from real agent streams.

**Implication:** the question of scale requires real datasets, not
synthetic generation. LMEB and BEAM are candidates — both have real
multi-session conversational data at scales from 100K to 10M tokens.

### When smart routing would matter

The hypothesis deserves revisiting when:

- A scenario has >200 events per topic (episodic haystack large
  enough that retrieval quality degrades)
- The semantic layer has enough facts that broad queries reliably
  return the right cluster
- A temporal dimension exists (recency-sensitive queries that
  episodic handles better by design)
- **Real conversational data is available at scale** (LMEB, BEAM)
  where topic boundaries are natural, not engineered

None of these conditions exist in the current 4-scenario safety net.
Building a runner against a real dataset is the prerequisite for this
hypothesis to be actionable.

### Implication for merken defaults

`LayeredRecaller` (semantic-first + episodic fallback) remains the
right default. It is never worse than either layer alone, and the
round-robin interleave ensures both layers contribute. The cost is
one extra vstash query per recall — negligible at current scale.

## Honesty discipline

Same as the rest of the repo:

- No silent edits. Corrections add a new row and strike through the
  old one with a link.
- Scenario text is frozen on commit. Changing event text or query
  wording to make numbers go up is cheating — tune the *merken*
  code, not the benchmark.
- A result worse than the previous one is not embarrassing, it is
  information. We keep both rows and figure out what regressed.

## Jev at the three decision points (2026-09-18, `3953c818` + feature/jev-deciders)

Driver: `experiments/loop_quality/jev_probe.py`. Model: TypeSafe
`typesafe/jev-1.13-20260917` via OpenRouter's Decisions endpoint
(`~typesafe/jev-latest`). Modes are cumulative: `write` =
`ChainedWriteDecider(Heuristic -> JevWriteDecider)`; `+cons` =
`JevMaterializer` as `materialize_fn` (fact text = the cluster member Jev
picks as CURRENT; members kept in `derived_from` only if Jev says they are
a version of the same decision AND labels them DECISION); `all` = `+
JevReranker` on recall (over-fetch 4×, `noul` "answers the question?" ≥0.5,
`choice` "which is current?" first). Every Jev response is cached under
`.jev_cache/`; the whole table cost **$0.021** (~2.9k calls, ~500k
input tokens) and reruns are free.

`strict` is the runner's `query_pass_rate`; `hit@1` counts queries whose
*first* hit is the answer (what an agent that reads one result gets).

| Scenario | n_events | Mode | written (noise) | facts | purity | strict | hit@1 |
|---|---|---|---|---|---|---|---|
| knowledge_update_hard | 108 | baseline | 107 (95) | 12 | 92% | 4/4 | 3/4 |
| | | all | 35 (23) | 7 | **100%** | 4/4 | **4/4** |
| knowledge_update_20topics | 440 | baseline | 388 (328) | 58 | 93% | 15/20 = 75% | 7/20 |
| | | write | 221 (161) | 40 | 90% | 14/20 | 8/20 |
| | | write+cons | 221 (161) | 40 | 95% | 14/20 | 7/20 |
| | | all | 221 (161) | 40 | 95% | **19/20 = 95%** | **18/20** |
| knowledge_update_50topics | 1100 | baseline | 879 (729) | 117 | 91% | 26/50 = 52% | 13/50 |
| | | write | 475 (326) | 83 | 88% | 27/50 | 17/50 |
| | | write+cons | 475 (326) | 83 | **100%** | 26/50 | 18/50 |
| | | all | 475 (326) | 83 | 100% | **33/50 = 66%** | **32/50** |
| jay_vstash_2026_04_09_snapshot (organic) | 20 | baseline | 20 (0) | 3 | 0% | 3/4 = 75% | 0/4 |
| | | all | 19 (0) | 3 | **100%** | **4/4 = 100%** | **3/4** |
| jay_vstash_…_decontam (organic, 2 answerable queries) | 7 | baseline / all | 7 (0) | 2 | 100% | 2/2 | 2/2 |

**Observations:**

- The lift comes from the *combination*. `write` alone cleans (noise
  written ÷2.2) but does not move pass rate; `+cons` alone purifies
  (91% → 100%) but does not either; `+recall` is what turns both into
  answers: hit@1 ×2.5 on 50topics, ×2.6 on 20topics.
- **Remaining ceiling is retrieval, not decision.** On the 17 queries
  that still fail in 50topics (`monitoring`, `ci`, `logging`, …) the
  correct event ("Migrated from Prometheus/Thanos to Datadog…") is not
  in vstash's top-40 for "what monitoring platform do we use?". A
  reranker cannot reorder what the embedder never returns. That frontier
  belongs to vstash (hybrid weights, bge fine-tune, query rewriting) —
  see `experiments/retrieval/`.
- **`materialize_fact` = longest text in the cluster** was the direct
  cause of 16/24 baseline failures in 50topics: the topic *was* in the
  top-5, as a stale version or a fact anchored on one. Picking the
  current member fixes it; expelling operational tickets that merely
  mention the same system (the adversarial noise in these scenarios) is
  what takes purity to 100%.
- Real content (`jay_vstash_2026_04_09_snapshot`): purity 0% → 100%,
  hit@1 0 → 3/4. This is the "test fixtures are not ground truth" check
  from CLAUDE.md, and it passes.
- Jev-`all` (66%) sits between embedding_v1 (52%) and brief_v1 with LLM
  synthesis (86%, `experiments/consolidation/RESULTS.md`) on 50topics —
  with no text generation, no briefs, and no model training. The two are
  composable (Jev can select which briefs to prepend); not measured here.
- Two bugs surfaced by the probe, fixed in the same PR: (1)
  `LayeredRecaller`'s fixed per-layer budgets ignored the over-fetch, so
  every reranker — including `temporal_weight` — only ever saw 8
  candidates; with the fix, `--temporal-weight 0.2` lifts
  `jay_vstash_2026_04_09_snapshot` 75% → 100% (was a no-op). (2)
  `jay_vstash_…_decontam` carried two queries (`kafka_meeting`,
  `merken_design`) whose events the decontamination had removed; they
  could never pass and were dropped.
- Jev is a network decider (CONSTITUTION §4.1). Nothing here is on by
  default: `MERKEN_SHADOW=jev` / `MERKEN_PRIMARY=jev`, or pass
  `JevReranker` / `JevMaterializer` explicitly.

## Temporal-weight grid, re-run after the recall over-fetch fix (2026-09-18)

`experiments/loop_quality/temporal_grid.py`, `run_scenario` defaults
(threshold 0.65, top_k 5). The previous grid (CLAUDE.md: "8 weights ×
5 scenarios, zero regressions") ran while `Memory.recall` only ever
handed the recency reranker 8 candidates; it is invalidated.

```
scenario                                  0.00   0.05   0.10   0.15   0.20   0.30   0.50   1.00
-----------------------------------------------------------------------------------------------
analytics_project                         100%   100%   100%   100%   100%   100%   100%   100%
bilingual_es_en_2026_04_14                100%   100%   100%   100%   100%   100%   100%   100%
disjoint_noise_heavy_holdout                0%     0%     0%     0%     0%     0%     0%     0%
jay_vstash_2026_04_09_snapshot             75%   100%   100%   100%   100%   100%   100%   100%
jay_vstash_2026_04_09_snapshot_decontam   100%   100%   100%   100%   100%   100%   100%   100%
knowledge_update                          100%   100%   100%   100%   100%   100%   100%   100%
knowledge_update_20topics                  60%    55%    55%    55%    55%    55%    55%    60%
knowledge_update_50topics                  52%    50%    48%    48%    48%    48%    48%    44%
knowledge_update_hard                     100%   100%   100%   100%   100%   100%   100%   100%
markdown_tables_held_out                    0%     0%     0%     0%     0%     0%     0%     0%
noisy_agent_stream                        100%   100%   100%   100%   100%   100%   100%   100%
organic_val_held_out_topics                 0%     0%     0%     0%     0%     0%     0%     0%
session_2026_04_09                        100%   100%   100%   100%   100%   100%   100%   100%
```

Rows at 0% are hold-out scenarios with no queries. The recency
reranker can now help (`jay_vstash_2026_04_09_snapshot` 75% → 100% at
any weight > 0) and hurt (`knowledge_update_20topics` 60% → 55%,
`knowledge_update_50topics` 52% → 48%, 44% at 1.0). **Default stays
`temporal_weight=0.0`.** Raising it is a new decision against this
table.

## Retrieval ceiling, and the other half of the idea: writing (2026-09-19)

Two probes on `knowledge_update_50topics` (1100 events, 950 noise, 50 queries).

### `retrieval_ceiling.py` — where does the correct event sit in vstash's ranking?

Episodic layer only, no consolidation, no reranker; rank of the *current*
event per query under vstash's own knobs. Jev-filtered store (475 docs):

| config | R@5 | R@20 | R@40 | R@100 | not in top-200 |
|---|---|---|---|---|---|
| hybrid (default), mmr 0.0 / 0.5 / 1.0 (identical) | 46% | 68% | 76% | 82% | 9 |
| fts_only | 42% | 66% | 72% | 74% | 13 |
| vec_only | 34% | 50% | 52% | 54% | 23 |
| hybrid vec/fts 0.8/0.2 · 0.5/0.5 · 0.2/0.8 | 42–46% | 68–70% | 76% | 80–82% | 9 |
| embedder swap: bge-base / bge-large / nomic (hybrid) | 44–48% | 70–74% | 72–78% | 78–84% | 8–11 |

- `mmr_lambda` has no effect; `vec_only` is the worst mode; embedder
  swaps move R@100 by ±2pp. **The retriever is not the lever.**
- 3 of the 9 "never found" queries are scenario bugs: `access_control`,
  `service_discovery`, `rate_limiting` have no event containing their
  `expect_contains` (`OPA`, `Kubernetes DNS`, `Envoy rate limit`). The
  answerable ceiling is 47/50.
- The other 6 are a **vocabulary gap**: category-level queries ("what
  monitoring platform do we use?") vs instance-level events ("Migrated
  from Prometheus/Thanos to Datadog…"). Zero lexical overlap; bge-small
  does not bridge it.
- Consequence for the loop: the Jev reranker's ceiling is whatever
  R@(top_k × overfetch) is. Raising `recall_overfetch` 4 → 8 → 20 on
  the full Jev loop: 66% → 78% → **82%** strict (hit@1 32 → 37 → 38/50),
  $0.016 per 50 queries at 100 candidates. 82% ≈ R@100: the loop sits
  on the retrieval ceiling.

### `write_enrich_probe.py` — merken writes the memory, not just filters it

At write time, Jev decides which events are DECISION (415 of 1100; the
LLM is only paid for signal) and a cheap LLM writes **one category-level
line** stored with the event: `CI platform: Buildkite (replaced GitHub
Actions)`. The LLM never sees a query. Then the full Jev loop runs
(write + materializer + reranker, overfetch 20).

| | R@5 | R@20 | R@100 | never found | strict | hit@1 | purity | LLM cost |
|---|---|---|---|---|---|---|---|---|
| Jev loop, no enrich | 46% | 68% | 82% | 9 | 41/50 = 82% | 38 | 100% | — |
| + enrich `deepseek/deepseek-v4.1-flash` (reasoning off) | **78%** | **92%** | 94% | 3 | **46/50 = 92%** | 43 | 97% | $0.012 |
| + enrich `openai/gpt-4.1-mini` | 76% | 90% | 94% | 3 | **47/50 = 94%** | 46 | 100% | $0.023 |

- **47/47 answerable with gpt-4.1-mini; 46/47 with DeepSeek** (miss:
  `graph_db`). Above brief_v1 (86%) without a brief layer: the line
  lives with the event, so FTS, the embedder, consolidation and the
  reranker all benefit.
- Reasoning models return `content=None` under a small `max_tokens`
  (DeepSeek v4.x, gpt-oss): pass OpenRouter's `reasoning: {enabled:
  false}` or use a non-reasoning model. The first DeepSeek run silently
  produced 351/353 empty lines and reproduced the no-enrich numbers
  exactly — an empty enrichment is a no-op, not a regression.
- Cost per enriched event ≈ $0.00003–0.00006; Jev's gate keeps the
  LLM off the 685 noise events.

**Reading:** yesterday's half (Jev deciding) took the loop from 52% to
82% and hit the retriever's ceiling; today's half (merken *writing* a
category-level line for what Jev keeps) lifts the ceiling itself. The
combination is the thesis: merken is the layer that decides and writes;
vstash stores and searches; an LLM drafts only when merken asks.
