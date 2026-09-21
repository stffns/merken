# LongMemEval results

## Scope

This file records absolute R@5 numbers on LongMemEval. **It does not
measure whether merken's loop adds value** — LongMemEval is chat-replay
with no duplicates in the haystack, so the decision primitives collapse
to "ingest everything." For the benchmark that tries to catch loop
value, see `../../loop_quality/`.

## How to read this file

Every row records:

- **Date** — when the run was made.
- **Commit** — the merken commit SHA the run was made against.
- **Baseline** — `vstash` (substrate only) / `merken-always` (no
  filtering) / `merken-heuristic` (Phase 1 default decider).
- **Subset** — `longmemeval_s` (full distractor haystack) /
  `longmemeval_oracle` (oracle context only — sanity, not signal).
- **n** — number of questions evaluated. **Anything below ~50 is
  sanity, not signal.**
- **R@5** — recall @ 5 with 95% bootstrap CI (1000 iters, seed 0).
- **API/q** — API calls per query. Engram is local-first; always 0.
- **Notes** — what was different about this run, what we learned.

## Results

| Date | Commit | Baseline | Subset | n | R@5 (95% CI) | API/q | Notes |
|------|--------|----------|--------|---|--------------|-------|-------|
| 2026-04-08 | `2bcf502` | `merken-always` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | First real-data run. **Sanity only — n=3 produces a degenerate CI.** |
| 2026-04-08 | `2bcf502` | `merken-heuristic` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Same caveat. With no exact duplicates in the haystack, behaves identically to `merken-always`. |
| 2026-04-08 | `2bcf502` | `vstash` | `longmemeval_s` | 3 | 1.000 [1.000, 1.000] | 0 | Substrate-only baseline. Same caveat. |
| 2026-04-08 | `e18d7d4` | `merken-heuristic` | `longmemeval_s` | 10 | **0.900** [0.700, 1.000] | 0 | First non-degenerate CI. 9/10 hits. seed=42. |
| 2026-04-08 | `e18d7d4` | `vstash` | `longmemeval_s` | 10 | **0.900** [0.700, 1.000] | 0 | Identical hit set to merken-heuristic — confirms the heuristic decider is a no-op on this dataset. seed=42. |
| 2026-04-13 | `5a6c820` | `vstash` | `longmemeval_s` | **500** | **0.964** [0.948, 0.978] | 0 | **Phase A complete.** Full n=500 run, seed=42. Positions merken's substrate at parity with mempalace's claimed 96.6% raw (CIs overlap). |
| 2026-04-13 | `5a6c820` | `merken-heuristic` | `longmemeval_s` | **500** | **0.964** [0.948, 0.980] | 0 | Identical R@5 to vstash raw. Budget redistribution fix (commit `42d40ef`) closed the gap that existed at n=10. Heuristic decider is a no-op on this dataset (no duplicates). |
| 2026-04-14 | `482025e` | `merken-heuristic` | `longmemeval_s` | 100 | **0.980** [0.950, 1.000] | 0 | **Embedder swap probe.** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` instead of the default `BAAI/bge-small-en-v1.5`. n=100 (not 500) because the question is "does multilingual regress on English" — a no-worse check that tolerates a wider CI. CI [0.950, 1.000] overlaps the bge row's [0.948, 0.980] → no regression. Colab CPU, 40 min. Combined with the bilingual scenario in `../../loop_quality/RESULTS_multilingual.md`, this clears multilingual as the recommended embedder for bilingual users. |

### Wall-clock cost (informational, not part of the metric)

| Baseline | n | Elapsed | Per-question |
|---|---|---|---|
| `merken-always` | 3 | 92.0 s | ~30.7 s |
| `merken-heuristic` | 3 | 170.2 s | ~56.7 s |
| `vstash` | 3 | 129.2 s | ~43.1 s |
| `merken-heuristic` | 10 | 405.5 s | ~40.6 s |
| `vstash` | 10 | 355.8 s | ~35.6 s |
| `vstash` | **500** | 8923.0 s (2.5h) | ~17.8 s |
| `merken-heuristic` | **500** | 10166.6 s (2.8h) | ~20.3 s |

The n=500 run used vstash 0.28.0 batch ingest for the vstash baseline
(single-transaction writes), cutting per-question cost from ~35s to
~18s. merken-heuristic still ingests sequentially (events pass through
the decider one by one) at ~20s/question — 14% overhead from the audit
log, consistent with the n=10 measurement.

Run on Mac (MLX backend on Apple Silicon, vstash 0.28.0).
~490 turns ingested per question median.

**Full 500-question extrapolation, this hardware:** ~6 hours per
baseline. CONSTITUTION §9's "under 5 minutes on a laptop" target is
**not currently met** for the full LongMemEval_s split. Two options:
(a) use a faster local embedder profile in vstash, (b) accept that
LongMemEval needs an overnight run and amend §9's language. Tracked
in the chat history; not yet in an issue.

## What the n=500 run says (Phase A complete)

- **merken-heuristic and vstash are tied at 0.964 (R@5).** CIs
  overlap completely: [0.948, 0.978] vs [0.948, 0.980]. The budget
  redistribution fix (commit `42d40ef`) closed the gap that existed
  at earlier runs where the empty semantic layer was stealing slots.
- **96.4% positions merken at parity with mempalace's 96.6% raw
  claim.** The 0.2pp difference is well within the CI. Engram's
  substrate (vstash with bge-small-en-v1.5) is competitive with the
  strongest verified raw-mode claim in the space.
- **18 questions missed out of 500.** These are the questions worth
  investigating — each represents a retrieval failure where the
  correct session's turns were in the haystack but didn't land in
  top-5. A `--dump json` flag would help identify patterns.
- **Heuristic exact-dedup still contributes zero on this dataset.**
  Same R@5 as vstash raw. Confirmed at scale what n=10 showed.

## What the n=10 run said (historical)

- merken-heuristic and vstash tied at 9/10 (R@5 = 0.900). CI was
  [0.700, 1.000] — too wide for claims. The n=500 run narrowed this
  to [0.948, 0.980].

## Findings from the first real-data runs

1. **The pipeline works.** End-to-end ingest → recall →
   session-attribution → R@k → bootstrap CI matches the design. No
   schema surprises against the real LongMemEval-cleaned dataset.
2. **Heuristic exact-dedup adds nothing on LongMemEval.** Real
   haystacks have no exact duplicates, so `HeuristicWriteDecider`'s
   `dup_exact` rule never fires and merken-heuristic collapses to
   merken-always behaviorally. This rule's value can only show in
   live agent loops where the same content gets re-ingested. **The
   benchmark in this directory cannot validate this rule.** It needs
   a different benchmark — the one in `../../loop_quality/`.
3. **Recall-based dedup was prohibitive.** The original
   `HeuristicWriteDecider` ran a vstash hybrid search on every write
   to detect duplicates. That made ingest O(N²) per haystack and the
   3-question run was killed mid-flight at ~25 minutes per baseline.
   Replaced with an in-process `set[str]` of normalized text. Same
   semantics, hash-fast. The recall callable stays in `WriteContext`
   for future similarity-based deciders that genuinely need it.
4. **Audit log overhead is not free.** Every decision (write or skip)
   triggers an extra `vstash.remember` call to the audit collection.
   On 3 questions × ~490 turns × 3 baselines, that's ~4400 extra
   writes beyond the user-facing ingest. Worth measuring on a bigger
   run before we decide whether to batch audit writes.
5. **Wall-clock variance between baselines is unexplained.** With one
   DB per (baseline, question), `merken-always` (92 s) < `vstash`
   (129 s) < `merken-heuristic` (170 s) is suspicious — the three
   baselines should be within ~10% of each other on identical
   hardware. Possibilities: model warmup spread across baselines,
   audit collection growth, or a per-DB cold-start cost.

## Competitive positioning (updated 2026-04-13)

The n=500 result positions merken against the published landscape:

| System | R@5 | Mode | Actually tests the system? |
|---|---|---|---|
| **merken** | **96.4%** [0.948, 0.980] | raw, full loop | **Yes** — decider, recaller, audit all active |
| mempalace "raw" | 96.6% | ChromaDB only | **No** — issue #214 showed the benchmark only calls ChromaDB, no mempalace code |
| mempalace rooms | 89.4% | with palace features | Yes — 7pp below merken |
| mempalace AAAK | 84.2% | with compression | Yes — 12pp below merken |
| Mem0 | ~85% | hybrid + GPT-4 | Yes — LLM in path, higher cost per query |

merken is the only system in this table that (a) publishes a CI,
(b) runs its actual decision loop during the benchmark, and (c)
matches the raw-retrieval ceiling without an LLM.

## Honesty discipline

If a number we publish here turns out to be wrong, the fix is to add
a new row with the corrected number *and* leave the old row in place
with a strikethrough and a link to the correction. We do not silently
edit history. See `notes/prior-art.md` for the cautionary tale that
pinned this rule down.

The n <= 10 numbers above are **absolute positioning only** -- they
cannot support any claim of the form "merken matches X" or "merken
beats Y." Claims like that require n >= 50 with a non-degenerate CI
on the same `longmemeval_s_cleaned` split against the same metric.
Until such a row exists in this table, the claim does not get made
anywhere in the repo.

---

## Mode A answer-quality eval (not R@k)

Different question. The table above measures whether retrieval
surfaces a chunk from the answer session (R@k). `mode_a_eval.py`
adds a layer: **did we actually give the user a correct answer?**
That means running a Builder on each question, optionally with
retrieved chunks, and scoring the final answer against the ground
truth with an LLM-as-judge. See
`experiments/retrieval/longmemeval/mode_a_eval.py`.

Three conditions per question:

- **control** -- Builder alone (llama3.1-8b), confident-mode system
  prompt, no retrieval.
- **rag** -- Builder with top-5 retrieved chunks (3-way dual
  retrieval, same as Mode A) inlined into the user prompt. No
  Judge.
- **mode_a** -- full Mode A v4 pipeline (draft + dual-3 retrieval
  + claim-level Judge + deterministic annotator + provenance
  footer).

Oracle: Gemini 2.5 Flash scores `(question, ground_truth, answer)`
triples as `supports | partial | contradicts | neutral`. Correct =
`supports + partial`. Different model family from the
Builder/Judge on purpose, to avoid intra-family bias.

### Run 1 (2026-04-21) -- N=49 / seed 42 / longmemeval_s

Commit: `ccf6ce2` base (Mode A v4, PR #33 merged) + this branch's
eval harness. One question hit an `APIConnectionError` from
Cerebras and was logged as an error row rather than breaking the
loop (per-question try/except working as intended).

#### Headline

| condition | correct | supports | partial | contradicts | neutral | avg_tok | avg_s |
|---|---|---|---|---|---|---|---|
| control | **8.2%** (4/49) | 1 | 3 | 3 | 42 | 288 | 0.7 |
| rag | **71.4%** (35/49) | 30 | 5 | 9 | 5 | 2210 | 1.4 |
| mode_a | **71.4%** (35/49) | 33 | 2 | 7 | 7 | 8759 | 5.3 |

**RAG and Mode A tie on top-line correctness.** Mode A is NOT
better than naive RAG on these 49 questions. Mode A costs ~4x
the tokens and ~4x the wall time for equal correctness.

Mode A telemetry:
- **Grounded (verbatim `quoted_evidence` present): 40/49 = 81.6%.**
  This is the value Mode A buys over RAG -- every answer with a
  cite-able source trail.
- Claim-level leak (>=1 unsupported sub-claim): 5/49 = 10.2%.
- Sub-claims total: 125, of which 14 = 11.2% are unsupported.
- Judge top-level verdict distribution: contradicts 40 / no_claim
  5 / neutral 4. Mode A overrode the Builder draft in 82% of
  cases.

#### By question_type

| type | n | control | rag | mode_a | delta (mode_a - rag) |
|---|---|---|---|---|---|
| knowledge-update | 7 | 0% | 57% | **71%** | **+14pp** |
| multi-session | 17 | 12% | 65% | 65% | 0 |
| single-session-assistant | 3 | 0% | 100% | 100% | 0 |
| single-session-preference | 2 | 100% | 50% | 50% | 0 (n=2, noise) |
| single-session-user | 9 | 0% | 100% | 100% | 0 |
| temporal-reasoning | 11 | 0% | 64% | 55% | **-9pp** |

**Mode A wins on `knowledge-update`** (facts that evolve across
sessions, where the later assertion corrects the earlier one).
The Judge / claim-level rigor picks the right chunk instead of
averaging across contradictory ones.

**Mode A loses on `temporal-reasoning`** (questions that need
arithmetic or time-window reasoning). The Judge refuses when it
should have answered, or the Builder's arithmetic fails and Mode
A doesn't catch it.

Trivial lookups (`single-session-*`) are 100% for both -- they
are solved at the retrieval layer.

#### RAG vs Mode A disagreements (4 wins each; cases visible in log)

Mode A wins the cases that need **aggregation across multiple
chunks**:
- "money raised for charity total" (GT $3,750): RAG listed items,
  Mode A summed them.
- "rollercoasters across events July-October" (GT 10): RAG
  enumerated without totaling, Mode A aggregated.
- "engineers I lead (before / now)" (GT 4 / 5): RAG got cut off
  mid-sentence, Mode A produced a clean both-numbers answer.
- "book discount %" (GT 20%): RAG said "not mentioned", Mode A
  retrieved it correctly.

RAG wins the cases where **Mode A over-refuses or mis-aggregates**:
- "how many projects led" (GT 2): RAG guessed "at least one",
  Mode A refused entirely ("I don't have information...").
- "total $ from markets" (GT $495): Mode A summed but got $595.
  The Judge's arithmetic produced a wrong corrected_text.
- "sports event 2 weeks ago": RAG got partial credit, Mode A
  refused. Temporal reasoning weakness.

#### Caveats (code-reviewer pass before the run flagged these)

1. **Same dedup bug on both conditions.** `retrieve()` in this
   run used a 120-char prefix as dedup key. Conversational turns
   with identical prefixes (`"user: "`, `"assistant: "`) alias and
   drop distinct excerpts. Both RAG and Mode A hit this
   identically, so the A/B is still fair, but the absolute
   numbers under-represent what a bug-free retrieval could
   deliver. Fix landed on the same branch for Run 2.
2. **Control uses `confident` builder mode.** The Builder is told
   not to hedge. Refusals still dominate (42/49 = 86% control
   neutral) because the model genuinely lacks the personal
   information. But on the 7 non-refusal controls, the confident
   prompt pushes toward confabulation -- which the oracle scores
   as `contradicts`. Mostly this makes control look worse.
3. **RAG uses question-only query; Mode A uses question + draft
   query.** Two retrieval pools differ. Mode A's draft expansion
   can fetch chunks RAG would miss (knowledge-update advantage)
   OR noisy chunks RAG would skip.
4. **RAG truncates each excerpt to 800 chars** before inlining;
   Mode A's Judge sees the full excerpt text. If an answer lives
   past char 800 in a long chunk, RAG misses it. At N=49 this did
   not dominate the comparison but is a source of residual bias.

#### What this supports (conservative)

- Mode A at v4 **does not improve top-line correctness** over a
  naive inlined-context RAG baseline on LongMemEval_s at N=49.
- Mode A **does improve auditability**: 81.6% of answers carry
  a verbatim source quote; all answers carry the source_id and
  claim-level decomposition in the audit row.
- Mode A **has a real edge on `knowledge-update` questions**
  (+14pp over RAG) and a real weakness on `temporal-reasoning`
  (-9pp).

#### What this does NOT support

- That Mode A is production-ready as a drop-in replacement for
  RAG. Same correctness at 4x cost is a negative trade unless
  the auditability / grounded-source property is worth the
  premium.
- That the 4x token cost buys nothing -- it buys the grounded
  evidence trail, which is invisible in "did the oracle say
  supports?" but central to the "verifiable truth with source"
  thesis.
- That temporal-reasoning regressions are unfixable -- the
  Judge's arithmetic behavior and over-refusal rate are both
  tunable.

### Next moves (surfaced by this run)

- **Rerun with dedup fix (Run 2)**, same seed, same N. Measure
  whether both conditions move together (expected) or whether
  Mode A recovers ground vs RAG (less expected but possible).
- **Tune Judge for temporal questions.** The `-9pp` on
  temporal-reasoning comes mostly from over-refusal. A prompt
  that lets the Judge emit `contradicts` when arithmetic can
  be inferred from excerpts (rather than demanding a verbatim
  quote for the computed answer) would likely recover several
  points.
- **Token-cost levers.** 4x is steep. Cost profiling on the v1
  audit rows: excerpts dominate the Judge input. Score-threshold
  cutoff + hash-dedup (the latter already shipped) should cut
  ~20-30%.
- **Graduate to N=100 or N=500** once the cost and Judge tuning
  stabilise. Current N=49 gives a direction but CIs are wide.

---

## Mode C answer-quality eval (streaming decider + KV-splice)

Same oracle + scoring rubric as the Mode A section above, applied
to Mode C -- the local-first, one-generation shape with a
continuous streaming decider and mid-stream KV-cache splices.
The benchmark lives at
`experiments/retrieval/longmemeval/mode_c_benchmark.py`.

Builder candidates (local mlx):

- `gemma-4-E2B-it-MLX-4bit` -- ~2B active / ~4B total MoE, 4-bit.
- `gemma-4-E4B-it-MLX-4bit` -- ~4B active / ~8B total MoE, 4-bit.

Retrieval substrate: the same `cerebras_midloop.retrieve` 3-way
dual helper the Mode A baselines use, so head-to-head comparisons
share the retrieval path.

Measurement correction (landed 2026-04-21, pre-grid): the oracle
prompt truncates the candidate to the FIRST 2000 chars. Mode C
outputs are ~3000-4000 chars (thinking preamble + answer body),
so head-truncation silently delivered the PREAMBLE to the oracle
instead of the answer. Fix: strip `<channel|>` preamble, then
tail-truncate `[-2000:]`. Pre-fix 33% dropped to honest 13.3%.
All numbers below are post-fix. Writeup in
`notes/mode-c-continuous-decider.md`.

### Mode C knob grid (2026-04-21) -- N=30 / seed 42 / longmemeval_s

Branch: `feature/mode-c-e2e-demo` (PR #36).

Seven variants of Mode C plus the honest baseline, same 30
questions (seed=42), identical retrieval substrate. Builder
varies per row.

| run | model | force_first | multi-chunk | correct | sup/par/con/neu | tok/q | wall/q | splc/q |
|---|---|---|---|---|---|---|---|---|
| baseline | E2B | - | - | **13.3%** (4/30) | 3/1/6/20 | 425 | 10.3s | 2.23 |
| H1 | E2B | t=30 | - | **23.3%** (7/30) | 4/3/9/13 | 452 | 11.9s | 2.13 |
| H2 | E2B (q-only) | - | - | **13.3%** (4/30) | 4/0/3/23 | 452 | 7.2s | 2.43 |
| H3 | E4B | - | - | **26.7%** (8/30) | 6/2/10/12 | 800 | 43.2s | 2.80 |
| H12 | E2B | - | top-K | **33.3%** (10/30) | 9/1/3/17 | 454 | 9.8s | 4.27 |
| H1+H12 | E2B | t=30 | top-K | **13.3%** (4/30) | 2/2/7/19 | 484 | 9.2s | 3.73 |
| **H3+H12** | **E4B** | **-** | **top-K** | **40.0%** (12/30) | **9/3/6/12** | **800** | **38.3s** | **5.03** |
| H1+H3+H12 | E4B | t=30 | top-K | **33.3%** (10/30) | 8/2/6/14 | 800 | 36.2s | 4.77 |

Multi-chunk policy (H12): on each decider firing, retrieve top-5
from vstash, keep the chunks where `score >= 0.0161` (up to 3),
splice them all in one firing up to a 2000-token budget. Replaces
top-1-per-firing, which deterministically spliced the rank-1 hit
even when the correct chunk sat at rank 2-3 of the pool.

Force-first (H1): unconditional decider firing at token t=30,
bypassing heuristic regex patterns. Motivated by the observation
that HeuristicClaimDetector rarely fires before t=700 on gemma
outputs (the preamble tokens are not claim-shaped), by which
point the Builder has committed to a refusal.

### Headline

- **H3+H12 is the winning config at 40.0%** -- nearly 3x the
  honest baseline, on the same 30 questions, same retrieval
  substrate, same oracle.
- **H12 (multi-chunk retrieval) is the single biggest lever
  (+20pp).** The BBQ diagnostic that motivated it showed the
  correct chunk at retrieval rank 3 while top-1-per-firing was
  spliceing rank 1 -- the model never saw the right content.
- **H3 (E4B Builder) adds +13pp on its own, +6.7pp on top of
  H12.** E4B's extra capacity is what turns multi-chunk context
  into correct answers instead of refusals.
- **H1 (force-first-fire) helps alone (+10pp) but is
  anti-additive with multi-chunk** (H1+H12 = 13.3%,
  H1+H3+H12 = 33.3% vs H3+H12 = 40.0%). Forcing retrieval at
  t=30 before the model has oriented to the question causes
  the three spliced chunks to confuse rather than ground the
  output.
- **H2 (question-only retrieval) is a no-op on this corpus.**
  Flat at 13.3%. The window_text drift that H2 was meant to fix
  does not dominate on LongMemEval.

### Mode C vs RAG vs Mode A

With the honest baseline fixed at 13.3% and the winning config
at 40.0%:

| shape | correct | tok/q | wall/q | API $/q |
|---|---|---|---|---|
| RAG-k3 (llama3.1-8b Cerebras) | 70-74% | 1394-2288 | ~1.1s | ~$0.0006 |
| Mode A v4 (llama3.1-8b + 235b Judge) | 64-71% | 9048 | ~5.3s | ~$0.008 |
| Mode C H3+H12 (E4B local) | **40.0%** | 800 | **~38s** | **$0** |

Mode C closes ~60% of the RAG gap with zero API spend, but
trades API cost for wall time: ~38s/q on E4B 4-bit MLX vs
~1s on Cerebras. The architectural claim (continuous decider +
mid-stream splicing works at all) is validated; the correctness
claim is "cheaper than RAG but only 57% as accurate".

### Per-question-type breakdown (H3+H12 vs baseline)

| type | baseline | H3+H12 | delta |
|---|---|---|---|
| knowledge-update | 0/3 | 2/3 | +2 |
| multi-session | 0/8 | 2/8 | +2 |
| single-session-assistant | 0/1 | 1/1 | +1 |
| single-session-preference | 0/1 | 1/1 | +1 |
| single-session-user | 4/8 | 5/8 | +1 |
| temporal-reasoning | 0/9 | 1/9 | +1 |

The multi-chunk splice mechanism closed the knowledge-update
and multi-session zeroes that were the most damning holes in
the baseline. Temporal-reasoning (1/9) remains the stubborn
category where local E4B with KV-splice does not yet compete.

### What this supports

- **The production shape (continuous streaming decider +
  mid-stream KV-cache splice) works mechanically on local mlx
  inference.** 30/30 runs completed, zero crashes, audit-row
  provenance captured for every splice.
- **The multi-chunk retrieval policy (H12) is load-bearing.**
  Top-1-per-firing was the hidden failure mode that the initial
  13.3% number exposed.
- **Builder capacity matters.** E4B's +13pp over E2B on the
  same pipeline confirms the gap was not all plumbing.

### What this does NOT support

- That Mode C is production-ready to replace RAG on an
  open-domain memory task. 40% vs 70-74% is a real correctness
  gap.
- That the refusal problem is fully solved. 12/30 questions
  still oracle as `neutral` (refusal / abstain), all on the
  E4B winning config -- something structural about
  gemma-it-4-bit-MLX is still refusing memory-backed questions.
- That these numbers generalize beyond N=30. Graduate to N=100
  before claiming stability.

### Artifact trail

Per-question audit rows in
`experiments/retrieval/longmemeval/mode_c_runs_v3/mode_c_n30_seed42_{tag}.jsonl`
(one file per grid cell; tags: (baseline), H1, H2, H3, H12,
H1_H12, H3_H12, H1_H3_H12). Per-stage debug trace on a winning
case lives at
`experiments/midloop_concept/medlocal/mode_c_trace.py`.

### Follow-up grid (2026-04-21 PM) -- retrieval-quality knobs

After the H3+H12 winner landed, a debug trace on a failing case
(`qid=1faac195`, "Where does my sister Emily live?", oracle
`neutral`) revealed that the correct chunk (Denver) was at
retrieval rank=1 but the 2nd and 3rd chunks that passed the
0.0161 absolute threshold were unrelated noise (Sweden welfare,
spinning Emily distractor, rabbit fur). A probe showed that
simpler queries produce much cleaner score gaps -- e.g.
`"Emily live"` gave target=0.0167, rank-2=0.0086 (2x gap), while
the production query `question + "\n" + window` gave
target=0.0167 but rank-3=0.0164 (~1% gap, noise passes).

Three retrieval-quality hypotheses tested on top of the H3+H12
winner (E4B + multi-chunk):

- **H14 relative threshold** -- cutoff = top1_score * 0.5
  (adapts to query-noise regime).
- **H15 short window** -- retrieval query uses only last 80
  chars of the 40-token window.
- **H16 question-only + threshold 0.008** -- re-measure H2
  with an absolute cutoff calibrated for the question-only
  (no window) regime where scores naturally collapse.

| run | correct | sup/par/con/neu | tok/q | wall/q | splc/q |
|---|---|---|---|---|---|
| H3+H12 (winner) | **40.0%** | 9/3/6/12 | 800 | 38.3s | 5.03 |
| H3+H12+H14 | 40.0% | 10/2/7/11 | 800 | 37.4s | 8.40 |
| H3+H12+H15 | 36.7% | 8/3/4/15 | 800 | 38.7s | 4.73 |
| H3+H12+H16 | 20.0% | 5/1/5/19 | 800 | 33.5s | 8.50 |

**None of the 3 improved over H3+H12.** The Emily-probe insight
did not generalize:

- **H14** held at 40% but spliced 67% more chunks per question
  (8.4 vs 5.03). The extra noise-tolerant chunks neither helped
  nor hurt -- the Builder ignored the extras. This is a robustness
  signal for H3+H12: adding more borderline chunks does not
  degrade correctness.
- **H15** (-3.3pp) lost on temporal and multi-session questions;
  the short window cut off reasoning state the decider needed to
  formulate a query.
- **H16** (-20pp) was the instructive failure. With the same
  question-only query across 3 firings, dedup pushed each
  subsequent firing to ranks 4-7 of a static pool -- and the low
  0.008 threshold accepted them all. The KV cache filled with
  three rounds of increasingly-off-topic chunks; the Builder
  defaulted to `neutral` (refusal) 19/30 times.

### What this new signal says about the gap

The 30pp correctness gap between Mode C H3+H12 (40%) and RAG-k3
(70%) is NOT primarily a retrieval-quality gap. With two
orthogonal attempts to clean up the retrieval pool (H15 short
window, H16 threshold calibration) failing and a third (H14
relative threshold) landing exactly flat, the evidence points
to the remaining gap living in:

1. **Builder refusal behavior** -- 12/30 `neutral` on H3+H12
   and 19/30 on H16 where noise increased. gemma-4-E4B-it
   defaults to "not in memory" when confidence dips, even when
   the target chunk is in cache. Requires prompt engineering
   (H6) or non-refusing Builder (H11) to attack.
2. **Temporal reasoning (1/9 on H3+H12)** -- not a retrieval
   problem; arithmetic across time-stamps is a Builder
   capability issue.

Retrieval-quality follow-ups are parked. Next moves target the
Builder side.

### Measurement correction #2 (2026-04-22) -- multi-channel extraction

Hand-auditing Variant A/B H6 smoke runs uncovered a second oracle
extraction bug that had been silently corrupting every N=30 grid
cell in this section:

- The extractor was ``raw.rsplit("<channel|>", 1)[-1][-2000:]``.
- When the Builder emitted multiple answer blocks before budget
  ran out (thinking -> channel -> answer -> turn -> thinking ->
  channel -> answer -> turn -> thinking-cut-by-budget), rsplit
  returned whatever came after the LAST ``<channel|>``, which
  was typically an incomplete thinking preamble. The earlier
  correct answers were invisible to the oracle.
- Separately: after an answer the Builder sometimes spammed
  ``<turn|>`` until budget -- thousands of consecutive tokens.
  That spam crashed Gemini into ``oracle_parse_failure``, also
  scored as ``neutral``.

Confirmed on two cases:

1. `3b6f954b` (Melbourne) emitted ``University of Melbourne``
   as a complete ``<channel|>...<turn|>`` block twice, but the
   old extraction served the third (truncated) thinking block
   to the oracle. Verdict was ``neutral``. Fixed extraction
   serves the last complete block, verdict ``supports``.
2. `4fd1909e` (Imagine Dragons) emitted ``Xfinity Center`` and
   then 600+ ``<turn|>`` tokens. Oracle parse-failed on the
   turn spam. Fixed extraction strips ``<turn|>`` runs and
   surfaces the correct answer, verdict ``supports``.

Fix shipped in `mode_c_benchmark.py`:
```
answer_blocks = re.findall(r"<channel\|>(.*?)<turn\|>", raw,
                           flags=re.DOTALL)
candidate = answer_blocks[-1].strip() if answer_blocks else ...
candidate = re.sub(r"(<turn\|>)+", "", candidate)[-2000:]
```

Rescorer: `experiments/retrieval/longmemeval/mode_c_rescore.py`
runs the corrected extraction against existing audit rows and
emits parallel `.jsonl` files under `mode_c_runs_v3_rescored/`
so historical runs can be revalidated without re-generating.
Oracle spend: ~$0.01/call * 330 calls = ~$3.30.

### Rescored grid (post-fix, 2026-04-22)

| run | old | **rescored** | delta |
|---|---|---|---|
| baseline | 13.3% | **30.0%** (9/30) | +5 |
| H1 force-first t=30 | 23.3% | **40.0%** (12/30) | +5 |
| H2 question-only | 13.3% | 30.0% (9/30) | +5 |
| H3 E4B Builder | 26.7% | 33.3% (10/30) | +2 |
| H12 multi-chunk | 33.3% | 33.3% (10/30) | 0 |
| H1+H12 | 13.3% | 30.0% (9/30) | +5 |
| H3+H12 (old winner) | 40.0% | 36.7% (11/30) | -1 |
| **H1+H3+H12** | 33.3% | **50.0% (15/30)** | +5 |
| H3+H12+H14 | 40.0% | **46.7% (14/30)** | +2 |
| H3+H12+H15 | 36.7% | 36.7% (11/30) | 0 |
| H3+H12+H16 | 20.0% | 30.0% (9/30) | +3 |

### Production winner: H1+H3+H12 (50.0%)

The earlier "H1 is anti-additive with multi-chunk" claim was a
measurement artifact. With the fixed extraction:

- **H1 force-first-fire adds +6.7pp** on top of H3+H12 (36.7% ->
  46.7%... correction: 36.7% -> 50.0% with H3+H12 alone vs
  H1+H3+H12). H1 is strictly additive.
- **H3+H12+H14 is +10pp over H3+H12 rescored** (33.3% vs 46.7%
  when H14's relative threshold replaces the absolute). H14 is
  additive too -- also misclassified as "no-op" in the broken
  grid.
- **H2 is still a no-op** (30% = baseline). The only
  retrieval-side change that does nothing.
- **H15 / H16 stay rejected** even rescored.

### Hand-audit of H1+H3+H12 fails (15/30)

Gemini 2.5 Flash was validated as a fair judge on this subset:

| category | n | rationale |
|---|---|---|
| Builder emitted "not in memory" literally | 10 | oracle neutral, correct |
| Builder gave specific wrong number | 3 | oracle contradicts, correct |
| Builder stuck in thinking loop (no answer block) | 2 | oracle neutral/contradicts, correct |
| **False negatives from oracle** | **0** | -- |

So the remaining 50% gap to 100% is genuine Builder failure:
- 67% of fails: refusal floor (the "not in memory" habit that
  H6 preface variants partly, but not fully, dislodge).
- 20% of fails: hallucinated numbers on aggregation /
  knowledge-update questions.
- 13% of fails: never-commit loops ("2024-05-" repeated until
  budget; stuck mid-thinking).

### Mode C vs RAG vs Mode A -- final

With honest baseline and winner both rescored:

| shape | correct | tok/q | wall/q | API $/q |
|---|---|---|---|---|
| RAG-k3 (Cerebras llama3.1-8b) | 70-74% | 1394-2288 | ~1.1s | ~$0.0006 |
| Mode A v4 (Cerebras + 235b Judge) | 64-71% | 9048 | ~5.3s | ~$0.008 |
| Mode C H1+H3+H12 (E4B local MLX) | **50.0%** | 800 | **~38s** | **$0** |

Gap to RAG-k3 narrowed from the pre-rescored 30pp to **~20pp**.
Still meaningful but not the chasm the broken measurement
implied.

### H6b (commit-to-context preface) -- new winner (2026-04-22)

Having the extraction fix + the rescored baseline, re-ran the H6
hypothesis (prompt engineering against Builder refusal) on top of
the winning config. Two variants tested on a 4-qid smoke (3
refusals + 1 known winner) before committing to a full N=30:

- **Variant A** -- Remove the explicit "say 'not in memory'" escape
  hatch. Net smoke: 1/4 (preserved winner, flipped wake-up case
  only, Emily/Melbourne kept refusing with substituted phrases like
  "I do not have information"). The phrase was convenient but not
  load-bearing -- gemma-E4B has the refusal habit intrinsically.
- **Variant B** -- Replace the preface with a "commit to the
  best interpretation of the context" instruction that affirms
  answers live in the context and forbids the "do not have"
  escape. Smoke: 3/4 true supports + 1 false-negative that the
  extraction fix revealed as a fourth supports (Imagine Dragons
  `<turn|>` spam crashing the oracle).

Promoted Variant B to N=30 with the full
`H1+H3+H12+H6b` stack:

| config | correct | sup/par/con/neu | tok/q | wall/q |
|---|---|---|---|---|
| H1+H3+H12 rescored (prior winner) | 15/30 = 50.0% | ~/~/~/~ | 800 | 37s |
| **H1+H3+H12+H6b** | **17/30 = 56.7%** | 15/2/8/5 | 800 | 36s |

Per-type vs the rescored prior winner:

| type | H1+H3+H12 | H1+H3+H12+H6b | delta |
|---|---|---|---|
| knowledge-update | 2/3 | 1/3 | -1 |
| multi-session | 2/8 | 2/8 | 0 |
| single-session-assistant | 1/1 | 1/1 | 0 |
| single-session-preference | 0/1 | 0/1 | 0 |
| **single-session-user** | 7/8 | **8/8** | **+1** |
| **temporal-reasoning** | 3/9 | **5/9** | **+2** |

Key moves:

- **single-session-user perfect (8/8).** Every direct-lookup
  question with the target in a cacheable chunk now succeeds.
  Emily/Denver flipped, Melbourne flipped to partial, every
  previously-refusing lookup committed to the correct answer.
- **temporal-reasoning 3/9 -> 5/9.** Wake-up-times
  (`gpt4_2c50253f`), brother's graduation days
  (`8c18457d`), vehicle-first-February (`gpt4_76048e76`) all
  flipped from neutral to supports. The preface broke the
  "not in memory" habit on questions where the answer was in
  cache but the Builder was hedging.
- **knowledge-update -1.** One case that had accidentally
  landed as partial in the prior run now contradicts because
  the model now commits confidently to the wrong number
  instead of refusing. Net trade: we want committed answers.

### Refusal vs hallucination trade

Variant B breaks the refusal floor but unmasks the underlying
hallucination ceiling:

| failure mode | H1+H3+H12 (prior) | H1+H3+H12+H6b |
|---|---|---|
| "not in memory" refusals | ~10/30 | **5/30** |
| Wrong-number hallucinations | ~3/30 | **8/30** |

The Builder now commits to answers it could not aggregate
correctly (multi-session totals, temporal deltas requiring
arithmetic). H6b fixes the easy cases (single-session lookups
with the target in cache) but can't give the model reasoning
capability it doesn't have.

### Updated Mode C vs RAG vs Mode A

| shape | correct | tok/q | wall/q | API $/q |
|---|---|---|---|---|
| RAG-k3 (Cerebras llama3.1-8b) | 70-74% | 1394-2288 | ~1.1s | ~$0.0006 |
| Mode A v4 (Cerebras + 235b Judge) | 64-71% | 9048 | ~5.3s | ~$0.008 |
| Mode C H1+H3+H12+H6b (E4B local) | **56.7%** | 800 | ~36s | **$0** |

Gap to RAG-k3 now **~13-17pp**, down from 30pp under the broken
extraction, down from 57pp under the even-more-broken head
truncation of the very first run. The remaining gap is
predominantly multi-session aggregation and knowledge-update
recency questions that Builder capability (not retrieval)
constrains.

### Next move

Graduate from H6 prompt engineering to H11 Builder swap:
qwen-2.5-3b-instruct or llama-3.2-3B-instruct via mlx-lm. Same
pipeline, different Builder. If the hallucination ceiling is
gemma-specific (safety-induced number confusion), a different
local model may lift the 8/30 contradict count closer to zero.
If the ceiling persists, the answer is Builder reasoning
capability, not model swap.

### H11 Qwen3.5-4B Builder swap (2026-04-22) -- rejected

Swapped the Builder to `mlx-community/Qwen3.5-4B-OptiQ-4bit` (4B
params, Apache 2.0, 262K context, `enable_thinking=False` toggle
for no preamble). Same pipeline (H1+H3+H12 + Variant B preface).

First smoke result: **0/4 correct**. Root cause was NOT the model
-- it was the splice envelope. Qwen read the V1 envelope literally
as ChatML-style turn markers and hallucinated 6+ forged copies of
the same chunk with incremented source indices. Splices=1 but the
model emitted the splice pattern as its own output. gemma had
masked this bug because its safety-tuning re-anchored to the user
question; Qwen is less safety-tuned and happily continued the
pattern.

Fix: envelope V2 (fenced `<<<MEMORY_EXCERPT>>>...<<<END>>>` block,
harder to read as a turn) + strip leading `"user:"/"assistant:"`
prefixes from chunk text before splicing. Second smoke: **2/4**.
Graduated to N=30.

Final: **Qwen3.5-4B v2env = 13/30 = 43.3%.** -13.4pp vs gemma
winner (56.7%). Qwen is 2.4x faster wall-clock (14.6s/q vs
35.6s/q) but trades correctness for speed.

Per-type comparison reveals a clear split:

| type | gemma H6b | Qwen3.5 v2env | winner |
|---|---|---|---|
| knowledge-update | 1/3 | **2/3** | Qwen +1 |
| multi-session | 2/8 | **4/8** | Qwen +2 |
| single-session-preference | 0/1 | **1/1** | Qwen +1 |
| single-session-assistant | 1/1 | 1/1 | tie |
| single-session-user | **8/8** | 5/8 | gemma +3 |
| temporal-reasoning | **5/9** | 0/9 | gemma +5 |
| **total** | **17/30** | 13/30 | **gemma +4** |

Qwen is stronger on aggregation and knowledge-update. Gemma
dominates temporal-reasoning because its thinking-mode-by-default
spends 300-400 tokens on arithmetic of dates; Qwen with
`enable_thinking=False` jumps to the answer body with no
arithmetic scratch, and loses 0/9 on temporal questions. Flipping
`enable_thinking=True` on Qwen would consume the same budget as
gemma -- no free lunch.

### Artifact shipped even though H11 was rejected

The envelope V2 + `strip_turn_prefixes` plumbing is Builder-
agnostic hardening, worth keeping in the codebase:

- `--splice-envelope {v1,v2}` CLI flag.
- `--strip-turn-prefixes` CLI flag.
- `--disable-thinking` CLI flag for Qwen3+ family.

Future Builder experiments should default to v2 envelope +
strip-prefixes unless the Builder has been empirically verified
on v1. The v1 was fit-for-gemma-only, the v2 is a more robust
starting point.

### Production state at end of session 2026-04-22

**Winner: gemma-4-E4B-it-MLX-4bit + H1+H3+H12+H6b at 56.7%.**

CLI reproduction:
```
python -m experiments.retrieval.longmemeval.mode_c_benchmark \
  --n 30 --seed 42 \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-MLX-4bit \
  --force-first-fire 30 --tag winner \
  --prompt-preface "<Variant B>"
```

Gap to RAG-k3 (70%) stays at ~13pp. H11 did not close the gap;
the gap lives in Builder reasoning capability (gemma) plus
Builder safety refusal (both). Next plausible moves: ensemble
(gemma for temporal/lookups, Qwen for aggregation), or a bigger
Builder (Qwen3.5-9B or Qwen3.5-27B).

### Measurement correction #3 (2026-04-22 late) -- envelope regurgitation

Jay asked: "the judge should evaluate the same way for every
Builder, right?" That pointed at a third measurement artifact
the H11 Qwen experiment exposed:

When the Builder regurgitated splice envelopes into its own
output (Qwen V2 envelope copy, or gemma V1 envelope copy on
some edge cases), the oracle was receiving up to 2000 chars of
**chunk content** -- not the model's actual answer. A Qwen
output that looked like:

```
You

<<<MEMORY_EXCERPT source=X>>>
The user visited sister Emily in Denver...
<<<END_MEMORY_EXCERPT>>>
```

...was being oracled as if "The user visited sister Emily in
Denver..." was the model's answer. It wasn't -- that was the
chunk content the model copied. Verdicts were structurally
inflated.

Fixed extraction strips:
- V2 envelope blocks (`<<<MEMORY_EXCERPT>>>...<<<END>>>`) and
  orphaned open/close tags when budget truncates mid-block
- V1 envelope headers (`[Source: X]` + following chunk text)
- `<think>...</think>` blocks
- `<|im_start|>` / `<|im_end|>` Qwen markers
- Redundant `<turn|>` runs (previous fix)

Rescored both gemma (no change) and Qwen (major drop):

| run | reported | envelope-aware |
|---|---|---|
| gemma H1+H3+H12+H6b | 17/30 = 56.7% | **17/30 = 56.7%** (unchanged) |
| Qwen3.5-4B v2env | 13/30 = 43.3% | **9/30 = 30.0%** (-4 false positives) |

gemma H6b was honest; the Qwen experiment looked -13pp vs winner
but was really **-26.7pp**. H11 is more decisively rejected.

Rule now enforced in `mode_c_benchmark.py` AND
`mode_c_rescore.py`: any future Builder swap MUST pass oracle
extraction that is Builder-agnostic and explicitly envelope-
aware. Four measurement artifacts have bitten this branch
already (head-truncation, channel-rsplit, turn-spam,
envelope-regurgitation) -- the rule deserves its own feedback
memory.

### Final rescored grid (2026-04-22 end-of-session)

All N=30 seed=42 longmemeval_s, honest extraction:

| config | correct | note |
|---|---|---|
| baseline E2B | 8/30 = 26.7% | no knobs |
| H1 force-first t=30 | 12/30 = 40.0% | |
| H2 q-only | 9/30 = 30.0% | no-op |
| H3 E4B Builder | 11/30 = 36.7% | |
| H12 multi-chunk | 9/30 = 30.0% | |
| H3+H12 | 12/30 = 40.0% | |
| H1+H3+H12 | 13/30 = 43.3% | |
| H3+H12+H14 | 13/30 = 43.3% | relative threshold |
| H3+H12+H15 | 11/30 = 36.7% | short window |
| H3+H12+H16 | 8/30 = 26.7% | q-only+low |
| **H1+H3+H12+H6b** | **17/30 = 56.7%** | **WINNER** |
| Qwen3.5-4B v2env | 9/30 = 30.0% | H11 rejected |

H6b preface is worth **+13.4pp** on top of the best non-preface
gemma config (H1+H3+H12 = 43.3%). That is the single largest
intervention in the knob grid. Even H3 Builder swap (E2B -> E4B
= +10pp) and H12 multi-chunk splice (+3pp over baseline) pale
next to it.

Gap to RAG-k3 (70%) stays ~13pp. Next plausible levers remain
Builder-side (ensemble, bigger Qwen) or reasoning-side
(claim-level Judge post-hoc verification, the Mode A pattern
applied to Mode C output).

### Qwen3.5-4B thinking-ON ablation (2026-04-22 late)

To isolate whether Qwen's 0/9 on temporal-reasoning was caused
by `enable_thinking=False` (no arithmetic scratch) or by the
model's underlying reasoning, re-ran the 4-qid smoke with
thinking-ON and `force_first_fire=30`:

| config (4-qid smoke) | Emily | Melbourne | Wake | Imagine |
|---|---|---|---|---|
| Qwen thinking-OFF fire=1 | partial | neutral | neutral | supports |
| **Qwen thinking-ON fire=30** | neutral | neutral | neutral | supports |

Thinking-ON regresses. Inspecting the outputs: Qwen starts its
thinking template ("Thinking Process: 1. Analyze the Request: ...")
and then falls into a ~700-token whitespace/newline loop,
exhausting the 800-token budget without reaching the answer body.
Only Imagine Dragons (where the chunk contains the answer
verbatim and the thinking block finishes quickly) produces a
correct answer.

This separates two previously conflated explanations for the
Qwen4B vs gemma gap:

- It is NOT "Qwen has no scratch space". With thinking-ON it
  has 800 tokens of scratch and still cannot answer.
- It IS architectural: Qwen's thinking template + the 4-bit
  MLX OptiQ quantization produces degenerate thinking blocks
  (whitespace loops, mid-thinking chunk quotation without
  synthesis) on these questions.

Qwen3.5-4B is **decisively rejected** as a Builder for this
task. H11 does not close the gap under any tested toggle. The
winner remains gemma-4-E4B-it + H1+H3+H12+H6b at 56.7%.

### Next-session handle

The 9B variant (`mlx-community/Qwen3.5-9B-OptiQ-4bit`) was
identified as the cleanest follow-up -- same family with 2.3x
params, Apache 2.0, thinking toggle -- but bandwidth on this
session (0.65 MB/s measured) made the 6GB download
impractical. Save for a session with better bandwidth. The
exact question the 9B would answer: "is the Qwen gap
parameter-count or architectural?" -- our thinking-ON ablation
already suggests architectural, so 9B may not rescue.

### H18 aggregation+temporal preface -- gap closed (2026-04-22)

After parking H11, the 13 remaining fails on the gemma H6b winner
split as:
- 6/8 multi-session fails (wrong numbers on aggregation)
- 4/9 temporal-reasoning fails (date arithmetic)
- 2/3 knowledge-update fails (recency)
- 1/1 preference fail

H18 targets the first two categories explicitly by extending the
preface with aggregation and arithmetic guidance:

```
For questions asking 'how many', 'total', 'sum', or aggregating
across events, READ ALL excerpts and ADD UP the numbers across
them. Do NOT report a single excerpt's number when the question
needs the total. For questions asking about days/weeks/months
between events, identify the two dates and compute the difference.
```

Smoke on 4 pinned qids (1 lookup + 1 multi-session + 1 temporal +
1 winner): 4/4. Graduated to N=30.

| config | correct | sup/par/con/neu |
|---|---|---|
| H1+H3+H12+H6b prior winner | 17/30 = 56.7% | 15/2/8/5 |
| **H1+H3+H12+H18 NEW WINNER** | **21/30 = 70.0%** | 16/5/6/3 |

Per-type deltas (vs H6b):

| type | H6b | H18 | delta |
|---|---|---|---|
| multi-session | 2/8 | 4/8 | **+2** |
| temporal-reasoning | 5/9 | 6/9 | **+1** |
| single-session-preference | 0/1 | 1/1 | **+1** |
| single-session-user | 8/8 | 8/8 | 0 (preserved) |
| knowledge-update | 1/3 | 1/3 | 0 |
| single-session-assistant | 1/1 | 1/1 | 0 |
| **total** | **17/30** | **21/30** | **+4** |

The single-session-user perfect score is preserved -- H18
strictly dominates H6b. The aggregation lever is real: "sum
across ALL excerpts" as an explicit instruction rescued 2/6
multi-session hallucinations.

### Final production state (2026-04-22 end-of-session)

**Winner: gemma-4-E4B-it-MLX-4bit + H1+H3+H12+H18 at 70.0%.**
Same territory as RAG-k3 (70-74%). 13 -> 21pp closed in this
branch from the dishonest initial 33% measurement.

Mode C vs RAG-k3 trade-offs at parity correctness:

| axis | RAG-k3 Cerebras | Mode C local H18 |
|---|---|---|
| correctness | 70-74% | **70.0%** |
| tokens/q | 1394-2288 | 800 |
| wall/q | ~1.1s | ~42.8s |
| API $/q | ~$0.0006 | **$0** |
| hosting | cloud | **local** |

Mode C now buys the same correctness at the cost of ~40x wall
time but zero API spend and full local inference. For latency-
sensitive paths the RAG path wins; for privacy-sensitive or
air-gapped deployments the Mode C path is now viable.

### What this session actually demonstrated

Over 2026-04-21 and 2026-04-22 the Mode C pipeline moved from
an "interesting but broken" 13.3% (honest baseline, post
head-truncation fix) to 70.0% (matched RAG). The progression:

1. 13.3% honest baseline (was 33% under head-truncation bug)
2. 36.7% H3+H12 (E4B + multi-chunk splice)
3. 50.0% H1+H3+H12 (+ force-first-fire, rescored with fixed
   channel extraction)
4. 56.7% H1+H3+H12+H6b (+ commit-to-context preface)
5. **70.0% H1+H3+H12+H18 (+ explicit aggregation/arithmetic preface)**

Four measurement artifacts caught en route (head-truncation,
channel-rsplit, turn-spam, envelope-regurgitation). H11 Qwen
Builder swap rejected. H18 preface is the single largest
correctness lever of the entire grid (+13.3pp over H6b).

### H11b abliterated gemma-4-E4B (2026-04-22 late) -- no improvement

Tested `Jiunsong/supergemma4-e4b-abliterated-mlx` (same base as
our gemma winner, safety-refusal ablated via orthogonal
projection on 17 of 42 layers, Apache 2.0). Hypothesis: remove
the residual refusal floor in H18 (~3/30 neutrals) by swapping
to the abliterated variant.

Immediate issue surfaced: abliteration breaks stop behavior.
After emitting the correct answer in the first
`<channel|>ANSWER<turn|>` block, the model hijacks its own turn
and fabricates a new user question + new answer. Our last-block
oracle extraction (gemma-optimal) picks up the hijack tail.

Tested 3 extraction strategies on both N=30 logs so the
comparison is fair:

| Builder | last | first | shortest |
|---|---|---|---|
| H18 gemma | **22/30 = 73.3%** | 21/30 = 70.0% | 19/30 = 63.3% |
| abliterated H18 | 17/30 = 56.7% | 19/30 = 63.3% | 19/30 = 63.3% |

Even under the abliterated-favoring extractors (first, shortest)
it plateaus at 63.3%, well below H18 gemma (70-73%). Oracle
variance is ~±2pp run-to-run (H18 last was 21/30 in the
production run, 22/30 in the rescore).

Why abliteration does not help on this corpus:

- H18 preface already neutralizes most refusals in regular
  gemma; removing safety is largely redundant
- contradicts counts are similar (6-8 on each) -- abliteration
  does not fix arithmetic/aggregation hallucination
- a new failure mode emerges: turn-hijacking contaminates the
  transcript and requires stop-at-first-turn plumbing to even
  measure fairly

H11b rejected. Production winner stays H1+H3+H12+H18 on
gemma-4-E4B-it-MLX-4bit.

### TODO for next session (not blocking)

- **Stop-at-first-turn plumbing**: modify
  `_stream_until_fire_or_eos` in `mode_c_demo.py` to detect
  `<turn|>` following a `<channel|>` block as an end signal,
  not just the tokenizer's EOS. Would let abliterated Builders
  be evaluated without the 6pp hijack penalty and enable
  production use with them if a future task benefits.
- **Oracle consensus**: multi-draw Gemini scoring (3-5 samples
  per answer, majority vote) would collapse the ±2pp run-to-run
  variance to <1pp. Each current headline number has an
  implicit CI of roughly ±2pp just from oracle noise.
- **N=100 graduation**: all current numbers are N=30 seed=42.
  A graduated N=100 pass with the H18 config would confirm the
  70-73% result outside oracle noise.

### H23 force-second-fire + gate (2026-04-22 late) -- rejected

After H18 closed the gap, analyzed the 9 remaining fails by
category and targeted two buckets:

- 2 "empty answer" cases (model stuck in thinking, no commit)
- 2 "recency confusion" cases (knowledge-update picked old value)

Implemented H23 in three levers:

- `force_second_fire_at_token` (default 400): trigger a second
  decider firing after the first fire if no natural cadence
  fire happened. Pulls next-fresh chunks (ranks 4-6 via
  `spliced_sources` dedup).
- `max_total_tokens` override (default 800 -> tested 1200):
  rescue cases where the Builder exhausts budget in thinking
  preamble before emitting an answer.
- Preface addendum: explicit recency rule
  ("most recent value is current, older are stale").
- **Gate on second fire**: skip if a complete
  `<channel|>...<turn|>` block has already appeared in the
  generated stream. Prevents injecting more chunks into an
  already-committed answer.

Smoke 4 qids (4 fail targets): 1/4 (H21+H22 only, no force-
second-fire); 2/4 (full H23). Looked promising. Also smoked
5 qids (2 winners + 3 empty-fails): 3/5, winners preserved.

Graduated to N=30. First attempt without gate showed
regressions -- cancelled. Second attempt with gate:

| config | correct | sup/par/con/neu | tok/q | wall/q | splc/q |
|---|---|---|---|---|---|
| H18 winner | **21/30 = 70.0%** | 16/5/6/3 | 800 | 42.8s | 4.13 |
| H23 gated | 19/30 = 63.3% | 17/2/7/4 | 1200 | 51.9s | 4.67 |

Per-type:

| type | H18 | H23 gated |
|---|---|---|
| knowledge-update | 1/3 | **2/3** (+1, recency worked on 031748ae) |
| multi-session | 4/8 | 3/8 (-1) |
| preference | **1/1** | 0/1 (-1) |
| single-session-user | 8/8 | 8/8 (preserved by gate) |
| temporal-reasoning | 6/9 | 5/9 (-1) |

Trade: +1 in knowledge-update, -3 spread across multi-session /
preference / temporal. The force_second_fire adds chunks that
confuse the Builder in ambiguous cases where it would otherwise
commit a correct partial or supports. Partials (5 -> 2)
converted to contradicts (6 -> 7), and neutrals (3 -> 4).
The gate preserved every single-session-user (8/8) but could
not catch every mid-commitment state in the other types.

Cost: +400 tok/q, +9s wall/q, for -2 correctness. Net
negative.

**H23 rejected.** Production winner stays H1+H3+H12+H18 at
70.0% (or 73.3% on rescore -- oracle variance).

Lesson: the smoke-then-graduate protocol worked as a guard
against *obvious* regressions but not against *distributed*
regressions. A 4-qid smoke showing 2/4 improvement missed
that other 26 questions would each have a small chance of
regressing. For low-signal changes (like H23's -2 net), the
smoke-only gate is insufficient -- a partial N like N=15 or
N=20 before full N=30 would have caught this cheaper than the
full 25-min run.

### H25b bypass score threshold (2026-04-22 last) -- rejected

Debug trace on the d851d5ba charity fail (GT $3,750, H18 answer
$1,000) revealed that a chunk containing ``"I helped raise over
$2,000..."`` scored 0.0159 and was dropped by the 0.0161
absolute threshold, missing by 0.0002. H25b: bypass the
threshold entirely and splice top-3 fresh chunks per firing
regardless of score.

Smoke on 6 qids (4 multi-session fails + 2 winners Imagine,
Emily):

| qid | H18 | H25b |
|---|---|---|
| d851d5ba charity | contradicts | contradicts (no change) |
| gpt4_e05b82a6 rollercoasters | contradicts | contradicts |
| a08a253f fitness | contradicts | contradicts |
| 6d550036 projects | contradicts | contradicts |
| 4fd1909e Imagine | supports | **neutral** (REGRESSED) |
| 1faac195 Emily | supports | supports |
| **total** | **3/6** | **1/6** |

With bypass the retrieval pool brings in noise chunks at very
low scores (0.0050, 0.0144 observed on Imagine) that dilute
the KV cache in winner cases. 0 fails rescued, 1 winner
regressed = strictly worse.

Analysis of why charity stayed contradicts: firing 3 under
H25b DID splice the `$2,000`-containing chunk that was
previously dropped. But GT $3,750 requires summing three
separate events (~$1,000 + ~$2,000 + ~$750). If the third
event's chunk isn't in the retrieval pool at all, no
thresholding change rescues it. The problem is **information
not in retrieval**, not **information dropped by threshold**.

H25b rejected.

### Final takeaway: H18 is the realistic ceiling

The remaining 9 fails at 70% are structural:

- 4 multi-session aggregation fails: chunks with specific
  numbers often not in top-10 retrieval at all. No threshold
  tweak rescues them.
- 2 knowledge-update fails: chunks lack absolute timestamps,
  so recency is ambiguous.
- 2 empty-answer fails: Builder thinking-stuck, not
  budget-related.
- 1 thinking-leak fail: stop-behavior bug.

None are fixable with preface or retrieval threshold tweaks.
Next-level levers require:

- **Claim-level Judge post-hoc** (Mode A pattern applied to
  Mode C): after generation, pass (question, answer, retrieved
  chunks) to Gemini for verification. Can aggregate numbers
  explicitly and correct undercount. Costs one extra API call
  per question.
- **Corpus-level chunking changes**: concatenate related
  conversational turns so aggregations sit in one chunk
  instead of being spread.
- **Re-retrieval with number-density boost**: for "how many /
  total" questions, re-retrieve with a query that prioritizes
  chunks containing numerics/entities.

All three are PR-level work, not session-level knob tweaks.

**Production winner for this session: H1+H3+H12+H18 at 70-73%.**

### H27v2 pool=50 + narrow-aggregation rerank + stop-at-first-turn (2026-04-22 late)

Debug trace on d851d5ba charity fail revealed the 4 $-amount
chunks live at ranks 8, 28, 35, 56 of a 62-chunk pool.
With the default ``RETRIEVAL_POOL=10`` only rank 8 ever made
it to splice; 28+ were invisible. H27v2 combines three
interventions:

- ``--stop-at-first-answer-block``: end generation when the
  Builder emits a complete ``<channel|>ANSWER<turn|>`` block.
  Saves ~30% wall time, prevents abliterated turn-hijack.
- ``--rerank-by-number-density``: detect aggregation intent
  via a narrow regex (``in total|how much money|how many
  times|total amount|sum of|added up|combined|across all|
  all the events|across...events|from...to...``). Broad
  trigger (``how many``) caused 5+ regressions on single-
  count questions (playlists, yarn skeins) -- narrowing
  eliminated those.
- For aggregation questions: expand pool to 50 (vs 10
  default), rerank chunks by numeric-pattern density,
  bypass score threshold so the reranked deep-rank chunks
  actually enter.

Smoke 8 qids: 6/8 = 75% with zero regressions on winners.
Graduated to N=30.

| config | correct | sup/par/con/neu |
|---|---|---|
| H18 winner | 21/30 = 70.0% | 16/5/6/3 |
| **H27v2** | **21/30 = 70.0%** | **19**/2/5/4 |

Tied on top-line but H27v2 produces **+3 more supports** (16 ->
19) and half the partials (5 -> 2). Correct answers become
more confident.

Per-type:
- knowledge-update: 1/3 -> 2/3 (+1, 031748ae engineers now
  applies recency correctly)
- multi-session: 4/8 -> 4/8 (charity + rollercoasters flip
  TO supports via aggregation rerank; 81507db6 + 2b8f3739
  regress from supports, net zero)
- temporal-reasoning: 6/9 -> 5/9 (-1)
- other types preserved

Flip analysis:
- **7 positive flips**: engineers (contradicts -> supports),
  charity & rollercoasters (contradicts -> supports via
  aggregation rerank specifically), preference + 3 partials
  promoted to supports.
- **4 negative flips**: 81507db6 graduation, 2b8f3739 market
  total (multi-session regressions); gpt4_4cd9eba1 +
  gpt4_a2d1d1f6 (temporal to partial/neutral). Distribution
  looks like oracle/sampling variance, not structural harm.

H27v2 is an equivalent alternative to H18 with:
- Same top-line correctness (70%)
- Stronger verdicts on correct answers (+3 supports)
- Faster wall time (~30% via stop-at-first-turn)
- Better tooling for aggregation questions (charity + rollercoasters
  were the two motivating fails and both flipped)

Either works as production config.

### Final session state (2026-04-22 full day)

Two candidates tied at 70% correctness:
- H1+H3+H12+H18 -- simpler pipeline, ~42.8s/q
- **H27v2** (H18 + stop + narrow rerank) -- ~30s/q, +3
  supports-confidence, Builder-agnostic envelope, 50-chunk
  pool for aggregation questions

Gap to RAG-k3 (70-74%) closed on correctness axis.

Interventions shipped:
- H1 force-first-fire, H3 E4B Builder, H12 multi-chunk top-K
- H14 relative threshold factor (optional)
- H18 aggregation+temporal+recency preface
- stop-at-first-answer-block (also helps abliterated)
- narrow-aggregation rerank + conditional pool=50 (H27v2)
- envelope-aware oracle extraction

Rejected: H2, H11 Qwen, H11b abliterated, H15, H16, H23
force-second-fire, H25b bypass always, Chunking A turn-pair
re-ingest, Chunking C broad trigger, **H28 conditional
stop-off** (aggregation exempt, didn't rescue non-agg
regressions), **H30 stop-off globally** (broke aggregation
flips -- stop-at-turn is necessary for aggregation wins).

Remaining fails are structural: retrieval information-not-in-
top-50, corpus-level missing timestamps, Builder-level
arithmetic ceiling. Next levers: Judge post-hoc (Mode A
pattern applied to Mode C output) and/or corpus rechunking.

### Seed robustness check (2026-04-22 EOD) -- the 70% was an artifact

Prompted by the observation that every prior Mode C and RAG-k3
number in this file was measured on seed=42 only. Three-seed
H18 replication + matched RAG-k3 reruns at N=30:

| seed | Mode C H18 | RAG-k3 (t=0.0, k=3) | gap (RAG-MC) |
|---|---|---|---|
| 42 | 21/30 (70.0%) | 22/30 (73.3%) | +3.3pp |
| 43 | 16/30 (53.3%) | 15/30 (50.0%) | **-3.3pp (Mode C wins)** |
| 44 | 15/30 (50.0%) | 19/30 (63.3%) | +13.3pp |
| **mean** | **17.3/30 (57.8%)** | **18.7/30 (62.2%)** | **+4.4pp** |

- Mode C stdev 3.21 correct, range [15, 21]
- RAG-k3 stdev 3.51 correct, range [15, 22]
- Mean gap +4.4pp within ~1.5 stdevs -- not statistically
  significant at N=30 x 3 seeds.

**What this means for earlier claims in this file:**

1. The headline "Mode C H18 matches RAG-k3 at 70%" in the H18
   section above was **seed=42 specific**. Both systems score
   ~70% on that sample because it contained 8 single-session-
   user questions (a category where both H18 and RAG get ~100%).
   A different sample of 30 from the same LongMemEval_s pool
   gives a different number.
2. Honest positioning going forward is the 3-seed mean with
   range: **Mode C H18 = 57.8% +- 10pp**, **RAG-k3 = 62.2% +-
   12pp**. The gap exists but is inside noise.
3. Per-seed asymmetry is informative: seed=44 is where Mode C
   collapses (50% vs RAG 63%, +13pp gap). That asymmetry is
   the specific signal worth investigating -- likely retrieval-
   side since Mode C's 5 structural fails (a08a253f, 6d550036,
   6a1eabeb, 4dfccbf7, gpt4_e061b84g) are mostly retrieval-
   adjacent.

**Preface provenance fix (same session):**
PROMPT_PREFACE_H6B and PROMPT_PREFACE_H18 are now committed
constants in `mode_c_demo.py` (commit ca1257a). Prior runs
passed these inline via `--prompt-preface` without saving the
string to source. An in-session reconstruction from the
RESULTS.md prose produced only 17/30 on seed=42 (vs historic
21/30), confirming the reconstructed string was a different
preface. The recovered constants reproduce 21/30 exactly.
Going forward: every preface in an anchor run must be a named
constant routed via `--preface-name`. Inline prefaces are
banned for anchor runs.

**Going-forward rules surfaced by this session:**

- Single-seed correctness numbers are provisional. 3-seed
  minimum before claiming any ceiling.
- Prefaces used in anchor runs MUST be committed as named
  constants. Inline `--prompt-preface` is for smoke only.
- Benchmark corpus manipulation (session summaries, injected
  timestamps, synthetic facts) is off-limits: it bakes
  benchmark-specific assumptions into merken and invalidates
  the generalization claim. Allowed retrieval-side interventions:
  query-time transforms (HyDE, query expansion), retrieval
  config (top_k, vec_weight, fts_weight, mmr_lambda,
  recency_boost), Judge post-hoc, source hygiene (filtering
  `sharegpt_*` haystack pollution).

## Retrieval v2 on seed=44 -- null result (2026-04-23)

Attack on the seed=44 13pp gap per `notes/mode-c-next-session.md`
task #14. Three changes in one edit:

1. **4th pure-vec pool** in `cerebras_midloop.retrieve()` dual
   mode: `mem.search(query, top_k=top_k, retrieval_mode="vec_only")`
   interleaved alongside hybrid + fts(q+draft) + fts(q-only).
2. **`sharegpt_` filter** in the dedup loop: drop any chunk whose
   ingested title session-id starts with `sharegpt_` (LongMemEval
   haystack pollution, not the user's own conversations).
3. **`--retrieval-pool` override** in `mode_c_benchmark.py` that
   bypasses the aggregation-aware `RETRIEVAL_POOL` /
   `AGGREGATION_RETRIEVAL_POOL` switch when the caller sets it
   explicitly. Used here with `--retrieval-pool 50`.

Code review via `code-reviewer` subagent before the run: zero
blockers, default path preserved for callers that do not pass
`--retrieval-pool`, `sharegpt_` filter is a no-op for non-
benchmark titles.

### Result: +1 flip, within noise

```
v1 (seed=44, retrieval v1): 15/30 (50.0%)
v2 (seed=44, retrieval v2): 16/30 (53.3%)
delta:                      +1 correct (+3.3pp)
flips_ok:                   1  (57f827a0: neutral -> partial)
flips_bad:                  0
stayed_fail:                14
stayed_ok:                  15
```

One lift: `57f827a0` (single-session-preference) went neutral to
partial. No flips on the 7 multi-session or the 6 temporal-
reasoning fails that together account for 13 of the 15 baseline
fails. `gpt4_1e4a8aec` (temporal) moved neutral -> contradicts,
i.e. the retrieval upgrade did not introduce a correct answer
but did push the model into a more specific (still wrong)
reply. `c4a1ceb8` (multi-session) moved contradicts -> neutral,
less-bad but still a fail.

**Gate decision:** plan called for promoting to 3-seed rerun
only if seed=44 reached >= 58%. 53.3% is below that threshold;
RAG-k3 seed=44 remains at 63.3%, gap stays at 10pp vs the 5pp
target. Retrieval v2 is **abandoned as a seed=44 intervention**.
Next up per queue: task #21 (unconditional Judge post-hoc,
local gemma).

### Why the upgrade did not move the numbers

Inspection of the 14 still-failing qids suggests the bottleneck
is not retrieval recall on seed=44:

- **Multi-session fails (7/14)** -- the target facts span
  multiple sessions in the haystack. The Builder surfaces one
  session's content but fails to combine facts across sessions.
  A wider pool / vec-only / sharegpt_ filter cannot close that
  gap because the retrieved chunks were already in the pool --
  the failure is in reasoning over retrieved content.
- **Temporal-reasoning fails (6/14)** -- dates and durations
  require arithmetic the Builder does not do reliably under
  stop-at-turn. H31 splice-awareness or a Judge pass is the
  more targeted intervention here.
- **57f827a0 lifted neutral -> partial** because the pure-vec
  pool brought in a semantically-adjacent preference chunk
  that the FTS pools had ranked below threshold. This is the
  one case where the vocabulary-mismatch hypothesis held.

The code changes are retained (committed) because (a) the
`sharegpt_` filter is a general source-hygiene improvement
applicable to every future LongMemEval run, (b) the pure-vec
pool is defensively sound and costs ~100ms, (c) the
`--retrieval-pool` flag is useful infrastructure for future
experiments. The null result is the load-bearing finding.

## H31 splice-awareness smoke -- rejected (2026-04-23)

Followed the retrieval v2 null result with the cheapest
remaining queue item: H31 = PREFACE_H18 + SPLICE_AWARENESS_V1
block that tells the Builder mid-stream tokens wrapped in
`<<<MEMORY_EXCERPT>>>...<<<END_MEMORY_EXCERPT>>>` are
authoritative retrieved facts. Smoke on 4 pinned qids that H18
gets right on seed=42 (caf03d32, a1eacc2a, 2b8f3739,
gpt4_4edbafa2). Gate: +1 flip with 0 regressions -> promote to
N=30.

Result:

| qid | type | H18 seed=42 | H31 smoke |
|---|---|---|---|
| caf03d32 | single-session-preference | supports | supports |
| a1eacc2a | knowledge-update | supports | supports |
| 2b8f3739 | multi-session (aggregation) | supports | **contradicts** |
| gpt4_4edbafa2 | temporal-reasoning | supports | supports |

1 regression, 0 flips. Gate failed. H31 **rejected**.

Root cause of the 2b8f3739 regression: aggregation question
(total $495 across 3 sales). H18 returned the correct total.
H31 committed to $345 from 2 sales ($225 + $120), missing the
third chunk. The SPLICE_AWARENESS_V1 preface strengthened the
Builder's authority-weighting of the excerpts actually present
in the splice, which made it MORE confident in an incomplete
aggregation rather than looking for missing chunks. Same
envelope, opposite behavior from the intended one.

Lesson: prefaces that strengthen excerpt-authority behave as
multipliers on whatever the retrieval pool happened to surface.
On retrieval misses (aggregation, multi-session), that multiplier
works against correctness.

Two interventions rejected in one day (retrieval v2 +1 flip on
seed=44, H31 -1 on smoke). The evidence converges on the
Builder being the bottleneck: widening retrieval does not help
because the target chunks are already in the pool; tightening
Builder-side excerpt authority does not help either. The
remaining queue item that directly attacks reasoning is task #21
(unconditional Judge post-hoc, local gemma), which re-reads the
Mode C answer against the retrieved chunks and can catch wrong
aggregations / temporal arithmetic after the fact.

## Scratchpad pre-inject k=2 smoke -- rejected (2026-04-23)

Third intervention of the session, probing a different axis: what
if the Builder commits to a direction before mid-stream splicing
fires? Pre-inject top-K chunks on the question alone BEFORE the
Builder starts, formatted as a `[Source: X]` scratchpad block in
the user message, mid-stream splicing still on. New flag
`--pre-inject-k` on `mode_c_benchmark.py`. Default 0 preserves
original Mode C.

Code review via `code-reviewer` subagent before the run: zero
blockers.

Smoke on the same 4 pinned qids (all 4 pass under H18 on seed=42):

| qid | type | H18 seed=42 | pre-inject k=2 |
|---|---|---|---|
| caf03d32 | single-session-preference | supports | **neutral** |
| a1eacc2a | knowledge-update | supports | supports |
| 2b8f3739 | multi-session (aggregation) | supports | **contradicts** |
| gpt4_4edbafa2 | temporal-reasoning | supports | supports |

**2 regressions, 0 flips. Pre-inject k=2 rejected -- worse than
H31.**

### Diagnostic on the two regressions

**caf03d32 (preference fail):** The pre-injected chunks came from
`caf03d32::answer_2fc6aabb::0` and `::6` (topically adjacent
sessions). The Builder entered a meta-cognitive evaluation spiral
over the scratchpad -- analyzing whether the excerpts contained
the specific advice the user wanted, concluding they did not,
then colliding with the preface's "do not say you don't know"
clause and recursing into "Constraint Override" analysis. Output
was a stream of self-reflection, not an answer. Oracle verdict:
neutral.

H18 baseline on the same qid (no pre-inject) answered correctly
because mid-stream splicing triggers AT claim-time: by then the
Builder already has an internal direction from its own draft,
and the spliced chunk either reinforces or stays inert. Pre-
inject changes the dynamic: chunks arrive BEFORE any internal
direction, so the Builder treats them as the sole context and
evaluates their sufficiency directly. That evaluation loop is
the failure mode.

**2b8f3739 (aggregation fail, GT $495):** Pre-inject surfaced 2
of 3 sales chunks. Builder enumerated them explicitly in its
answer ("Sale 1: $225, Sale 2: $120, Total = $345") and
committed. H18 baseline answered correctly -- mid-stream
splicing brought in the third sale chunk later in the stream
when the Builder's draft had generated a query that matched it
more specifically. Pre-inject lock-in happens upfront; mid-
stream splicing has a second chance.

### Pattern across the three rejections today

1. **Retrieval v2** (+1 flip on N=30): widening the pool,
   adding pure-vec, filtering sharegpt_ pollution did not move
   the multi-session / temporal-reasoning fails. Target chunks
   were in the pool but the Builder did not integrate them
   mid-stream.
2. **H31 splice-awareness** (-1 on smoke): preface-level
   authority-multiplier on mid-stream excerpts made the Builder
   commit more confidently to an incomplete aggregation.
3. **Pre-inject k=2** (-2 on smoke): upfront context lock-in
   triggered meta-cognitive evaluation on preference questions
   and early-commit on incomplete aggregation.

All three attempted to feed the Builder more / better / earlier
context. None worked. The failure mode is consistent: the
Builder (gemma-4-E4B local) is the bottleneck, and interventions
that alter what context it sees tend to surface one of two
pathologies -- meta-cognitive over-analysis of the scratchpad
or premature commitment to a subset of the available evidence.

Mid-stream splicing's subtle virtue is that it preserves the
Builder's own reasoning thread before the chunk arrives, and
arrives AT claim-time when the Builder has already committed to
wanting specific evidence. Pre-inject or authority-strengthening
both disrupt that dynamic.

The only remaining queue item that does NOT try to influence
the Builder's context is **task #21 Judge post-hoc**. Judge
lets the Builder run, then a second local-gemma pass re-reads
the answer against retrieved chunks and flags / rewrites
errors. Judge does not try to change how the Builder reasons;
it corrects after the fact. Based on today's evidence, it is
the architecturally correct next experiment.


---

## Step 3 pipeline N=30 seed=44: REJECTED -- -18.5pp vs baseline (2026-04-24)

The merken full-pipeline arm 1 (AlwaysWrite ingest + per-session
`generate_briefs` + layer-scoped dual retrieval with brief prefix)
**regresses** on LongMemEval seed=44 N=30.

- Pipeline+RAG    : 14/27 = **51.9%** (3/30 errored on int-type ground-truth; fix in runner for replication)
- Baseline RAG-k3 : 19/27 = **70.4%** on the same 27 common qids
  (headline 19/30 = 63.3% reported previously)
- Delta           : **-18.5pp** on the common set, 0 gains / 5 losses.

Reference:
- Pipeline run: `pipeline_runs/pipeline_rag_seed44_n30_n30_20260423T074602Z.jsonl`
- Baseline run: `mode_a_eval_grids/grid-rag_k3_seed_robustness_n30_seed44.jsonl`
- Plan          : `notes/step3-pipeline-plan.md`
- Smoke runs    : `pipeline_runs/pipeline_rag_seed44_n1_smoke*`

### Retrieval health

Every question retrieved the max brief budget (pool=9, hits=3, no
collapse). 22/27 retrieved at least one brief from the ground-truth
answer-session. So this is NOT a brief-retrieval collapse.

| answer-session brief retrieved? | correct | wrong |
|---|---|---|
| yes (any top-3 hit is from answer_*) | 13 | 9  |
| no                                    | 1  | 4  |

Even with perfect brief retrieval (4 qids had 3/3 hits from the
answer session), accuracy on those was 3/7, all 4 losses.

### Failure mode: compression loss on specific-fact questions

Five flips, all losses vs baseline:

| qid | type | baseline | pipeline | mechanism |
|---|---|---|---|---|
| `86f00804` | single-session-user | supports | contradicts | Book title dropped from brief; briefs talk about OTHER books the user read/considered. All 3 brief_hits were from the right session -- retrieval was perfect, synthesis lost the specific title. |
| `4f54b7c9` | multi-session (aggregation, GT=5 items) | supports | contradicts | Briefs enumerated only 4 of 5 inherited items. Compression dropped one item (diamond necklace from grandmother). |
| `f685340e_abs` | knowledge-update (GT="not enough info") | supports | contradicts | "every other week" compressed to "weekly" during brief synthesis. Precision loss. |
| `726462e0` | single-session-user (GT=10% discount) | supports | neutral | Answer session's brief was NOT in top-3. Retrieval returned 3 briefs from unrelated sessions. Builder hedged. |
| `gpt4_1a1dc16d` | temporal-reasoning (which-first ordering) | supports | contradicts | Brief labeled "Horror Movie Marathon Planning" mentions Rachel meeting but drops the date. Builder concluded on the wrong ordering. |

The pattern: **brief_v1 compresses specifics out of existence**. Per-
session briefs produce decent topic/state summaries (the 099778bb
aggregation of "women hold 20 leadership positions / team comprises
100 leadership positions" worked), but lose the fidelity needed for
"what was X" questions that dominate LongMemEval.

### Why this is the opposite of the expected result

Step 2 PASS on qid=099778bb (aggregation: 20 women / 100 total =
20%) led us to expect multi-session and temporal-reasoning questions
would become near-trivial. In practice the aggregation-synthesis win
is the exception: that question needed only the SHAPE of the fact,
not the numerical precision. Most LongMemEval questions ("what book,
what discount, how many items, which event first") need the
numerical / nominal specifics, and briefs lose those.

Baseline advantage: raw chunks preserve the original conversational
wording including the specific book title, discount percentage, item
count, event dates. Dual-search retrieval surfaces the right chunk
~70% of the time, and the Builder quotes the value verbatim.

### Gate verdict and pivot

Per `notes/step3-pipeline-plan.md`:
> "negative -> pipeline HURTS. Diagnose (likely retrieval collapse on semantic-layer briefs). Pivot."

Retrieval is NOT collapsing. The brief_v1 shape is the wrong
substrate for LongMemEval's specific-fact question profile.

**Next steps, ordered by signal-per-effort:**

1. **Abandon brief_v1 as LongMemEval substrate.** Keep it in
   production for "topic + temporal state" workloads (its validated
   domain per `experiments/consolidation/RESULTS.md`). Do NOT ship a
   brief_v1-based arm against the LongMemEval headline.
2. **H-F from the plan: structured claim extraction** (typed claims
   preserved per-item) is the architecturally correct pivot if
   merken wants a LongMemEval-specific substrate. Separate project.
3. **Judge post-hoc (task #21 queued from the prior session plan)**
   is the remaining context-agnostic lever. It reads the Builder's
   answer against retrieved chunks and rewrites; does not touch
   brief synthesis. Worth running before any substrate re-design.
4. **Writer retrain via Cerebras-labels loop (Jay 2026-04-24 pivot)**
   remains valid as an independent project -- the 86% median OOD
   filter recall on seed=44 is a known ceiling. Not blocked by this
   null.

### Invariants this run did NOT violate

- No corpus manipulation: production brief_v1 prompt used byte-for-
  byte; per-session wrapper is a LongMemEval-shape adapter.
- Temperature pinned to 0.0 for Builder and brief synthesis.
- Episodic retrieval matches baseline dual regime (layer-scoped).
- cite_footer appended for answer-shape parity.
- Same oracle (Gemini 2.5 Flash), same BUILDER (llama3.1-8b via Cerebras).

### Honest caveats

- N=30 single-seed carries ~±2pp Gemini oracle variance. -18.5pp is
  well outside that noise floor, but the specific gain/loss pattern
  on any one qid is variance-prone.
- 3/30 questions errored on int-type ground_truth (fixed in the
  runner for replication). Baseline handled them but all 3 got
  `contradicts` there too, so they are not the source of the delta.
- This result only tests one seed. Seeds 42 and 43 would confirm
  the regression, but given the mechanism (compression loss on
  specific-fact questions, a property of the prompt not the seed),
  replication is not expected to move the headline.

## Builder scale-up: gpt-oss-120b vs llama3.1-8b (single-seed, 2026-04-27)

LoCoMo 3-seed showed the LoCoMo Builder swap (`cerebras/llama3.1-8b`
to `cerebras/gpt-oss-120b`) lifted the 3-seed mean from 56.3% +- 3.4pp
to 70.7% +- 1.6pp, with the temporal shape going from 43.3% to 80.0%
(+36.7pp). Cross-benchmark transfer to LongMemEval is the natural
follow-up: LME's temporal-reasoning shape was clavado at 36-38%
under every config tested in the prior session (granularity, hybrid
weights, reranker, prompt-fix). Does the Builder lever transfer?

Setup: `experiments/retrieval/longmemeval/run_vstash_ask.py
--seed 44 --n 30 --top-k 8 --backend cerebras --model gpt-oss-120b`.
Per-turn ingest, default vec/fts weights, no rerank -- the canonical
LME 56.7% baseline stack. Apples-to-apples Builder swap.

Oracle wrinkle: Gemini monthly quota was exhausted, so all 30 oracle
calls returned 429. The model answers themselves were generated
correctly. Re-judged with `experiments/retrieval/locomo/rejudge_with_cerebras.py`
which writes an `oracle_cer` field per row using the Cerebras
llama3.1-8b oracle. Baseline 56.7% was already Cerebras-graded;
no oracle-asymmetry between the two columns.

### Results

Single-seed N=30 seed=44 (rejudged with Cerebras):

| question_type             | n  | baseline | gpt-oss-120b | delta    |
|---------------------------|---:|---------:|-------------:|---------:|
| knowledge-update          | 3  | 100%     | 33.3%        | -66.7pp  |
| multi-session             | 9  | 44.4%    | 55.6%        | +11.1pp  |
| single-session-preference | 2  | 50.0%    | 50.0%        | 0pp      |
| single-session-user       | 5  | 100%     | 100%         | 0pp      |
| temporal-reasoning        | 11 | 36.4%    | **63.6%**    | **+27.3pp** |
| TOTAL                     | 30 | 56.7%    | 63.3%        | +6.6pp   |

Trust: +50.0% -> +56.7% (+6.7pp). Per-question: 6 gains, 4 losses,
net +2.

### Findings

- **Temporal-reasoning unsticks on LME too.** +27.3pp (36.4% -> 63.6%)
  echoes the +36.7pp 3-seed move on LoCoMo. The "temporal-reasoning
  is a Builder bottleneck" hypothesis from the prior session is
  empirically validated cross-benchmark.
- **Multi-session +11.1pp.** Smaller but consistent direction.
- **Knowledge-update -66.7pp.** All 3 LME knowledge-update questions
  in this sample were lost: 2 are user-says-two-different-things
  scenarios (Harajuku apartment duration, Crash Course episodes) where
  baseline llama3.1-8b confidently picked one number and gpt-oss-120b
  flagged the contradiction without selecting. The latter is a
  reasoning failure: knowledge-update questions test "use the most
  recent statement", not "detect inconsistency." gpt-oss-120b is
  miscalibrated on this shape -- refusal-on-ambiguity is correct for
  adversarial questions (LoCoMo +trust) but wrong for knowledge-update.
- **N=30 with 3 knowledge-update questions is small.** longmemeval_s
  has ~36 knowledge-update items total; sampling 3 is too few to
  call this a robust regression.
- **Net +6.6pp on a Cerebras-Cerebras-rejudge comparison.** The
  effect is real on temporal-reasoning, marginal-but-positive overall.

### Decisions

- **Builder scale-up transfers cross-benchmark on the temporal shape.**
  This was the single biggest open question from the 2026-04-25
  session ("LME temporal-reasoning stays at 36-38% under EVERY config
  tested"). The answer is: it stays at 36% for retrieval-side levers,
  but moves +27pp under Builder swap. Same lever as LoCoMo.
- **The "shape-targeted LoRA on temporal" Phase 2 move is dominated
  by Builder choice on both benchmarks.** Re-evaluate whether LoRA
  is still the right next move; if gpt-oss-120b is the production
  Builder, LoRA work should target gpt-oss-120b's remaining
  failure modes (knowledge-update calibration, adversarial-vs-
  ambiguity disambiguation).
- **3-seed LME pending.** seeds 42 and 43 should be run before this
  result is treated as a settled cross-benchmark conclusion. Not
  blocking the Phase 2 narrative -- the LoCoMo 3-seed already
  carries the Builder-as-lever conclusion.

### Files

- New artifact: `pipeline_runs/vstash_ask_seed44_n30_gptoss120b-builder-scaleup_20260427T060026Z.jsonl`
- Cerebras rejudge: `pipeline_runs/vstash_ask_seed44_n30_gptoss120b-builder-scaleup_20260427T060026Z_rejudged_cer.jsonl`
- Run log: `pipeline_runs/lme_gptoss120b_seed44_log.txt`

## Builder scale-up 3-seed CONFIRMED + per-shape CORRECTED (2026-04-27 session 2)

After patching `run_vstash_ask.py` to accept `--oracle cerebras`
(mirrors the LoCoMo runner_rerank pattern; the prior version hardcoded
Gemini for grading) and re-running seeds 42 / 43 plus rejudging the
seed=42/43 baselines with Cerebras (the seed=44 baseline was already
Cerebras-graded), the apples-to-apples 3-seed comparison is:

| seed | baseline | gpt-oss-120b | delta |
|------|---------:|-------------:|------:|
| 42   | 19/30 = 63.3%   | 22/30 = 73.3% | +10.0pp |
| 43   | 19/30 = 63.3%   | 20/30 = 66.7% | +3.3pp |
| 44   | 17/30 = 56.7%   | 19/30 = 63.3% | +6.7pp |
| **mean** | **61.1% +- 3.8pp** | **67.8% +- 5.1pp** | **+6.7pp** |

Trust mean: +52.2% -> +56.7% (+4.4pp).

The single-seed-vs-single-seed morning headline ("+6.6pp on seed=44")
landed correctly under 3-seed-mean comparison: +6.7pp. But the
per-shape claims from seed=44 were inflated by single-seed cherry-pick:

| shape (3-seed mean) | baseline | gpt-oss-120b | 3-seed delta | morning single-seed |
|---|---:|---:|---:|---:|
| temporal-reasoning  | 44.4% | 57.1% | **+12.7pp** | +27.3pp (overstated 2x) |
| knowledge-update    | 80.5% | 69.5% | **-11pp**   | -66.7pp (overstated 6x) |
| multi-session       | 45.2% | 58.4% | +13.2pp     | +11.1pp |
| single-session-user | 95.8% | 88.9% | -7pp (saturated, noise) | 0pp |

### Findings (revised)

**Headline +6.7pp 3-seed cross-benchmark transfer holds.** The single-seed
LME morning result was directionally correct on the overall metric.
The per-shape numbers were inflated by ~2x on temporal and ~6x on
knowledge-update; both still real but smaller.

**Cross-benchmark asymmetry:** LoCoMo gets +14.4pp / temporal +36.7pp
3-seed; LME gets +6.7pp / temporal +12.7pp 3-seed. gpt-oss-120b helps
both benchmarks but LoCoMo roughly 2x more than LME. Hypothesis: LME
sessions are 9981 chars mean (LoCoMo 2843) and span much longer
multi-session journals; the Builder scale-up may be hitting a context-
length ceiling on LME that doesn't bind on LoCoMo.

**Per-seed stability splits:** LoCoMo gpt-oss-120b stdev TIGHTENS
(3.4pp -> 1.6pp); LME stdev WIDENS (3.8pp -> 5.1pp). The LME widening
is driven by temporal-reasoning being unstable across seeds: seed=43
shows 0pp move, seed=44 shows +27pp move. The +12.7pp 3-seed mean is
real but seed-to-seed swings are wide.

**The single-seed reporting deprecated rule from 2026-04-25 LoCoMo
applies cross-benchmark.** Morning headlines from seed=44 alone
overstated the per-shape effects by 2-6x.

### Decisions (revised)

- **Builder scale-up cross-benchmark transfer is REAL but UNEVEN**.
  LoCoMo benefits ~2x more than LME. The "Builder is the dominant
  lever" claim holds; the "transfers identically across benchmarks"
  was over-stated.
- **Knowledge-update LME small-N regression (-11pp 3-seed) is real
  but smaller than the single-seed alarm.** Still worth investigating
  the failure mode (gpt-oss-120b refuses-on-ambiguity vs "use most
  recent"). N=10 across 3 seeds is enough to know it's not noise but
  too small for high-resolution forensics.
- **gpt-oss-120b stays as production Builder candidate**. Both
  benchmarks lift; the LoCoMo 3-seed +14.4pp is the strongest signal
  in Phase 2. LME +6.7pp is moderate but not negative.
- **3-seed reporting is mandatory going forward**, even when single-
  seed numbers look strong. The cost is ~10-20 min of compute per
  benchmark; the benefit is preventing 2x-6x overstatements.

### Files (3-seed)

- Seeds 42/43 new artifacts: `pipeline_runs/vstash_ask_seed4{2,3}_n30_gptoss120b-builder-scaleup-3seed_*.jsonl`
- Cerebras-rejudged baselines (for apples-to-apples): `pipeline_runs/vstash_ask_seed4{2,3}_n30_vstash_ask_seed{42,43}_*_rejudged_cer.jsonl`
- Run logs: `pipeline_runs/lme_gptoss120b_seed4{2,3}_log.txt`

## Jev on the recall path — Mode A grid (2026-09-21)

Grids: `grids/jev_recall*.yml`, driver `mode_a_eval.py` (new `rag_jev`
condition kind: dual pool at `pool_k` → `merken.classifiers.jev.JevReranker`
→ top `top_k` → same RAG-baseline Builder prompt). N=49, seed 42,
`longmemeval_s`. Builder **`gpt-oss-120b`** (Cerebras retired
`llama3.1-8b`; override with `MERKEN_LME_BUILDER`). Oracle
`gemini-2.5-flash` routed via OpenRouter (no Gemini key in this
environment; same model family as before).

### Harness bug found on the way: the Builder was being truncated

`MAX_TOKENS_DRAFT = 300` was sized for llama3.1-8b. `gpt-oss-120b`
*reasons* first; at 300 tokens it returned `completion_tokens == 300`
with **empty content** on 9/49 baseline answers and 5/49 Jev answers,
all scored as wrong. Longer context → more reasoning → more blanks, so
the bug penalised the baseline more than the leaner Jev conditions and
manufactured a fake Jev "win" in the first grids (rows kept below for
the record). **Any gpt-oss-120b Mode A number produced with
`MAX_TOKENS_DRAFT=300` — including the 2026-04-27 3-seed 67.8% — is
likely understated;** the April jsonl logs are not in the repo, so
this is a suspicion, not a verified correction.

Two more bit-rots fixed: `mem.search(..., fts_only=True)` →
`retrieval_mode="fts_only"` (vstash ≥0.3x), and `Conversation` now
keeps `session_dates` / `question_date` from the dataset (they were
parsed and dropped).

### Clean comparison (both conditions `max_tokens: 1500`, 0 blanks)

| condition | correct | knowledge-update (7) | multi-session (16) | temporal (11) | single-session (15) | avg tok/q | avg wall/q |
|---|---|---|---|---|---|---|---|
| `rag_t00_k5_mt1500` | **81.6%** (40/49) | 7 | 11 | 7 | 15 | 2380 | 0.5s |
| `rag_jev_v3_k5_pool5_mt1500` | 79.6% (39/49) | 6 | 11 | 7 | 15 | **955** | 2.3s |

Jev v3 = answers-the-question filter (`noul` ≥ 0.5) with `min_keep = top_k`
backfill, real session dates prefixed to every excerpt, and a time-aware
"which text answers for the time the question refers to" `choice`
(present → most recent session; "when I first started" → that session).

Flips rag → jev: +1 (`multi-session`, fitness days), −2 (`multi-session`
"how many graduation ceremonies" — an aggregation count where reordering
5 excerpts lost one; `knowledge-update` "personal best 5K" — the filter
dropped the excerpt with the newer time).

### Reading

- **On LongMemEval, Jev on recall buys tokens, not accuracy.** −1 question
  net at N=49 (inside noise), 2.5× fewer Builder tokens, +1.8 s/q of Jev
  calls. LME chunk retrieval is already at R@5 96%, the dual pool is ≤40
  excerpts, and a 120B Builder reads 2.4k tokens without trouble: there is
  little noise to filter and "current state" is rarely ambiguous in the
  text. This is the opposite regime from `knowledge_update_50topics`
  (90% noise, versions that overwrite each other), where the same
  reranker took the loop from 52% to 82% (`experiments/loop_quality/RESULTS.md`).
- Two failure shapes to remember before putting any filter on a recall
  path: **aggregation questions** (counts/sums across sessions) need
  every partial answer, and **knowledge-update** needs the superseding
  excerpt *kept*, not merely ranked. `min_keep` covers the first only
  partially; a safer default is "reorder, never drop" when the pool is
  already small.
- Superseded rows (kept for the record; both sides truncated):
  `rag_t00_k5` 71.4% (9 blanks) vs `rag_jev_k5_pool5` 71.4% (5 blanks),
  `rag_jev_v2` 73.5%, `rag_jev_v3` 73.5%.
