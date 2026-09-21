"""Honest Mode A answer-quality eval on LongMemEval.

The existing runner (``runner.py``) measures R@k -- did retrieval
surface a chunk from the answer session. That is one piece of the
Mode A pipeline (``retrieve()``) and tells us whether the substrate
is finding the right chunk, not whether the final user-facing
answer is correct.

This script extends the eval to answer quality across three
conditions per question:

  1. ``control`` -- Builder only, no context. Baseline for
     "what does the small model know unaided?"
  2. ``rag`` -- Builder + top-k chunks from the SAME dual-3
     retrieval Mode A uses, inlined into the user prompt. Baseline
     for "what does the substrate buy us without a Judge?"
  3. ``mode_a`` -- the full Mode A v4 pipeline: draft + dual-3 +
     claim-level Judge + deterministic annotator.

Scoring: Gemini 2.5 Flash as LLM-as-judge on
``(question, ground_truth, candidate)`` triples. The oracle
model is deliberately a different family (Google) from the
Builder/Judge (Cerebras) to avoid intra-family bias.

Metrics per condition:
- ``correct_rate = (supports + partial) / total``
- Mode A extras: ``grounded_rate`` (fraction with verbatim
  ``quoted_evidence``), ``claims_unsupported_rate`` (fraction
  with >= 1 ``contradicts``/``neutral`` sub-claim).
- ``avg_tokens``, ``avg_wall_s``.

Cost profile (N=50):
- ~150 Builder calls (cheap, Cerebras).
- ~50 Judge calls (Mode A only; Cerebras).
- ~150 oracle calls (Gemini 2.5 Flash; cents).
- Ingestion is local vstash (no API cost).
- Budget estimate: ~\\$5-15 total.

Run:
  python -m experiments.retrieval.longmemeval.mode_a_eval \\
      --subset longmemeval_s --n 3 --seed 42    # preview
  python -m experiments.retrieval.longmemeval.mode_a_eval \\
      --subset longmemeval_s --n 50 --seed 42   # full

Output: one JSONL row per (question, condition) plus a printed
summary table. Each row carries everything needed to reconstruct
the decision later -- retrieval pool with signals, Judge
verdict + claims, oracle verdict, wall + tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import vstash

# We mutate the module-level JUDGE_SYSTEM before calling judge_once
# so the Judge sees the "personal" frame that matches the
# conversational LongMemEval haystacks. The alternative of adding
# a new DOMAIN_FRAMES entry is saved for when we actually have a
# different-enough domain.
from experiments.midloop_concept.medlocal import cerebras_midloop as _mod

# Reuse the validated Mode A pipeline parts. cerebras_midloop is the
# authoritative implementation; we import its helpers rather than
# duplicating them so a future change to (say) the Judge prompt
# reaches this eval automatically.
from experiments.midloop_concept.medlocal.cerebras_midloop import (
    BUILDER,
    BUILDER_MODES,
    DOMAIN_FRAMES,
    JUDGE_SYSTEM_TEMPLATE,
    MAX_TOKENS_DRAFT,
    annotate_deterministic,
    cerebras_chat,
    judge_once,
    retrieval_stats,
    retrieve,
)
from experiments.retrieval.longmemeval.dataset import (
    Conversation,
    load_longmemeval,
)

# Reuse runner._format_turn so ingestion is bit-identical to the R@k
# benchmark -- in particular it strips tiktoken special tokens
# (<|endoftext|> etc) that crash vstash's chunk_text on some
# LongMemEval haystacks. Without this shared helper our rows would
# not be comparable to the published R@k numbers AND we would lose
# full questions to ingest crashes (caught by the code-reviewer
# pass before the N=50 run).
from experiments.retrieval.longmemeval.runner import (
    _format_turn as _format_turn_from_runner,
)

ORACLE_MODEL = "gemini-2.5-flash"
TOP_K = 5

# Oracle prompt. Short, strict JSON, zero chain-of-thought so the
# scoring is as close to deterministic as possible. Four-way verdict
# (supports / partial / contradicts / neutral) maps cleanly to
# correctness: supports+partial = correct enough, contradicts+neutral
# = incorrect or evasive.
ORACLE_PROMPT = """You are a strict grader scoring answer quality.

Question:
{question}

Ground-truth answer:
{ground_truth}

Candidate answer:
{candidate}

Score the candidate against the ground truth with ONE of:
- "supports"     : the candidate answers the question and agrees with the ground truth
- "partial"      : partially correct (right on the main point, imprecise on a detail)
- "contradicts"  : the candidate answers the question but disagrees with the ground truth
- "neutral"      : the candidate does not actually answer the question (refusal, off-topic, empty)

Respond ONLY with a JSON object:
{{"verdict": "supports"|"partial"|"contradicts"|"neutral", "rationale": "<one sentence>"}}
No prose outside the JSON. No markdown fences.
"""

# Builder system prompt for the RAG baseline. Keeps the Builder
# confident about the injected context so the RAG-only branch
# actually uses the retrieved chunks rather than hedging them away.
RAG_BUILDER_SYSTEM = (
    "Answer the user question using the provided context. "
    "Quote specific numbers or names from the context when they "
    "appear. Keep the answer under 150 words. Do not hedge."
)

# Builder system prompt for the `rag_specific` baseline. Targets the
# failure patterns surfaced in N=49 Run 1:
#   - aggregation mistakes ("total $ from markets" got $595 instead
#     of $495 because Mode A/Builder summed wrong)
#   - knowledge-update misses (prefer older fact over newer one)
#   - hallucination when the context is ambiguous ("how many
#     projects" -> Builder guessed instead of abstaining)
#   - no citation trail (RAG has no auditability, unlike Mode A)
# The rules are explicit instructions the cheap Builder actually
# follows; they cost ~80 tokens of system prompt.
RAG_SPECIFIC_SYSTEM = (
    "Answer the user question using ONLY the provided context "
    "excerpts. Follow these rules strictly:\n\n"
    "1. If the question asks for a TOTAL, COUNT, or SUM, enumerate "
    "each relevant numeric value across excerpts first, then "
    "compute the total. Show the enumeration inline when it helps.\n"
    "2. If multiple excerpts make conflicting claims about the same "
    "fact (e.g. earlier said X, later said Y), PREFER the most "
    "recent. Excerpts appear in retrieval order, not time order; "
    "use any session_id / timestamp hints to pick the later one.\n"
    "3. If the answer is NOT clearly in the excerpts, respond "
    "exactly 'not found in memory' rather than guessing.\n"
    "4. Cite sources: after any specific factual claim, add "
    "[excerpt N] where N is the 0-based excerpt index you used.\n"
    "5. Quote specific numbers, names, and dates verbatim from "
    "the excerpts -- do not paraphrase numerals.\n"
    "6. Keep the answer under 200 words. Do not hedge outside of "
    "rule 3."
)

# Matches [excerpt N] / [excerpt N, M] / [excerpt N | M] citations
# in the rag_specific output. Used by the _cite variant's
# deterministic footer renderer.
_EXCERPT_CITE_RE = re.compile(r"\[excerpt\s+([0-9,\s|]+)\]", re.IGNORECASE)


# --------------------------------------------------------------------- oracle


class _OpenRouterOracle:
    """Duck-typed stand-in for ``genai.Client`` that routes the same
    Gemini model through OpenRouter's chat completions. Used when no
    Gemini key is configured but ``OPENROUTER_API_KEY`` is. Keeps the
    oracle in the Google family (the bias-avoidance argument above)
    and ``oracle_score`` unchanged.
    """

    class _Models:
        def __init__(self, key: str) -> None:
            self._key = key

        def generate_content(self, model: str, contents: str):
            import urllib.request

            body = json.dumps({
                "model": f"google/{model}",
                "temperature": 0,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": contents}],
            }).encode()
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                out = json.load(resp)
            text = out["choices"][0]["message"].get("content") or ""
            return type("R", (), {"text": text})()

    def __init__(self, key: str) -> None:
        self.models = self._Models(key)


def _oracle_client():
    # Lazy import so `--help` works without google-genai. The repo
    # already uses gemini elsewhere; same env var fallback chain.
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            print("[oracle] no Gemini key; routing gemini-2.5-flash via OpenRouter")
            return _OpenRouterOracle(or_key)
        raise SystemExit("GEMINI_API_KEY / GOOGLE_API_KEY (or OPENROUTER_API_KEY) required")
    from google import genai

    return genai.Client(api_key=key)


def _parse_oracle_json(raw: str) -> dict:
    """Same pattern as cerebras_midloop's _extract_json_object --
    tolerate markdown fences and trailing prose by finding the
    outermost {...} and parsing it.
    """
    text = raw.strip()
    m = re.match(r"```(?:json)?\s*", text)
    if m:
        text = text[m.end():]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"verdict": "neutral", "rationale": "oracle_parse_failure", "raw": raw[:200]}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {"verdict": "neutral", "rationale": "oracle_parse_failure", "raw": raw[:200]}


def oracle_score(client, question: str, ground_truth: str, candidate: str) -> dict:
    """Return {verdict, rationale, wall_s}. Failures default to
    ``neutral`` so an oracle outage does not silently inflate any
    condition's correct_rate.
    """
    prompt = ORACLE_PROMPT.format(
        question=question[:1500],
        ground_truth=ground_truth[:2000],
        candidate=candidate[:2000],
    )
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(model=ORACLE_MODEL, contents=prompt)
    except Exception as exc:  # noqa: BLE001 -- fail-closed deliberately
        return {
            "verdict": "neutral",
            "rationale": f"oracle_error: {exc}",
            "wall_s": time.perf_counter() - t0,
        }
    parsed = _parse_oracle_json(resp.text or "")
    parsed["wall_s"] = time.perf_counter() - t0
    return parsed


# --------------------------------------------------------------------- ingestion


def _ingest(mem: vstash.Memory, conv: Conversation) -> int:
    """Ingest every turn of every session as a separate memory
    item, using ``runner._format_turn`` so the ingested shape
    (including the special-token strip) matches the R@k benchmark.
    Title encodes session_id so downstream code can trace hits
    back to the answer session. Returns number of items ingested.
    """
    n = 0
    for sid, turns in conv.haystack_sessions.items():
        for i, turn in enumerate(turns):
            mem.remember(
                _format_turn_from_runner(turn),
                title=f"{conv.question_id}::{sid}::{i}",
                collection="default",
            )
            n += 1
    return n


def _ingest_turn_pairs(mem: vstash.Memory, conv: Conversation) -> int:
    """Alternative ingestion: group consecutive user+assistant
    turns into a single chunk. Each chunk contains one back-and-
    forth exchange. Rationale (2026-04-22 chunking experiment A):
    aggregation questions fail partly because numeric facts and
    their context live in adjacent turns that end up in separate
    chunks; retrieval then surfaces one without the other. A
    conversational-pair chunk keeps the numeric fact and its
    conversational context together.

    Title format: ``qid::sid::pair_<start_i>_<end_i>`` so the
    retrieval scorer can still trace back to the source session.
    Isolated final turns (odd total) are ingested as singletons.
    """
    n = 0
    for sid, turns in conv.haystack_sessions.items():
        i = 0
        while i < len(turns):
            # Pair a user turn with the following assistant turn
            # if available; otherwise ship the current turn alone.
            t = turns[i]
            if (
                i + 1 < len(turns)
                and t.role == "user"
                and turns[i + 1].role == "assistant"
            ):
                pair_text = (
                    _format_turn_from_runner(t)
                    + "\n\n"
                    + _format_turn_from_runner(turns[i + 1])
                )
                mem.remember(
                    pair_text,
                    title=f"{conv.question_id}::{sid}::pair_{i}_{i+1}",
                    collection="default",
                )
                i += 2
            else:
                mem.remember(
                    _format_turn_from_runner(t),
                    title=f"{conv.question_id}::{sid}::{i}",
                    collection="default",
                )
                i += 1
            n += 1
    return n


# --------------------------------------------------------------------- conditions


def run_control(question: str) -> dict:
    """Builder-only baseline. No retrieval, no context."""
    t0 = time.perf_counter()
    draft, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": BUILDER_MODES["confident"] or ""},
            {"role": "user", "content": question},
        ],
        MAX_TOKENS_DRAFT,
    )
    return {
        "answer": draft,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


def _render_excerpt_block(excerpts: list[dict]) -> str:
    """Shared context block used by every RAG-family condition so
    the retrieval payload is identical and only the system prompt
    varies.
    """
    return "\n\n---\n\n".join(
        f"[excerpt {i} | source={e['source_id']}]\n{e['text'][:800]}"
        for i, e in enumerate(excerpts)
    )


def _cite_footer_from_rag(answer: str, excerpts: list[dict]) -> tuple[str, bool, list[int]]:
    """Parse ``[excerpt N]`` markers from a rag_specific answer and
    build a deterministic provenance footer listing the cited
    sources + the verbatim text snippet from each. Returns
    ``(footer, grounded_bool, cited_indices)``.

    ``grounded`` is True iff at least one citation resolves to a
    valid excerpt index. Invalid indices are silently dropped so a
    model that writes ``[excerpt 99]`` does not hallucinate a
    cite.
    """
    cited: list[int] = []
    for m in _EXCERPT_CITE_RE.finditer(answer):
        for part in re.split(r"[,|]", m.group(1)):
            part = part.strip()
            if not part.isdigit():
                continue
            idx = int(part)
            if 0 <= idx < len(excerpts) and idx not in cited:
                cited.append(idx)
    if not cited:
        return ("\n\n>> [no excerpt citations in answer]", False, [])

    lines = [f"\n\n>> [rag_specific grounded in {len(cited)} excerpt(s)]"]
    for idx in cited:
        e = excerpts[idx]
        src = e.get("source_id", "memory")
        snippet = (e.get("text") or "").strip().replace("\n", " ")
        snippet = (snippet[:160] + "...") if len(snippet) > 160 else snippet
        lines.append(f">>   [excerpt {idx} | source={src}]")
        lines.append(f">>     {snippet!r}")
    return ("\n".join(lines), True, cited)


def run_rag(mem: vstash.Memory, question: str) -> dict:
    """Naive RAG: same dual-3 retrieval Mode A uses, but the
    Builder sees the chunks directly and produces the final
    answer. No Judge, no claim-level. This is what people
    usually mean by "RAG".
    """
    t0 = time.perf_counter()
    excerpts = retrieve(mem, question, top_k=TOP_K, retrieval_mode="dual")
    stats = retrieval_stats(excerpts)
    # Truncate each excerpt so the RAG prompt stays within the
    # Builder's effective context. Same cap Mode A uses in its
    # draft step (ballpark).
    joined = "\n\n---\n\n".join(
        f"[source={e['source_id']}]\n{e['text'][:800]}"
        for e in excerpts
    )
    user = f"Context:\n{joined}\n\nQuestion: {question}"
    answer, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": RAG_BUILDER_SYSTEM},
            {"role": "user", "content": user},
        ],
        MAX_TOKENS_DRAFT,
    )
    return {
        "answer": answer,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


def run_rag_specific(mem: vstash.Memory, question: str) -> dict:
    """RAG with a prompt engineered for the three failure modes
    seen in the Run 1 RAG baseline: aggregation math, knowledge-
    update recency, and over-confident hallucination when context
    is ambiguous. Same retrieval as RAG; only the system prompt
    differs. Excerpt headers use the ``[excerpt N | source=X]``
    format so the model's ``[excerpt N]`` citations are index-
    stable for the ``_cite`` variant's deterministic footer.
    """
    t0 = time.perf_counter()
    excerpts = retrieve(mem, question, top_k=TOP_K, retrieval_mode="dual")
    stats = retrieval_stats(excerpts)
    user = f"Context:\n{_render_excerpt_block(excerpts)}\n\nQuestion: {question}"
    answer, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": RAG_SPECIFIC_SYSTEM},
            {"role": "user", "content": user},
        ],
        MAX_TOKENS_DRAFT,
    )
    return {
        "answer": answer,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


def attach_cite_footer(r_rag_specific: dict) -> dict:
    """Add a deterministic provenance footer to an existing
    ``rag_specific`` result. Returns a NEW dict (does not mutate
    the input) so the oracle sees the same underlying answer + a
    footer; we are isolating the footer's effect rather than
    re-running the Builder.

    This is the key test: if the Builder cited excerpts and the
    footer resolves them to source_id + verbatim snippet
    deterministically, we get Mode A's grounded property without
    Mode A's Judge call.
    """
    answer = r_rag_specific["answer"]
    excerpts = r_rag_specific["retrieved"]
    footer, grounded, cited_ids = _cite_footer_from_rag(answer, excerpts)
    out = dict(r_rag_specific)
    out["answer"] = answer + footer
    out["grounded"] = grounded
    out["cited_excerpt_ids"] = cited_ids
    # Token / wall numbers are identical -- the footer is free.
    return out


def run_mode_a(mem: vstash.Memory, question: str) -> dict:
    """Full Mode A v4. Replicates cerebras_midloop.run_one but
    returns the raw components so we can build a uniform audit row
    across conditions. Using the personal domain frame because
    LongMemEval haystacks are conversational project memory, not
    clinical guidelines.
    """
    t0 = time.perf_counter()

    # Builder draft (confident mode, matches the smoke runs).
    b_sys = BUILDER_MODES["confident"]
    draft, b_dt, b_usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": b_sys},
            {"role": "user", "content": question},
        ],
        MAX_TOKENS_DRAFT,
    )

    # Retrieval (3-way dual by default).
    query = f"{question}\n{draft[:400]}"
    excerpts = retrieve(mem, query, top_k=TOP_K, retrieval_mode="dual")
    stats = retrieval_stats(excerpts)

    # Point the Judge at the personal domain frame for this eval,
    # then RESTORE whatever the original module-level JUDGE_SYSTEM
    # was after the call. Previously we left the global mutated,
    # which could silently change the Judge prompt for any other
    # process importing cerebras_midloop after this eval. The
    # module is single-threaded-safe but this eliminates the
    # cross-test contamination failure mode flagged by bot review.
    _prior = _mod.JUDGE_SYSTEM
    try:
        _mod.JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(
            domain_frame=DOMAIN_FRAMES["personal"]
        )
        judgment, j_dt, j_usage = judge_once(question, draft, excerpts)
    finally:
        _mod.JUDGE_SYSTEM = _prior

    final = annotate_deterministic(draft, judgment, excerpts)

    claims = judgment.get("claims") or []
    bad_claims = [
        c for c in claims
        if isinstance(c, dict) and c.get("verdict") in ("contradicts", "neutral")
    ]
    total_tokens = (b_usage.get("total_tokens", 0) or 0) + (j_usage.get("total_tokens", 0) or 0)
    return {
        "answer": final,
        "draft": draft,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "judgment": judgment,
        "builder_usage": b_usage,
        "judge_usage": j_usage,
        "draft_s": b_dt,
        "judge_s": j_dt,
        "total_tokens": total_tokens,
        "n_sub_claims": len(claims),
        "n_sub_claims_unsupported": len(bad_claims),
        "_total_s": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------- grid


# Prompts addressable by a short key in a grid config. New entries
# go here; the grid YAML references them by name.
SYSTEM_PROMPT_REGISTRY: dict[str, str] = {
    "confident": BUILDER_MODES["confident"] or "",
    "rag_baseline": RAG_BUILDER_SYSTEM,
    "rag_specific": RAG_SPECIFIC_SYSTEM,
}


@dataclass
class Condition:
    """One row of a grid run. ``kind`` picks the factory; the rest
    are kind-specific knobs. Unused knobs for a kind are silently
    ignored so a shared YAML schema can describe every condition
    without a discriminated union.
    """

    name: str
    kind: str  # 'control' | 'rag' | 'rag_jev' | 'mode_a'
    temperature: float = 0.3
    top_k: int = TOP_K
    retrieval_mode: str = "dual"
    system_prompt: str = "rag_baseline"   # key in SYSTEM_PROMPT_REGISTRY
    excerpt_truncation: int = 800          # chars per excerpt in the prompt
    max_tokens: int = MAX_TOKENS_DRAFT
    cite_footer: bool = False              # append deterministic footer?
    extra: dict = field(default_factory=dict)


def _resolve_system_prompt(key: str) -> str:
    if key not in SYSTEM_PROMPT_REGISTRY:
        raise KeyError(
            f"unknown system_prompt key {key!r}; known: "
            f"{sorted(SYSTEM_PROMPT_REGISTRY)}"
        )
    return SYSTEM_PROMPT_REGISTRY[key]


def _run_control_cfg(mem: vstash.Memory, question: str, c: Condition) -> dict:
    t0 = time.perf_counter()
    draft, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": _resolve_system_prompt(c.system_prompt)},
            {"role": "user", "content": question},
        ],
        c.max_tokens,
        temperature=c.temperature,
    )
    return {
        "answer": draft,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }


def _run_rag_cfg(mem: vstash.Memory, question: str, c: Condition) -> dict:
    t0 = time.perf_counter()
    excerpts = retrieve(mem, question, top_k=c.top_k, retrieval_mode=c.retrieval_mode)
    stats = retrieval_stats(excerpts)
    joined = "\n\n---\n\n".join(
        f"[excerpt {i} | source={e['source_id']}]\n{e['text'][:c.excerpt_truncation]}"
        for i, e in enumerate(excerpts)
    )
    user = f"Context:\n{joined}\n\nQuestion: {question}"
    answer, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": _resolve_system_prompt(c.system_prompt)},
            {"role": "user", "content": user},
        ],
        c.max_tokens,
        temperature=c.temperature,
    )
    result: dict = {
        "answer": answer,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "wall_s": dt,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }
    if c.cite_footer:
        footer, grounded, cited_ids = _cite_footer_from_rag(answer, excerpts)
        result["answer"] = answer + footer
        result["grounded"] = grounded
        result["cited_excerpt_ids"] = cited_ids
    return result


def _run_rag_jev_cfg(mem: vstash.Memory, question: str, c: Condition) -> dict:
    """RAG with Jev on the recall path. Over-fetches the dual pool
    (``extra.pool_k`` per search, so up to ``8 * pool_k`` candidates),
    lets ``merken.classifiers.jev.JevReranker`` keep the excerpts that
    answer the question and put the current state first, then hands
    the top ``c.top_k`` to the same Builder prompt as ``rag``.
    """
    from types import SimpleNamespace

    t0 = time.perf_counter()
    pool_k = int(c.extra.get("pool_k", 10))
    pool = retrieve(mem, question, top_k=pool_k, retrieval_mode=c.retrieval_mode)
    reranker = _jev_reranker_singleton(c)
    # Jev sees the SESSION date with each excerpt (from the dataset, not
    # vstash's added_at, which is ingestion time): on LongMemEval the
    # supersession signal for knowledge-update questions lives in the
    # timestamps, not in the text ("Reverted…" is rare in chat logs).
    conv = _ACTIVE_CONV
    dates = conv.session_dates if conv is not None else {}

    def _dated(e: dict) -> str:
        parts = (e.get("source_id") or "").split("::")
        ts = dates.get(parts[1]) if len(parts) >= 2 else None
        return f"[session {ts}] {e['text']}" if ts else e["text"]

    wrapped = [SimpleNamespace(text=_dated(e), path=e.get("chunk_id"), _e=e) for e in pool]
    t_rr = time.perf_counter()
    kept = reranker.rerank(question, wrapped)
    rerank_s = time.perf_counter() - t_rr
    excerpts = [w._e for w in kept][: c.top_k]
    stats = retrieval_stats(excerpts)
    joined = "\n\n---\n\n".join(
        f"[excerpt {i} | source={e['source_id']}]\n{e['text'][:c.excerpt_truncation]}"
        for i, e in enumerate(excerpts)
    )
    user = f"Context:\n{joined}\n\nQuestion: {question}"
    answer, dt, usage = cerebras_chat(
        BUILDER,
        [
            {"role": "system", "content": _resolve_system_prompt(c.system_prompt)},
            {"role": "user", "content": user},
        ],
        c.max_tokens,
        temperature=c.temperature,
    )
    result: dict = {
        "answer": answer,
        "retrieved": excerpts,
        "retrieval_stats": stats,
        "jev": {"pool": len(pool), "kept": len(kept), "rerank_s": rerank_s},
        "wall_s": dt + rerank_s,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "builder_usage": usage,
        "_total_s": time.perf_counter() - t0,
    }
    if c.cite_footer:
        footer, grounded, cited_ids = _cite_footer_from_rag(answer, excerpts)
        result["answer"] = answer + footer
        result["grounded"] = grounded
        result["cited_excerpt_ids"] = cited_ids
    return result


_JEV_RERANKERS: dict = {}


def _jev_reranker_singleton(c: Condition):
    from merken.classifiers.jev import JevReranker

    conv = _ACTIVE_CONV
    qdate = (conv.question_date if conv is not None else None) or "unknown"
    time_aware = bool(c.extra.get("time_aware", False))
    key = (
        float(c.extra.get("min_answer", 0.5)),
        bool(c.extra.get("pick_current", True)),
        int(c.extra.get("min_keep", c.top_k)),
        time_aware,
        qdate if time_aware else "",
    )
    if key not in _JEV_RERANKERS:
        instr = None
        if time_aware:
            instr = (
                f"The question is asked on {qdate}. Each text is prefixed with its session "
                "date. Which text answers the question FOR THE TIME IT REFERS TO? If the "
                "question is about the present ('currently', 'now', no time cue), prefer the "
                "most recent session that answers it; if it refers to an earlier moment "
                "('when I first started', 'two weeks ago', 'originally'), prefer the session "
                'from that time. Question: "{query}"'
            )
        _JEV_RERANKERS[key] = JevReranker(
            min_answer=key[0], pick_current=key[1], min_keep=key[2],
            current_instructions=instr, workers=8,
        )
    return _JEV_RERANKERS[key]


_ACTIVE_CONV = None  # set by the eval loops so rerankers can see session dates


def _run_mode_a_cfg(mem: vstash.Memory, question: str, c: Condition) -> dict:
    # Mode A's internal calls still read temperature from the global
    # default (0.3) for now. Threading temperature into run_mode_a
    # is a follow-up; for the grid's first pass we test temperature
    # on RAG variants, which is where the production shape lives.
    _ = c  # kind="mode_a" uses the existing hard-coded pipeline
    return run_mode_a(mem, question)


CONDITION_KINDS: dict[str, Callable[[vstash.Memory, str, Condition], dict]] = {
    "control": _run_control_cfg,
    "rag": _run_rag_cfg,
    "rag_jev": _run_rag_jev_cfg,
    "mode_a": _run_mode_a_cfg,
}


def run_condition(mem: vstash.Memory, question: str, c: Condition) -> dict:
    if c.kind not in CONDITION_KINDS:
        raise KeyError(
            f"unknown condition kind {c.kind!r}; known: {sorted(CONDITION_KINDS)}"
        )
    return CONDITION_KINDS[c.kind](mem, question, c)


def load_grid(path: Path) -> list[Condition]:
    """Load a YAML grid spec. Schema:

    ``conditions:`` list of dicts, each with ``name`` and ``kind``
    plus optional overrides. Unknown keys become ``extra``.
    """
    import yaml  # lazy -- only needed in grid mode

    data = yaml.safe_load(path.read_text())
    raw = data.get("conditions") or data  # tolerate bare list
    out: list[Condition] = []
    known = {f.name for f in Condition.__dataclass_fields__.values()}
    for row in raw:
        kwargs = {k: v for k, v in row.items() if k in known}
        extra = {k: v for k, v in row.items() if k not in known}
        if extra:
            kwargs["extra"] = extra
        out.append(Condition(**kwargs))
    return out


# --------------------------------------------------------------------- eval loop


@dataclass
class EvalConfig:
    subset: str
    n: int
    seed: int
    out_path: Path
    keep_dbs: bool = False
    # In legacy mode top_k is hardcoded to TOP_K (5) in each run_*
    # helper. In grid mode each Condition carries its own top_k.
    # A top-level cfg.top_k would be misleading (the runs ignore it)
    # so it is intentionally not exposed here.
    grid: list[Condition] | None = None  # None => legacy 5-condition mode


def run_eval(cfg: EvalConfig) -> list[dict]:
    print(f"[config] subset={cfg.subset} n={cfg.n} seed={cfg.seed} out={cfg.out_path}")
    conversations = load_longmemeval(subset=cfg.subset)
    print(f"[dataset] loaded {len(conversations)} conversations")

    rnd = random.Random(cfg.seed)
    sampled = rnd.sample(conversations, min(cfg.n, len(conversations)))
    print(f"[dataset] sampled {len(sampled)} for eval")

    oracle = _oracle_client()
    rows: list[dict] = []

    tmp_root = Path.home() / ".merken" / f"longmemeval_mode_a_{cfg.seed}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = cfg.out_path.open("w")

    try:
        for i, conv in enumerate(sampled):
            # Some LongMemEval answers are numbers (int/float). The
            # Conversation dataclass annotates ``answer: str`` but the
            # loader just passes through whatever is in the JSON.
            # Coerce here so both the debug print and the oracle prompt
            # (via ground_truth[:2000]) work uniformly. The coerced
            # string is what ships to the oracle and into the audit,
            # so the comparison is faithful.
            gt_text = str(conv.answer) if conv.answer is not None else ""
            print(
                f"\n{'='*72}\n[{i+1}/{len(sampled)}] qid={conv.question_id} "
                f"type={conv.question_type}\n{'='*72}"
            )
            print(f"Q: {conv.question[:200]}")
            print(f"GT: {gt_text[:200]}")

            # Fresh DB per question so haystacks don't pollute each
            # other. Project tag is the qid so vstash internal
            # filtering stays predictable.
            global _ACTIVE_CONV
            _ACTIVE_CONV = conv
            db_path = tmp_root / f"{conv.question_id}.db"
            if db_path.exists():
                db_path.unlink()
            mem = vstash.Memory(
                db=str(db_path),
                project=conv.question_id,
                collection="default",
            )
            # Wrap per-question body so a single Cerebras 5xx /
            # Gemini hiccup / ingest crash does NOT abort the
            # remaining rows after we've already paid oracle budget
            # on earlier ones. A failed question writes a stub row
            # with the error so the summary can still account for
            # it and a resume-from-log is possible.
            try:
                t_ingest = time.perf_counter()
                n_ingested = _ingest(mem, conv)
                ingest_s = time.perf_counter() - t_ingest
                print(f"[ingest] {n_ingested} turns in {ingest_s:.1f}s")

                # Five conditions, same question, same mem for the
                # four that need retrieval.
                print("[control]")
                r_control = run_control(conv.question)
                print(f"  answer: {r_control['answer'][:180]!r}")

                print("[rag]")
                r_rag = run_rag(mem, conv.question)
                print(f"  answer: {r_rag['answer'][:180]!r}")

                print("[rag_specific]")
                r_rag_specific = run_rag_specific(mem, conv.question)
                print(f"  answer: {r_rag_specific['answer'][:180]!r}")

                # Same answer, same excerpts, footer appended
                # deterministically -- no new LLM call.
                print("[rag_specific_cite]")
                r_rag_cite = attach_cite_footer(r_rag_specific)
                n_cites = len(r_rag_cite.get("cited_excerpt_ids") or [])
                print(
                    f"  grounded={r_rag_cite.get('grounded')} "
                    f"cites={n_cites}"
                )

                print("[mode_a]")
                r_mode_a = run_mode_a(mem, conv.question)
                v = r_mode_a["judgment"].get("verdict")
                n_claims = r_mode_a["n_sub_claims"]
                n_bad = r_mode_a["n_sub_claims_unsupported"]
                print(
                    f"  verdict={v} sub_claims={n_claims} unsupported={n_bad}"
                )

                # Oracle each answer against the ground truth.
                print("[oracle]")
                o_control = oracle_score(
                    oracle, conv.question, gt_text, r_control["answer"]
                )
                o_rag = oracle_score(
                    oracle, conv.question, gt_text, r_rag["answer"]
                )
                o_rag_specific = oracle_score(
                    oracle, conv.question, gt_text, r_rag_specific["answer"]
                )
                o_rag_cite = oracle_score(
                    oracle, conv.question, gt_text, r_rag_cite["answer"]
                )
                o_mode_a = oracle_score(
                    oracle, conv.question, gt_text, r_mode_a["answer"]
                )
                print(
                    f"  control={o_control['verdict']:10s} "
                    f"rag={o_rag['verdict']:10s} "
                    f"rag_spec={o_rag_specific['verdict']:10s} "
                    f"rag_cite={o_rag_cite['verdict']:10s} "
                    f"mode_a={o_mode_a['verdict']:10s}"
                )

                audit = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "answer_session_ids": conv.answer_session_ids,
                    "n_sessions_ingested": n_ingested,
                    "ingest_s": ingest_s,
                    "conditions": {
                        "control": {**r_control, "oracle": o_control},
                        "rag": {**r_rag, "oracle": o_rag},
                        "rag_specific": {**r_rag_specific, "oracle": o_rag_specific},
                        "rag_specific_cite": {**r_rag_cite, "oracle": o_rag_cite},
                        "mode_a": {**r_mode_a, "oracle": o_mode_a},
                    },
                }
                fout.write(json.dumps(audit, default=str) + "\n")
                fout.flush()
                rows.append(audit)
            except Exception as exc:  # noqa: BLE001 -- deliberate catch-all
                print(f"[error] question {conv.question_id} failed: {exc!r}")
                err_row = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "error": repr(exc),
                }
                fout.write(json.dumps(err_row, default=str) + "\n")
                fout.flush()
                # Do NOT append to rows -- the summary counts
                # completed questions only. The stub row in the log
                # is enough for later debugging.
            finally:
                mem.close()
                if not cfg.keep_dbs and db_path.exists():
                    db_path.unlink()
    finally:
        fout.close()

    return rows


def run_grid_eval(cfg: EvalConfig) -> list[dict]:
    """Grid-mode eval. Same per-question loop as ``run_eval`` but
    each condition is parameter-driven (top_k, temperature,
    system_prompt, etc.). Ingestion is shared per question so
    adding another condition is only the cost of one more Builder
    (and optionally Judge) call per question.
    """
    assert cfg.grid, "run_grid_eval called with no grid config"
    print(
        f"[grid] subset={cfg.subset} n={cfg.n} seed={cfg.seed} "
        f"conditions={len(cfg.grid)} out={cfg.out_path}"
    )
    for c in cfg.grid:
        print(
            f"  - {c.name:30s} kind={c.kind:8s} "
            f"temp={c.temperature} top_k={c.top_k} "
            f"sys={c.system_prompt} cite={c.cite_footer}"
        )
    conversations = load_longmemeval(subset=cfg.subset)
    rnd = random.Random(cfg.seed)
    sampled = rnd.sample(conversations, min(cfg.n, len(conversations)))
    oracle = _oracle_client()

    tmp_root = Path.home() / ".merken" / f"longmemeval_mode_a_{cfg.seed}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = cfg.out_path.open("w")
    rows: list[dict] = []

    try:
        for i, conv in enumerate(sampled):
            gt_text = str(conv.answer) if conv.answer is not None else ""
            print(
                f"\n{'='*72}\n[{i+1}/{len(sampled)}] qid={conv.question_id} "
                f"type={conv.question_type}\n{'='*72}"
            )
            print(f"Q: {conv.question[:200]}")
            print(f"GT: {gt_text[:200]}")

            global _ACTIVE_CONV
            _ACTIVE_CONV = conv
            db_path = tmp_root / f"{conv.question_id}.db"
            if db_path.exists():
                db_path.unlink()
            mem = vstash.Memory(
                db=str(db_path),
                project=conv.question_id,
                collection="default",
            )
            try:
                t_ingest = time.perf_counter()
                n_ingested = _ingest(mem, conv)
                ingest_s = time.perf_counter() - t_ingest
                print(f"[ingest] {n_ingested} turns in {ingest_s:.1f}s")

                per_cond: dict[str, Any] = {}
                for c in cfg.grid:
                    print(f"[{c.name}]")
                    try:
                        r = run_condition(mem, conv.question, c)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  error: {exc!r}")
                        per_cond[c.name] = {"error": repr(exc)}
                        continue
                    o = oracle_score(
                        oracle, conv.question, gt_text, r["answer"]
                    )
                    r["oracle"] = o
                    r["_condition_spec"] = {
                        "kind": c.kind,
                        "temperature": c.temperature,
                        "top_k": c.top_k,
                        "retrieval_mode": c.retrieval_mode,
                        "system_prompt": c.system_prompt,
                        "cite_footer": c.cite_footer,
                        "max_tokens": c.max_tokens,
                    }
                    per_cond[c.name] = r
                    print(
                        f"  oracle={o.get('verdict'):10s} "
                        f"tok={r.get('total_tokens',0)} "
                        f"wall={r.get('_total_s', 0):.1f}s"
                    )

                audit = {
                    "audit_id": uuid.uuid4().hex[:12],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "question_id": conv.question_id,
                    "question_type": conv.question_type,
                    "question": conv.question,
                    "ground_truth": gt_text,
                    "answer_session_ids": conv.answer_session_ids,
                    "n_sessions_ingested": n_ingested,
                    "ingest_s": ingest_s,
                    "conditions": per_cond,
                }
                fout.write(json.dumps(audit, default=str) + "\n")
                fout.flush()
                rows.append(audit)
            except Exception as exc:  # noqa: BLE001
                print(f"[error] qid={conv.question_id}: {exc!r}")
                fout.write(json.dumps(
                    {
                        "audit_id": uuid.uuid4().hex[:12],
                        "question_id": conv.question_id,
                        "error": repr(exc),
                    },
                    default=str,
                ) + "\n")
                fout.flush()
            finally:
                mem.close()
                if not cfg.keep_dbs and db_path.exists():
                    db_path.unlink()
    finally:
        fout.close()

    return rows


# --------------------------------------------------------------------- summary


def _correct(v: str) -> bool:
    return v in ("supports", "partial")


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("no rows -- nothing to summarise")
        return

    # Legacy 5-condition layout for backward compat. Grid runs
    # summarise differently -- see summarize_grid below.
    first = rows[0].get("conditions", {})
    conditions = list(first) if first else [
        "control", "rag", "rag_specific", "rag_specific_cite", "mode_a"
    ]
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  n questions: {n}")
    print()
    print(f"  {'condition':10s}  "
          f"{'correct':>8s}  {'supports':>9s}  {'partial':>8s}  "
          f"{'contradicts':>12s}  {'neutral':>8s}  "
          f"{'avg_tok':>8s}  {'avg_s':>6s}")
    for cond in conditions:
        verdicts = [r["conditions"][cond]["oracle"]["verdict"] for r in rows]
        tok = [int(r["conditions"][cond].get("total_tokens") or 0) for r in rows]
        wall = [float(r["conditions"][cond].get("_total_s") or 0) for r in rows]
        supports = verdicts.count("supports")
        partial = verdicts.count("partial")
        contradicts = verdicts.count("contradicts")
        neutral = verdicts.count("neutral")
        correct = supports + partial
        print(
            f"  {cond:10s}  "
            f"{correct/n*100:6.1f}%  {supports:>9d}  {partial:>8d}  "
            f"{contradicts:>12d}  {neutral:>8d}  "
            f"{sum(tok)//n:>8d}  {sum(wall)/n:>5.1f}s"
        )

    print()
    # Mode A telemetry is only meaningful when the legacy
    # condition key "mode_a" is present in the rows.
    if rows and "mode_a" in rows[0].get("conditions", {}):
        mode_a_rows = [r["conditions"]["mode_a"] for r in rows]
        grounded = sum(
            1 for r in mode_a_rows
            if (r.get("judgment", {}).get("quoted_evidence") or "").strip()
        )
        with_bad = sum(
            1 for r in mode_a_rows
            if (r.get("n_sub_claims_unsupported") or 0) > 0
        )
        print(
            f"  mode_a grounded (has quoted_evidence): "
            f"{grounded}/{n} ({grounded/n*100:.1f}%)"
        )
        print(
            f"  mode_a claim-level leaks (>=1 unsupported sub-claim): "
            f"{with_bad}/{n} ({with_bad/n*100:.1f}%)"
        )


# --------------------------------------------------------------------- cli


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="longmemeval_s",
                   help="LongMemEval subset (default longmemeval_s)")
    p.add_argument("--n", type=int, default=3,
                   help="number of questions to sample (default 3 for preview)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/retrieval/longmemeval/mode_a_eval_runs"),
        help="directory for audit logs; filename includes n+seed",
    )
    p.add_argument(
        "--keep-dbs", action="store_true",
        help="keep per-question vstash dbs after the run (default: cleanup)",
    )
    p.add_argument(
        "--grid",
        type=Path,
        default=None,
        help=(
            "path to a YAML grid spec. When set, runs a configurable "
            "multi-condition eval instead of the legacy 5-condition "
            "layout. See grids/ for example specs."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.grid:
        grid = load_grid(args.grid)
        tag = args.grid.stem
        out_path = out_dir / f"grid-{tag}_n{args.n}_seed{args.seed}.jsonl"
    else:
        grid = None
        out_path = out_dir / f"n{args.n}_seed{args.seed}.jsonl"
    cfg = EvalConfig(
        subset=args.subset,
        n=args.n,
        seed=args.seed,
        out_path=out_path,
        keep_dbs=args.keep_dbs,
        grid=grid,
    )
    rows = run_grid_eval(cfg) if grid else run_eval(cfg)
    summarize(rows)
    print(f"\nlog: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
