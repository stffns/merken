"""Mode A concept smoke test -- Cerebras + vstash.

Proves the midloop pattern end-to-end without training anything.
The Builder is a small Cerebras model (``llama3.1-8b``) simulating
a CHW-edge assistant that hallucinates or under-specifies clinical
claims. The midloop components are LLM-as-judge calls to a bigger
Cerebras model (``qwen-3-235b-a22b-instruct-2507``):

  1. ClaimDetector (LLM call): does the draft contain a verifiable
     medical claim? If yes, what queries should we run against
     memory?
  2. snapvec.search over a vstash seeded with 517 WHO/ICRC
     protocol chunks.
  3. ClaimVerifier (LLM call): does the draft match the retrieved
     guidelines? If contradicts, what do the guidelines say?
  4. Regenerate (Builder call): re-ask the Builder with a system
     whisper injected -- "note per authoritative guidelines, X".

No training. No local model. Runs in ~30 seconds per question on
Cerebras' fabric. If this proves the concept (at least one baseline
hallucination gets corrected via retrieval), we unlock either Mode
C (local small-model detector embedded in the generation loop) or
a proper Mode A A/B eval on a larger set.

Usage:
  python cerebras_midloop.py seed [chunks.jsonl ...]
  python cerebras_midloop.py run  [--question "..." | --all]

The default ``--all`` runs a hardcoded 5-question smoke set chosen
to hit known-answers in the 517-chunk corpus. Output is a side-by-
side report to stdout + a structured JSON log at
``experiments/midloop_concept/medlocal/cerebras_smoke.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Models -- keep the asymmetry small-builder / big-judge to mirror
# the production midloop topology (Builder is the runtime model,
# Judge is the more capable sidecar).
# llama3.1-8b was retired from Cerebras (404 as of 2026-09). gpt-oss-120b is
# the canonical Builder since the 2026-04-27 3-seed runs (CHANGELOG); override
# with MERKEN_LME_BUILDER to reproduce older rows.
BUILDER = os.environ.get("MERKEN_LME_BUILDER", "gpt-oss-120b")
JUDGE = "qwen-3-235b-a22b-instruct-2507"
# Token budgets kept tight -- this pipeline is deliberately thrifty.
# Draft caps where a clinical answer comfortably fits; Judge caps
# where the verdict + quoted evidence + optional rewrite fit.
MAX_TOKENS_DRAFT = 300
# Bumped 2026-04-21 from 600 to 1200 when the Judge schema grew
# a `claims` array. 3-8 sub-claim entries at ~60 tokens each plus
# the original fields fit comfortably under 1200 with headroom.
MAX_TOKENS_JUDGE = 1200

DB_PATH = str(Path.home() / ".merken" / "medlocal_concept.db")
PROJECT = "medlocal_concept"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS = [
    REPO_ROOT / "experiments/midloop_pilot/scaleup_out_hf_v1bfrozen/protocols.jsonl",
    REPO_ROOT / "experiments/midloop_pilot/scaleup_out/protocols.jsonl",
]

SMOKE_QUESTIONS = [
    (
        "severe_dehydration_child",
        "A 3-year-old child arrives with sunken eyes, skin pinch "
        "returns very slowly, and lethargy. What IV fluids should "
        "I start and at what dose/rate?",
    ),
    (
        "severe_pneumonia_infant",
        "A 2-month-old infant has severe pneumonia (chest indrawing, "
        "oxygen saturation 89%). What antibiotic should I give and "
        "at what dose?",
    ),
    (
        "postpartum_hemorrhage_txa",
        "A woman with postpartum hemorrhage did not respond to "
        "uterotonics. What is the tranexamic acid dose and how "
        "should I administer it?",
    ),
    (
        "cotrimoxazole_hiv_adult",
        "An adult patient was just diagnosed with HIV and has a "
        "CD4 count of 180. Should I start co-trimoxazole "
        "prophylaxis, and if so, when?",
    ),
    (
        "blood_donor_screening",
        "A potential blood donor appears pale, has a persistent "
        "cough, and a tattoo from last month. Should I accept them?",
    ),
]

# Personal-memory smoke set -- questions about Jay's own project history.
# All five have verbatim answers sitting in the engram vstash. A
# Builder model with no access to that memory will either hallucinate
# plausibly or refuse; either case exercises the Mode A grounding
# path. Domain is deliberately non-clinical so we also test the
# Judge prompt's generalisation beyond WHO guidelines.
PERSONAL_SMOKE_QUESTIONS = [
    (
        "write_filter_baseline",
        "In the user's merken project, what version of the "
        "write-filter classifier is the current graduated "
        "baseline, and what was its key metric improvement "
        "over the previous version?",
    ),
    (
        "consolidation_threshold",
        "What cosine similarity threshold does merken's "
        "PeriodicConsolidator use for clustering, and why was "
        "that specific value chosen?",
    ),
    (
        "silt_rule",
        "What is Silt's rule about proposing new algorithms in "
        "the merken / engram project?",
    ),
    (
        "engram_longmemeval_r5",
        "In the engram project's LongMemEval benchmark at "
        "n=500, what R@5 score did engram achieve?",
    ),
    (
        "four_primitives",
        "What are the four decision primitives defined in "
        "merken's CONSTITUTION, and what are the default "
        "implementations for each?",
    ),
]

# --------------------------------------------------------------- cerebras

def _cerebras_client():
    # Lazy import so `--help` works without the SDK.
    from cerebras.cloud.sdk import Cerebras
    return Cerebras()


def cerebras_chat(
    model: str,
    messages: list[dict],
    max_tokens: int,
    *,
    temperature: float = 0.3,
) -> tuple[str, float, dict]:
    """Return (text, wall_seconds, usage). Bubbles SDK exceptions up
    after a short retry window.

    Cerebras intermittently returns 5xx ``queue_exceeded`` / 429
    rate-limit under load. A smoke run that fires 2 calls per
    question times N questions has a non-trivial probability of
    hitting one. Retry the transient classes up to 4 total
    attempts with exponential backoff of 2s / 4s / 8s between
    attempts (so attempts 1..4 are at t=0, t=2, t=6, t=14).
    Non-retryable errors (auth, bad request, not found) bubble
    up immediately.

    We catch ``APIStatusError`` (the common parent of every HTTP
    error the SDK raises) rather than a specific subclass so a
    future SDK version that reclassifies 503 as
    ``ServiceUnavailableError`` does not silently break the
    retry. Status-code gating via ``response.status_code`` keeps
    4xx client errors out of the retry loop.

    ``usage`` is a dict of prompt_tokens / completion_tokens /
    total_tokens taken from ``resp.usage`` when Cerebras populates
    it. Empty dict when the SDK response omits the field, so the
    caller can sum defensively.
    """
    from cerebras.cloud.sdk import APIStatusError

    client = _cerebras_client()
    t0 = time.perf_counter()
    # Attempts 1..4; between each pair we sleep backoffs[i] seconds.
    # Three backoffs => four attempts total (matches the docstring).
    backoffs = [2, 4, 8]
    for attempt in range(len(backoffs) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            break
        except APIStatusError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            # Only transient classes retry: 5xx (server) and 429
            # (rate-limit). 4xx client errors (400/401/403/404/422
            # etc) are bugs on our side; retrying would just waste
            # budget and hide the real error.
            is_transient = isinstance(status, int) and (
                status >= 500 or status == 429
            )
            if not is_transient or attempt == len(backoffs):
                raise
            backoff = backoffs[attempt]
            print(
                f"    cerebras {status} (attempt {attempt+1}/"
                f"{len(backoffs)+1}), backing off {backoff}s: {exc}"
            )
            time.sleep(backoff)
    dt = time.perf_counter() - t0
    usage = {}
    u = getattr(resp, "usage", None)
    if u is not None:
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
    # Defensive: the Cerebras SDK has always populated `choices`
    # in observed runs, but treat an empty list as an empty draft
    # rather than letting an IndexError crash the smoke loop.
    choices = getattr(resp, "choices", None) or []
    content = choices[0].message.content if choices else ""
    return (content or "").strip(), dt, usage


def _extract_json_object(raw: str) -> dict:
    """Tolerant JSON extract -- mirrors case_generator's approach.

    Handles ``{...}`` objects that may be wrapped in markdown fences
    or preceded by prose. Raises ValueError on total failure so the
    caller can fall back rather than crash the pipeline.
    """
    text = raw.strip()
    m = re.match(r"```(?:json)?\s*", text)
    if m:
        text = text[m.end():]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Locate outermost {...}. Using rfind handles cases where the
    # model appends an explanation object after the answer object;
    # we take the first valid-parse attempt.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM response: {raw[:200]!r}")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}; candidate={candidate[:200]!r}") from e


# --------------------------------------------------------------- midloop stages

# Judge prompt template. The ``{domain_frame}`` placeholder is the
# only per-domain variance -- everything below it (JSON schema,
# verdict rules, verbatim-evidence rule) is invariant across domains.
# Keeping one prompt body avoids drift: a fix to the HARD RULES
# applies to both clinical and personal runs.
JUDGE_SYSTEM_TEMPLATE = (
    "{domain_frame}\n\n"
    "Each excerpt header carries retrieval signals from the memory "
    "store: `score` is an RRF-family relevance score (higher = more "
    "relevant; typical range 0.005-0.030 on this corpus); `layer` "
    "is the memory layer tag (episodic = raw, brief/semantic = "
    "distilled). Prefer grounding your verdict in high-score and "
    "distilled-layer excerpts when multiple candidates touch the "
    "same claim. These are hints, not hard cutoffs.\n\n"
    "Your job in ONE JSON response:\n"
    "  1. decide the verdict of the draft against the excerpts.\n"
    "  2. if contradicts, produce the corrected answer using only "
    "     information grounded in the excerpts.\n"
    "  3. decompose the final rendered answer (the corrected_text "
    "     if verdict=contradicts, otherwise the draft) into atomic "
    "     sub-claims and verify EACH one against the excerpts.\n\n"
    "Respond ONLY with a JSON object:\n"
    "{{\n"
    '  "has_claim": bool,              // draft contains a verifiable factual claim?\n'
    '  "verdict": "supports"|"contradicts"|"neutral"|"no_claim",\n'
    '  "cited_excerpt_ids": [int, ...], // 0-based indices into excerpts\n'
    '  "quoted_evidence": "...",       // verbatim sentence(s) from the excerpts\n'
    '  "corrected_text": "...",        // required iff verdict=contradicts; empty otherwise\n'
    '  "claims": [                    // per sub-claim verification; see Rules below\n'
    "    {{\n"
    '      "text": "...",             // the atomic sub-claim as a short sentence\n'
    '      "verdict": "supports"|"contradicts"|"neutral",\n'
    '      "supporting_excerpt_id": int|null, // null only for neutral\n'
    '      "quoted_evidence": "..."   // verbatim substring from that excerpt; "" when neutral\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Rules (top-level verdict):\n"
    "- supports: some excerpt VERBATIM matches the draft's top-line "
    "answer. Fill cited_excerpt_ids + quoted_evidence. Leave "
    "corrected_text empty.\n"
    "- contradicts: some excerpt contradicts the draft. Fill all "
    "fields; corrected_text is the final user-facing answer.\n"
    "- neutral: excerpts do not cover the claim. Leave cited_excerpt_ids "
    "[] and quoted_evidence \"\".\n"
    "- no_claim: draft has no verifiable factual claim.\n\n"
    "Rules (claims array -- applies when has_claim=true):\n"
    "- Decompose the FINAL rendered answer into 3-8 atomic "
    "sub-claims. 'Final rendered answer' = corrected_text when "
    "verdict=contradicts, draft otherwise. **Do NOT include any "
    "statements that appear only in the draft when a "
    "corrected_text is present** -- the draft is the rejected "
    "version; only the corrected_text is user-facing.\n"
    "- Atomic means one specific fact per sub-claim (e.g. 'start "
    "co-trimoxazole prophylaxis' and 'CD4 threshold is 200' are "
    "two separate sub-claims).\n"
    "- For each sub-claim, emit verdict/supporting_excerpt_id/"
    "quoted_evidence using the SAME verbatim-substring rule as the "
    "top-level quoted_evidence.\n"
    "- If a sub-claim's verdict is 'supports', supporting_excerpt_id "
    "is the excerpt whose text grounds the claim and "
    "quoted_evidence is a verbatim substring of that excerpt.\n"
    "- If a sub-claim's verdict is 'contradicts', "
    "supporting_excerpt_id is the excerpt that contradicts the "
    "claim (NOT null -- future audit / training code needs the "
    "index to trace the evidence) and quoted_evidence is a "
    "verbatim substring of that excerpt.\n"
    "- If verdict is 'neutral' for a sub-claim, "
    "supporting_excerpt_id = null and quoted_evidence = \"\".\n"
    "- claims may be [] when has_claim=false.\n\n"
    "HARD RULES (enforce every time):\n"
    "- quoted_evidence (top-level AND per-claim) MUST be a substring "
    "of one excerpt. If you cannot find verbatim supporting evidence "
    "in the excerpts, verdict = 'neutral'. Do NOT synthesise evidence "
    "from your own general knowledge.\n"
    "- corrected_text MUST only contain claims grounded in "
    "quoted_evidence. Keep it to <= 120 words, numbered points ok.\n"
    "- The 'claims' array is the lever that makes answer-level "
    "verdicts honest. A 'supports' top-level verdict that ALSO has "
    "sub-claims with verdict=contradicts means the top-line is "
    "correct but the body drifts from the source -- emit that "
    "faithfully, do NOT hide the drift.\n"
    "- No prose outside the JSON. No markdown fences."
)

# Builder modes. The default ("auto") sends the user's question
# with no system prompt -- this is the realistic CHW-agent edge
# case where the Builder often refuses or hedges, so Mode A's
# grounding path mostly fires on "I cannot verify"-shaped drafts.
# "confident" mode forces the Builder to answer authoritatively
# with specific claims, exercising the Mode A hallucination-
# correction path instead. Used for Run 3 adversarial smoke
# (2026-04-21) after Run 2 revealed 4/5 personal drafts were
# refusals and only 1/5 was a real confabulation.
BUILDER_MODES = {
    "auto": None,
    "confident": (
        "You are answering a user question. Provide a direct, "
        "confident, specific answer. Do not hedge. Do not say "
        "'I am not sure', 'I cannot verify', 'I do not have "
        "information', or anything similar -- answer with "
        "concrete facts, numbers, names, and steps. If you are "
        "uncertain, commit to your best informed guess as if it "
        "were correct. Keep the answer under 200 words."
    ),
}

DOMAIN_FRAMES = {
    "clinical": (
        "You are the verification + rewriter stage of a retrieval-"
        "grounded assistant. The user asked a clinical question; a "
        "smaller Builder model produced a draft. You also see excerpts "
        "retrieved from an authoritative guideline memory (WHO/ICRC/"
        "MSF)."
    ),
    "personal": (
        "You are the verification + rewriter stage of a retrieval-"
        "grounded assistant. The user asked a question about their "
        "own project history (decisions, benchmark results, "
        "architecture notes); a smaller Builder model that has NO "
        "access to that history produced a draft. You also see "
        "excerpts retrieved from the user's authoritative project "
        "memory. Treat those excerpts as ground truth about the "
        "user's project -- the Builder's draft is almost certainly "
        "a hallucination unless the excerpts confirm it verbatim."
    ),
}

# Default Judge prompt keeps the clinical framing so existing
# callers (and the clinical smoke set) are unchanged. run_smoke
# overrides this per invocation based on --domain.
JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(domain_frame=DOMAIN_FRAMES["clinical"])


def judge_once(
    question: str, draft: str, excerpts: list[dict],
) -> tuple[dict, float, dict]:
    """One Judge call that performs detect + verify + rewrite in
    a single pass. Returns (parsed_json, wall_s, usage_dict).

    ``excerpts`` is a list of dicts (see ``retrieve``) carrying
    text plus the vstash signals (score, layer, ...). The Judge
    sees each excerpt rendered with a header that exposes those
    signals so a sub-claim anchored to a high-score excerpt is
    distinguishable from one anchored to a rank-20 low-score
    chunk. This is the "glass box" principle -- the Judge
    decides using information we never invented, only surfaced.
    """
    if not excerpts:
        # No retrieval -> no verification possible. Short-circuit
        # without a Judge call to save tokens.
        return (
            {
                "has_claim": False,
                "verdict": "no_retrieval",
                "cited_excerpt_ids": [],
                "quoted_evidence": "",
                "corrected_text": "",
                "claims": [],
            },
            0.0,
            {},
        )
    joined_parts = []
    for i, e in enumerate(excerpts):
        src = e.get("source_id", "memory")
        score = e.get("score")
        layer = e.get("layer") or "unknown"
        # Score is an RRF-family float; round to 4dp so the Judge
        # doesn't fixate on spurious precision. The literal string
        # "none" flags missing scores without collapsing to 0.0.
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "none"
        header = (
            f"[excerpt {i} | source={src} | score={score_str} "
            f"| layer={layer}]"
        )
        joined_parts.append(f"{header}\n{e.get('text', '')}")
    joined = "\n---\n".join(joined_parts)
    user = (
        f"Question:\n{question}\n\n"
        f"Draft response:\n{draft}\n\n"
        f"Authoritative excerpts:\n{joined}\n\n"
        "Return the JSON object now."
    )
    text, dt, usage = cerebras_chat(
        JUDGE,
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        MAX_TOKENS_JUDGE,
    )
    try:
        parsed = _extract_json_object(text)
    except ValueError:
        # Fail-closed: neutral verdict so we render the draft with
        # a "generated, unverified" tag rather than fabricate a
        # correction.
        parsed = {
            "has_claim": False,
            "verdict": "neutral",
            "cited_excerpt_ids": [],
            "quoted_evidence": "",
            "corrected_text": "",
            "claims": [],
        }
    # Ensure every returned judgment has a claims LIST so the
    # annotator does not need to None-guard or type-check it.
    # Defensive type-check: if the model returned a non-list
    # (e.g. a dict keyed by claim text, or a stray string), drop
    # it silently to []. Better to miss claim-level signaling on
    # one malformed response than to crash the whole smoke run.
    # Older stored logs predate this field; reading them stays
    # safe because the annotator treats missing as [].
    if not isinstance(parsed.get("claims"), list):
        parsed["claims"] = []
    return parsed, dt, usage


# Substrings that mark a draft as a refusal / hedge. Used by
# annotate_deterministic to detect the supports-on-refusal case:
# Judge says "supports" + finds a verbatim quote, but the Builder
# drafted "I don't know", so pasting a "[confirmed: X]" footer on
# top of the hedge produces a contradictory user-facing output.
# When this fires we synthesize a body from the quoted evidence
# with a distinct "recovered from uncertain draft" marker so the
# consumer can tell this is a retrieval-recovered answer, not an
# answer the Builder endorsed. Matching is case-insensitive
# substring; kept short to avoid false positives on drafts that
# merely hedge on a sub-claim inside a confident main answer.
REFUSAL_MARKERS = (
    "i'm not aware",
    "i am not aware",
    "i'm unable to verify",
    "i am unable to verify",
    "i cannot verify",
    "i can't verify",
    "i don't have info",
    "i do not have info",
    "i couldn't find",
    "i could not find",
    "no information",
)


def _draft_is_refusal(draft: str) -> bool:
    lower = draft.lower()
    return any(m in lower for m in REFUSAL_MARKERS)


def _claims_warning(claims: list[dict]) -> str:
    """If any sub-claim has verdict=contradicts or neutral, emit a
    warning section listing them. Empty string when all sub-claims
    are grounded.

    The warning is appended AFTER the primary provenance footer so
    the consumer sees top-line attribution first, then the per-
    sub-claim caveats. This preserves the reading-path for the
    happy case (all sub-claims supported -> no warning at all) and
    makes the drift loud when it exists.
    """
    if not claims:
        return ""
    unsupported = [
        c for c in claims
        if isinstance(c, dict) and c.get("verdict") in ("contradicts", "neutral")
    ]
    if not unsupported:
        return ""
    lines = [f"\n\n>> [{len(unsupported)} sub-claim(s) not grounded in memory]"]
    for c in unsupported:
        v = c.get("verdict", "?")
        text = (c.get("text") or "").strip().replace("\n", " ")
        text = (text[:120] + "...") if len(text) > 120 else text
        if v == "contradicts":
            ev = (c.get("quoted_evidence") or "").strip()
            ev_short = (ev[:120] + "...") if len(ev) > 120 else ev
            lines.append(f">>   contradicted: {text!r}")
            if ev_short:
                lines.append(f">>     source says: \"{ev_short}\"")
        else:
            lines.append(f">>   unsupported: {text!r}")
    return "\n".join(lines)


def annotate_deterministic(
    draft: str,
    judgment: dict,
    excerpts: list[dict],
) -> str:
    """Assemble the final user-facing response with provenance
    footer. Zero extra LLM calls.

    - contradicts: corrected_text + [corrected from draft; src |
      quoted...] footer.
    - supports:
        * draft is not a refusal: draft + [confirmed: src |
          quoted...] footer (the happy path).
        * draft IS a refusal: body is replaced with the quoted
          evidence and a "recovered from uncertain draft" marker
          so the consumer does not see "I don't know" followed by
          "confirmed: ...".
    - neutral / no_claim / no_retrieval: draft + [generated, no
      memory coverage] footer.

    Regardless of top-level verdict, if the Judge returned a
    ``claims`` array with any sub-claim whose verdict is
    ``contradicts`` or ``neutral``, a second warning section is
    appended listing those unsupported sub-claims. This is the
    lever that closes the answer-level-correct / claim-level-
    leaky gap surfaced on 2026-04-21 (cotrimoxazole threshold
    <200 vs <350, consolidation threshold 0.65 vs 0.70).

    The footer is a single-line marker so the downstream consumer
    can grep for it or strip it. An inline-per-sentence annotation
    would need another LLM call; the per-question footer gives the
    user the source attribution at minimal token cost.
    """
    verdict = judgment.get("verdict", "neutral")
    cited_ids = judgment.get("cited_excerpt_ids") or []
    quoted = (judgment.get("quoted_evidence") or "").strip()
    claims = judgment.get("claims") or []

    def _cite_sources() -> str:
        srcs = []
        for i in cited_ids:
            if isinstance(i, int) and 0 <= i < len(excerpts):
                srcs.append(excerpts[i].get("source_id", "memory"))
        return ", ".join(srcs) if srcs else "unknown"

    claims_note = _claims_warning(claims)

    if verdict == "contradicts":
        body = (judgment.get("corrected_text") or draft).strip()
        quoted_short = (quoted[:200] + "...") if len(quoted) > 200 else quoted
        footer = (
            f"\n\n>> [corrected from prior draft: {_cite_sources()}]"
            f"\n>> quoted evidence: \"{quoted_short}\""
        )
        return body + footer + claims_note

    if verdict == "supports":
        quoted_short = (quoted[:200] + "...") if len(quoted) > 200 else quoted
        if _draft_is_refusal(draft) and quoted:
            body = (
                f"Per authoritative memory ({_cite_sources()}): {quoted}"
            )
            footer = (
                f"\n\n>> [recovered from uncertain draft: {_cite_sources()}]"
                f"\n>> quoted evidence: \"{quoted_short}\""
            )
            return body + footer + claims_note
        footer = (
            f"\n\n>> [confirmed: {_cite_sources()}]"
            f"\n>> quoted evidence: \"{quoted_short}\""
        )
        return draft.strip() + footer + claims_note

    if verdict == "no_claim":
        return draft.strip() + "\n\n>> [no verifiable claim]" + claims_note

    # neutral, no_retrieval, or fail-closed fallback
    return (
        draft.strip() + "\n\n>> [generated, no memory coverage]" + claims_note
    )


# --------------------------------------------------------------- seed

def seed_vstash(chunk_paths: list[Path]) -> int:
    from vstash import Memory
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(DB_PATH).exists():
        print(f"NOTE: {DB_PATH} already exists; remove it first if you want a clean seed")
    mem = Memory(db=DB_PATH, project=PROJECT)
    n = 0
    try:
        for path in chunk_paths:
            path = Path(path)
            if not path.exists():
                print(f"skip {path} (missing)")
                continue
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    mem.remember(
                        text=row["text"],
                        title=row["protocol_id"],
                        tags=f'source:authoritative,protocol:{row["protocol_id"]}',
                    )
                    n += 1
                    if n % 50 == 0:
                        print(f"  seeded {n}...")
    finally:
        mem.close()
    print(f"done: seeded {n} chunks into {DB_PATH}")
    return n


# --------------------------------------------------------------- pipeline

def retrieve(
    mem,
    query: str,
    top_k: int = 5,
    *,
    retrieval_mode: str = "hybrid",
) -> list[dict]:
    """Return an excerpt pool as a list of dicts, each carrying the
    text plus every vstash signal we currently propagate:

      {
        "source_id": str,
        "text":      str,
        "score":     float | None,    # RRF score when available
        "layer":     str   | None,    # vstash layer tag
        "chunk_id":  str   | None,    # stable id for audit trails
        "added_at":  str   | None,    # ISO timestamp, freshness signal
        "collection": str  | None,    # bucket within the db
      }

    Downstream code (`judge_once`, the audit row) uses these to
    render a score-aware Judge prompt and to surface retrieval
    quality in the audit log -- see Silt's rule "before proposing
    an algorithm, look at the distribution of the data" and the
    glass-box invariant in the merken CONSTITUTION.

    ``retrieval_mode``:
      - ``"hybrid"`` (default): one vstash.search call with the
        stock adaptive RRF weighting. Cheap, ~100ms.
      - ``"dual"``: run four complementary searches and
        interleave the results -- see the inline comment in the
        branch for the full rationale. Each search is a vstash
        call (~100ms each), no LLM spend. The merged pool is
        NOT hard-capped; final size is bounded by
        ``top_k + top_k*3 + top_k*3 + top_k = 8 * top_k``
        candidates before text-prefix dedup. Evolved across
        Runs 4 -> 5 (2026-04-21) as the stuck-neutral
        diagnostic narrowed the root cause to Builder-draft
        synonym drift. 4th pure-vec pool added 2026-04-23 to
        attack seed=44 vocabulary-mismatch fails.
    """
    try:
        if retrieval_mode == "dual":
            # dual runs four searches so the Judge sees candidates
            # from complementary ranking regimes. The cost is four
            # ~100ms vstash calls, still zero LLM spend.
            #
            # 1. hybrid(question+draft): semantic + keyword, biased
            #    toward vec. Catches paraphrases and conceptually
            #    related chunks.
            # 2. fts(question+draft): pure keyword, wider top_k so
            #    specific chunks don't get buried by meta-intros.
            # 3. fts(question only): pure keyword on the STABLE
            #    half of the query. The Builder's draft introduces
            #    synonym drift (e.g. "TMP/SMX" vs "cotrimoxazole")
            #    that can push the one actionable chunk out of
            #    fts ranks. Searching the question alone sidesteps
            #    that drift.
            # 4. vec(question+draft): pure vector, no FTS weight.
            #    Complementary to (1) which uses adaptive RRF: when
            #    the hybrid pipeline still tilts toward FTS because
            #    the draft is keyword-dense, this branch guarantees
            #    a parallel pool ranked purely by embedding
            #    similarity. Targets vocabulary-mismatch fails
            #    where the answering chunk shares semantics but
            #    not tokens with the query (seed=44 qids:
            #    6e984302, gpt4_e061b84g).
            #
            # Diagnostic 2026-04-21 that motivated (3): the hiv-who
            # chunk with the CD4<350 threshold is fts rank 3 for
            # "cotrimoxazole prophylaxis HIV CD4" but drops below
            # top-10 when the Builder draft appends "TMP/SMX" to
            # the query.
            fts_k = top_k * 3
            # Extract the question half by splitting on the first
            # blank line (the pipeline forms the query as
            # f"{question}\n{draft[:400]}"). Fallback to the full
            # query when no newline is present.
            q_only = query.split("\n", 1)[0].strip() or query
            hybrid_hits = mem.search(query, top_k=top_k)
            fts_hits = mem.search(query, top_k=fts_k, retrieval_mode="fts_only")
            fts_q_hits = (
                mem.search(q_only, top_k=fts_k, retrieval_mode="fts_only")
                if q_only != query
                else []
            )
            vec_hits = mem.search(query, top_k=top_k, retrieval_mode="vec_only")
            # Interleave the four pools so no ranking regime
            # monopolises the prefix. The Judge sees a balanced
            # candidate set and has to pick the right chunk on
            # content, not on position.
            hits: list = []
            pools = [hybrid_hits, fts_hits, fts_q_hits, vec_hits]
            # strict=False: pools may differ in length when a search
            # returns fewer hits than requested; truncation to the
            # shortest pool is intentional, with the tail loop below
            # picking up the remainder.
            for row in zip(*pools, strict=False):
                hits.extend(row)
            # Tail-append anything still remaining in the longest pool.
            max_len = max(len(p) for p in pools)
            for i in range(min(len(p) for p in pools), max_len):
                for p in pools:
                    if i < len(p):
                        hits.append(p[i])
        else:
            hits = mem.search(query, top_k=top_k)
    except Exception as e:
        print(f"    search error for {query[:60]!r}: {e}")
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for h in hits:
        text = getattr(h, "text", None) or getattr(h, "content", None) or ""
        if not text:
            continue
        # Dedup key. Previously ``text[:120]`` which collides on
        # conversational haystacks (LongMemEval turns like
        # ``"user: hi, how are you"`` / ``"user: hi, how are you
        # doing today?"`` share the same 120-char prefix), silently
        # dropping distinct excerpts. Fix 2026-04-21: prefer
        # vstash's ``chunk_id`` (the authoritative per-chunk id)
        # when it's on the SearchResult; fall back to the full-
        # content tuple (source_id + full text) so we never alias
        # two different rows on any prefix.
        chunk_id = getattr(h, "chunk_id", None)
        if chunk_id is not None:
            key = f"chunk:{chunk_id}"
        else:
            # Fallback when vstash does not publish chunk_id on this
            # SearchResult: digest the (source, text) pair so the
            # dedup set stays bounded in memory regardless of
            # excerpt length. Full-content concat was O(total_text)
            # per key and labelled "hash:" without actually hashing
            # (bot review flagged both issues). blake2s over the
            # composed bytes is ~microseconds and 32 bytes per key.
            src_raw = (
                getattr(h, "title", None)
                or getattr(h, "document_title", None)
                or getattr(h, "tags", None)
                or ""
            )
            digest = hashlib.blake2s(
                f"{src_raw}::{text}".encode("utf-8", "replace"),
                digest_size=16,
            ).hexdigest()
            key = f"hash:{digest}"
        if key in seen:
            continue
        seen.add(key)
        # vstash SearchResult exposes title/tags; title is the
        # protocol_id we set at seed time. Fall back to a short
        # prefix tag if the structure shifts.
        src_id = (
            getattr(h, "title", None)
            or getattr(h, "document_title", None)
            or getattr(h, "tags", None)
            or "memory"
        )
        # Drop LongMemEval haystack pollution. Session ids
        # prefixed ``sharegpt_`` are filler dialogues from the
        # ShareGPT haystack, NOT the user's own sessions. Title
        # shape in benchmark ingest is
        # ``<question_id>::<session_id>::<turn>`` (see
        # mode_a_eval._ingest), so session_id sits after the
        # first "::". For non-benchmark callers (seed_vstash
        # uses bare protocol_id titles) this check is a no-op.
        src_str = str(src_id)
        parts = src_str.split("::", 2)
        if len(parts) >= 2 and parts[1].startswith("sharegpt_"):
            continue
        # Pull every signal vstash publishes. Missing fields stay
        # None so downstream code can uniformly treat them as
        # "unknown" without a KeyError.
        out.append({
            "source_id": str(src_id),
            "text": text,
            "score": getattr(h, "score", None),
            "layer": getattr(h, "layer", None),
            "chunk_id": getattr(h, "chunk_id", None),
            "added_at": getattr(h, "added_at", None),
            "collection": getattr(h, "collection", None),
        })
    return out


def retrieval_stats(excerpts: list[dict]) -> dict:
    """Summarise the excerpt pool for the audit log. No per-query
    decision is derived from these numbers yet; we surface them so
    later sessions can spot patterns (e.g. "all neutrals had
    max_score < 0.012") without re-running everything.
    """
    scores = [
        e.get("score") for e in excerpts
        if isinstance(e.get("score"), (int, float))
    ]
    layers: dict[str, int] = {}
    for e in excerpts:
        layer = e.get("layer") or "unknown"
        layers[layer] = layers.get(layer, 0) + 1
    if not scores:
        return {
            "n_excerpts": len(excerpts),
            "n_scored": 0,
            "max_score": None,
            "min_score": None,
            "mean_score": None,
            "layers": layers,
        }
    return {
        "n_excerpts": len(excerpts),
        "n_scored": len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "mean_score": sum(scores) / len(scores),
        "layers": layers,
    }


def run_one(
    mem,
    question: str,
    *,
    builder_mode: str = "auto",
    retrieval_mode: str = "hybrid",
) -> dict:
    """Two-call pipeline: Builder draft + Judge finalize.

    Retrieval runs once with the draft+question as the query.
    Annotation is deterministic (no extra LLM call).
    ``builder_mode`` selects a Builder system prompt from
    ``BUILDER_MODES``; ``"auto"`` sends no system prompt and is
    the realistic production default.
    """
    print(f"\n{'='*72}\nQ: {question}\n{'='*72}")

    # --- stage 1: Builder draft ------------------------------------
    b_sys = BUILDER_MODES.get(builder_mode)
    b_messages: list[dict] = []
    if b_sys is not None:
        b_messages.append({"role": "system", "content": b_sys})
    b_messages.append({"role": "user", "content": question})
    draft, b_dt, b_usage = cerebras_chat(BUILDER, b_messages, MAX_TOKENS_DRAFT)
    print(f"\n[BUILDER | {b_dt:.2f}s | tok={b_usage.get('total_tokens', '?')}]\n{draft}")

    # --- stage 2: retrieve (no LLM) --------------------------------
    # Query uses both the question and the draft so the retrieval
    # surfaces chunks relevant to whatever the Builder actually said,
    # not just to the abstract question.
    query = f"{question}\n{draft[:400]}"
    excerpts = retrieve(mem, query, top_k=5, retrieval_mode=retrieval_mode)
    r_stats = retrieval_stats(excerpts)
    top_src = excerpts[0].get("source_id") if excerpts else None
    print(
        f"[RETRIEVE] got {len(excerpts)} excerpts"
        + (f"; top_src={top_src}" if top_src else "")
        + (
            f"; scores[max={r_stats['max_score']:.4f} "
            f"min={r_stats['min_score']:.4f} "
            f"mean={r_stats['mean_score']:.4f}]"
            if r_stats["max_score"] is not None else ""
        )
    )

    # --- stage 3: Judge single call --------------------------------
    judgment, j_dt, j_usage = judge_once(question, draft, excerpts)
    claims = judgment.get("claims") or []
    bad_claims = [
        c for c in claims
        if isinstance(c, dict) and c.get("verdict") in ("contradicts", "neutral")
    ]
    print(
        f"[JUDGE | {j_dt:.2f}s | tok={j_usage.get('total_tokens', '?')}] "
        f"has_claim={judgment.get('has_claim')} "
        f"verdict={judgment.get('verdict')} "
        f"cited={judgment.get('cited_excerpt_ids')} "
        f"sub_claims={len(claims)} unsupported={len(bad_claims)}"
    )
    if judgment.get("verdict") == "contradicts":
        print(f"         corrected_text: {judgment.get('corrected_text', '')[:180]}...")
    for c in bad_claims:
        v = c.get("verdict", "?")
        text = (c.get("text") or "")[:100]
        print(f"         sub-claim {v}: {text!r}")

    # --- stage 4: deterministic annotation (no LLM) ----------------
    final = annotate_deterministic(draft, judgment, excerpts)
    print(f"\n[FINAL]\n{final}")

    total_tokens = (b_usage.get("total_tokens", 0) or 0) + (j_usage.get("total_tokens", 0) or 0)
    # Audit-ready row: every Mode A run IS a labeled training example.
    # The builder / judge identifiers + timestamp let downstream
    # trainers slice by model version and freshness.
    #
    # `retrieved` now carries every vstash signal (score, layer,
    # chunk_id, added_at, collection) per excerpt so downstream
    # training code can build features on retrieval quality
    # without re-running the pipeline. `retrieval_stats` is the
    # per-query summary for fast pandas-style filtering.
    # `judgment.claims` is the per-sub-claim label array (the
    # claim-level ground truth for a sub-claim classifier).
    return {
        "audit_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "builder_model": BUILDER,
        "judge_model": JUDGE,
        "question": question,
        "draft": draft,
        "draft_s": b_dt,
        "draft_usage": b_usage,
        "retrieved": excerpts,
        "retrieval_stats": r_stats,
        "judgment": judgment,
        "judge_s": j_dt,
        "judge_usage": j_usage,
        "final": final,
        "total_tokens": total_tokens,
        "verdict": judgment.get("verdict"),
        "fired": judgment.get("verdict") == "contradicts",
        "n_sub_claims": len(claims),
        "n_sub_claims_unsupported": len(bad_claims),
    }


def run_smoke(
    questions: list[tuple[str, str]],
    *,
    log_name: str = "cerebras_smoke.jsonl",
    builder_mode: str = "auto",
    retrieval_mode: str = "hybrid",
) -> list[dict]:
    from vstash import Memory
    if not Path(DB_PATH).exists():
        sys.exit(
            f"error: vstash db not found at {DB_PATH}. "
            f"run `python {Path(__file__).name} seed` first."
        )
    mem = Memory(db=DB_PATH, project=PROJECT)
    logs: list[dict] = []
    try:
        for qid, q in questions:
            row = run_one(
                mem, q,
                builder_mode=builder_mode,
                retrieval_mode=retrieval_mode,
            )
            row["qid"] = qid
            row["builder_mode"] = builder_mode
            row["retrieval_mode"] = retrieval_mode
            logs.append(row)
    finally:
        mem.close()

    out = Path(__file__).parent / log_name
    with out.open("w") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    print(f"\nlog: {out}")

    # --- summary table ---------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n = len(logs)
    by_verdict: dict[str, int] = {}
    for r in logs:
        v = r.get("verdict") or "unknown"
        by_verdict[v] = by_verdict.get(v, 0) + 1
    total_tok = sum(r.get("total_tokens", 0) for r in logs)
    avg_tok = total_tok / max(n, 1)
    print(f"  questions:            {n}")
    for v, c in sorted(by_verdict.items()):
        print(f"  verdict={v:12s}   {c}/{n}")
    print(f"  total tokens used:    {total_tok}")
    print(f"  avg tokens/question:  {avg_tok:.0f}")
    for r in logs:
        print(
            f"    - {r['qid']:30s} verdict={r.get('verdict'):12s} "
            f"tok={r.get('total_tokens')}"
        )
    return logs


# --------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp_seed = sp.add_parser("seed", help="ingest protocol chunks into vstash")
    sp_seed.add_argument("chunks", nargs="*", type=Path)

    sp_run = sp.add_parser("run", help="run baseline + midloop on smoke questions")
    sp_run.add_argument("--question", type=str, help="single free-form question")
    sp_run.add_argument("--all", action="store_true", help="run the full smoke set")
    sp_run.add_argument(
        "--domain",
        choices=sorted(DOMAIN_FRAMES.keys()),
        default="clinical",
        help=(
            "which Judge framing + default smoke set to use. "
            "'clinical' = WHO/ICRC guidelines (default); "
            "'personal' = user's project memory."
        ),
    )
    sp_run.add_argument(
        "--db",
        type=str,
        default=None,
        help="override the vstash db path (default is the clinical medlocal db)",
    )
    sp_run.add_argument(
        "--project",
        type=str,
        default=None,
        help="override the vstash project filter (default medlocal_concept)",
    )
    sp_run.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="override the output jsonl filename (default cerebras_smoke.jsonl)",
    )
    sp_run.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "dual"],
        default="hybrid",
        help=(
            "Retrieval strategy. 'hybrid' (default) = one vstash "
            "adaptive-RRF search. 'dual' = hybrid + fts_only merged, "
            "for corpora where vec-dominant ranking buries exact-"
            "keyword matches (e.g. dense clinical guideline chunks)."
        ),
    )
    sp_run.add_argument(
        "--builder-mode",
        choices=sorted(BUILDER_MODES.keys()),
        default="auto",
        help=(
            "Builder system prompt. 'auto' (default) sends no system "
            "prompt and lets the small model hedge/refuse naturally. "
            "'confident' forces authoritative answers -- used for the "
            "adversarial smoke that exercises Mode A's hallucination-"
            "correction path instead of its refusal path."
        ),
    )

    args = ap.parse_args()
    if args.cmd == "seed":
        paths = args.chunks or DEFAULT_CHUNKS
        seed_vstash(paths)
        return 0

    if args.cmd == "run":
        # Resolve per-domain defaults so a plain --domain personal
        # run picks the right DB, project, Judge prompt, and log.
        global DB_PATH, PROJECT, JUDGE_SYSTEM
        if args.domain == "personal":
            default_db = str(Path.home() / ".vstash" / "memory.db")
            default_project = "engram"
            default_log = "cerebras_personal_smoke.jsonl"
            default_questions = PERSONAL_SMOKE_QUESTIONS
        else:
            default_db = DB_PATH
            default_project = PROJECT
            default_log = "cerebras_smoke.jsonl"
            default_questions = SMOKE_QUESTIONS
        DB_PATH = args.db or default_db
        PROJECT = args.project or default_project
        JUDGE_SYSTEM = JUDGE_SYSTEM_TEMPLATE.format(
            domain_frame=DOMAIN_FRAMES[args.domain]
        )
        log_name = args.log_name or default_log

        qs = [("ad_hoc", args.question)] if args.question else default_questions

        # Non-default knobs get suffixed logs so they do not
        # clobber a baseline-mode log for the same domain.
        if args.log_name is None:
            parts = []
            if args.builder_mode != "auto":
                parts.append(args.builder_mode)
            if args.retrieval_mode != "hybrid":
                parts.append(args.retrieval_mode)
            if parts:
                stem = Path(log_name).stem
                suffix = Path(log_name).suffix or ".jsonl"
                log_name = f"{stem}_{'_'.join(parts)}{suffix}"

        print(
            f"[config] domain={args.domain} builder_mode={args.builder_mode} "
            f"retrieval_mode={args.retrieval_mode} "
            f"db={DB_PATH} project={PROJECT} log={log_name}"
        )
        run_smoke(
            qs,
            log_name=log_name,
            builder_mode=args.builder_mode,
            retrieval_mode=args.retrieval_mode,
        )
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
