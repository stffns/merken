"""Jev-backed deciders: TypeSafe's System One model at merken's decision points.

Jev does not generate text. It evaluates typed questions (``choice`` /
``score`` / ``noul``) against a state and returns a decision with
calibrated ``probabilities`` and ``confidence`` — the exact shape of
merken's ``Decision``. One HTTP call, ~300 ms, ~$0.00002 per event.

Three things live here, one per decision point:

- ``JevWriteDecider`` — DECISION/NOISE ``should_remember`` classifier.
  Same instructions as ``merken.classifiers.llm.LLMWriteDecider`` so
  the two are comparable; no torch, no model download.
- ``JevReranker`` — post-retrieval reranker for ``Memory.recall``:
  keeps hits that answer the query, puts the one describing the
  *current* state first.
- ``JevMaterializer`` — ``materialize_fn`` for ``Memory.consolidate``:
  extractive fact = the cluster member Jev picks as the current
  state; members stay in the provenance only if they are a version of
  the same decision and DECISION-class.

**Opt-in, network.** CONSTITUTION §4.1 (local-first) still holds: nothing
here is wired by default. Activate with ``MERKEN_SHADOW=jev`` /
``MERKEN_PRIMARY=jev`` or by passing the objects to ``Memory``.
Every call degrades safely: a write decider error means *write*
(losing a decision is worse than storing noise), a reranker error
returns the hits unchanged, a materializer error falls back to
``materialize_fact``.

Transport: OpenRouter's Decisions endpoint (``OPENROUTER_API_KEY``) or
TypeSafe's API (``TYPESAFE_API_KEY``). Both take the same body and
return the same answers. See ``experiments/loop_quality/jev_probe.py``
for the ablation that justified each piece.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from merken.consolidation import Fact, materialize_fact
from merken.policies.types import Decision, Event, WriteContext

if TYPE_CHECKING:
    from vstash import SearchResult

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"  # alpha path; may move
OPENROUTER_MODEL = "~typesafe/jev-latest"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"

# Verbatim from merken/classifiers/llm.py so the deciders are comparable.
DEFAULT_INSTRUCTIONS = (
    "Classify the event as DECISION or NOISE. A DECISION records a lasting "
    "choice, commitment, specification, rollout plan, or reference data "
    "worth keeping. A NOISE event is routine status, attendance, burndown, "
    "or ephemeral log that nobody needs to re-read."
)
DEFAULT_CRITERIA: dict[str, str] = {
    "DECISION": (
        "Records a lasting choice, commitment, specification, rollout plan, "
        "diagnosis, or reference data worth keeping"
    ),
    "NOISE": (
        "Routine status, attendance, burndown, or ephemeral log that nobody "
        "needs to re-read"
    ),
}

CallFn = Callable[[dict[str, Any]], dict[str, Any]]


class JevClient:
    """Minimal HTTP client for the Jev decisions endpoint. Stdlib only.

    ``call_fn`` replaces the transport (tests inject a fake that maps
    a request body to a response body). With no explicit ``api_key``
    the client resolves, in order, ``OPENROUTER_API_KEY`` then
    ``TYPESAFE_API_KEY``, and picks the matching endpoint + model.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        model: str | None = None,
        timeout_s: float = 20.0,
        max_retries: int = 3,
        call_fn: CallFn | None = None,
    ) -> None:
        if call_fn is None and api_key is None:
            if os.environ.get("OPENROUTER_API_KEY"):
                api_key = os.environ["OPENROUTER_API_KEY"]
                url, model = url or OPENROUTER_URL, model or OPENROUTER_MODEL
            elif os.environ.get("TYPESAFE_API_KEY"):
                api_key = os.environ["TYPESAFE_API_KEY"]
                url, model = url or TYPESAFE_URL, model or TYPESAFE_MODEL
            else:
                raise RuntimeError(
                    "JevClient needs OPENROUTER_API_KEY or TYPESAFE_API_KEY "
                    "(or an explicit api_key / call_fn)."
                )
        self._api_key = api_key
        self.url = url or OPENROUTER_URL
        self.model = model or OPENROUTER_MODEL
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._call_fn = call_fn

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """POST one state + questions, return the parsed response (``answers`` etc.)."""
        body = {"model": self.model, "state": state, "questions": questions}
        if self._call_fn is not None:
            return self._call_fn(body)
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                retryable = e.code in (429, 500, 502, 503, 529)
                if not retryable or attempt >= self._max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt >= self._max_retries:
                    raise
            time.sleep(min(8.0, 1.0 * 2**attempt))
            attempt += 1

    def decide_many(
        self,
        items: Sequence[tuple[Any, dict[str, dict[str, Any]]]],
        *,
        workers: int = 8,
    ) -> list[dict[str, Any]]:
        """``decide`` over many (state, questions) pairs, in parallel, order preserved."""
        if len(items) <= 1 or workers <= 1:
            return [self.decide(s, q) for s, q in items]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda it: self.decide(it[0], it[1]), items))


# --------------------------------------------------------------- should_remember


class JevWriteDecider:
    """DECISION/NOISE classifier on Jev's ``choice`` primitive.

    ``confidence_threshold``: below it the event is *written* with
    reason ``low_conf:<c>`` so shadow analysis can find the doubtful
    cases. Above it, Jev's choice is authoritative. Keeping on doubt is
    the same asymmetry ``classification.py`` documents: a false
    negative drops a valuable event, a false positive costs storage.
    """

    name = "jev-classifier"

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        confidence_threshold: float = 0.6,
        instructions: str = DEFAULT_INSTRUCTIONS,
        criteria: dict[str, str] | None = None,
        max_input_chars: int = 100_000,
    ) -> None:
        self._client = client or JevClient()
        self._threshold = confidence_threshold
        self._max_input_chars = max_input_chars
        self._questions = {
            "label": {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(criteria or DEFAULT_CRITERIA),
            }
        }

    def label(self, text: str) -> dict[str, Any]:
        """Raw Jev answer for ``text`` (``choice``, ``probabilities``, ``confidence``)."""
        state = text[: self._max_input_chars]
        return self._client.decide(state, self._questions)["answers"]["label"]

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        try:
            answer = self.label(event.text)
            choice = answer["choice"]
            p_d = float(answer["probabilities"].get("DECISION", 0.0))
            p_n = float(answer["probabilities"].get("NOISE", 0.0))
            conf = float(answer.get("confidence", max(p_d, p_n)))
        except Exception as exc:  # noqa: BLE001 — never block a write
            return Decision(
                write=True,
                reason=f"jev_error:{exc.__class__.__name__}",
                confidence=0.0,
                policy=self.name,
            )
        if conf < self._threshold:
            return Decision(
                write=True,
                reason=f"low_conf:{conf:.3f} P(D)={p_d:.3f}",
                confidence=conf,
                policy=self.name,
            )
        return Decision(
            write=choice == "DECISION",
            reason=f"{choice.lower()} P(D)={p_d:.3f} P(N)={p_n:.3f}",
            confidence=conf,
            policy=self.name,
        )


# ------------------------------------------------------------------- recall


class JevReranker:
    """Rerank recall hits: answers-the-query first, current state on top.

    Two questions:

    1. per hit, ``noul`` "this text directly answers the question" —
       hits scoring below ``min_answer`` drop out (unless that would
       leave nothing, in which case the original order is kept);
    2. over the top ``pick_from`` survivors, ``choice`` "which gives
       the CURRENT, most up-to-date answer" — that hit goes first.

    Step 2 is what turns "the right topic is somewhere in the top-k"
    into "the first hit is the answer" on knowledge-update content:
    Jev cannot tell Redis-then-Caffeine apart one event at a time, but
    it can when it sees both.
    """

    name = "jev-reranker"

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        min_answer: float = 0.5,
        pick_current: bool = True,
        pick_from: int = 6,
        min_keep: int = 0,
        current_instructions: str | None = None,
        workers: int = 8,
    ) -> None:
        self._client = client or JevClient()
        self._min_answer = min_answer
        self._pick_current = pick_current
        # ``{query}`` is substituted. Override when "current" is not simply
        # "most recent" (e.g. questions that ask about an earlier time).
        self._current_instructions = current_instructions or (
            'Which text gives the CURRENT, most up-to-date answer to: "{query}"? '
            "Prefer the one that supersedes the others."
        )
        self._pick_from = pick_from
        # Never return fewer than this many hits: after the answers-filter,
        # backfill from Jev's ordering. Aggregation questions ("total I
        # earned across sales") need several excerpts that each only
        # *partially* answer; a strict filter starves the reader.
        self._min_keep = min_keep
        self._workers = workers

    def rerank(self, query: str, hits: list[SearchResult]) -> list[SearchResult]:
        if len(hits) <= 1:
            return list(hits)
        try:
            return self._rerank(query, hits)
        except Exception:  # noqa: BLE001 — never lose the retrieval result
            return list(hits)

    def _rerank(self, query: str, hits: list[SearchResult]) -> list[SearchResult]:
        q = {
            "answers": {
                "type": "noul",
                "instructions": f'This text directly answers the question: "{query}"',
            }
        }
        responses = self._client.decide_many(
            [((h.text or ""), q) for h in hits], workers=self._workers
        )
        scores = [float(r["answers"]["answers"]["noul"]) for r in responses]
        # Stable: ties keep vstash's order.
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        kept = [hits[i] for i in order if scores[i] >= self._min_answer]
        if not kept:
            return list(hits)
        if len(kept) < self._min_keep:
            extra = [hits[i] for i in order if scores[i] < self._min_answer]
            kept = kept + extra[: self._min_keep - len(kept)]
        if self._pick_current and len(kept) >= 2:
            head = kept[: self._pick_from]
            keys = [f"h{i}" for i in range(len(head))]
            r = self._client.decide(
                dict(zip(keys, [(h.text or "") for h in head], strict=True)),
                {
                    "current": {
                        "type": "choice",
                        "instructions": self._current_instructions.format(query=query),
                        "criteria": dict.fromkeys(keys),
                    }
                },
            )
            first = head[keys.index(r["answers"]["current"]["choice"])]
            kept = [first] + [h for h in kept if h is not first]
        return kept


# --------------------------------------------------------------- consolidate


class JevMaterializer:
    """``materialize_fn`` for ``Memory.consolidate``: extractive, provenance-pure.

    For a cluster of ``(id, text)`` members:

    1. ``choice`` "which event describes the CURRENT state" → that text
       becomes the fact (prefixed with the observation count, like
       ``materialize_fact``). Never invents text.
    2. every other member stays in ``derived_from`` only if Jev says it
       is a version/amendment of the *same* decision **and** classifies
       it DECISION. Operational tickets that merely mention the same
       system (the adversarial noise in ``knowledge_update_*``) are
       expelled, which is what took cluster purity 91% -> 100% on
       ``knowledge_update_50topics``.

    Falls back to ``materialize_fact`` on any error.
    """

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        same_decision_min: float = 0.5,
        require_decision_label: bool = True,
        workers: int = 8,
    ) -> None:
        self._client = client or JevClient()
        self._same_min = same_decision_min
        self._require_label = require_decision_label
        self._workers = workers
        self._labeler = JevWriteDecider(self._client) if require_decision_label else None

    def __call__(self, cluster: list[tuple[str, str]]) -> Fact:
        if len(cluster) <= 1:
            return materialize_fact(cluster)
        try:
            return self._materialize(cluster)
        except Exception:  # noqa: BLE001
            return materialize_fact(cluster)

    def _materialize(self, cluster: list[tuple[str, str]]) -> Fact:
        ids = [id_ for id_, _ in cluster]
        keys = [f"e{i}" for i in range(len(cluster))]
        r = self._client.decide(
            dict(zip(keys, [t for _, t in cluster], strict=True)),
            {
                "current": {
                    "type": "choice",
                    "instructions": (
                        "Which event describes the CURRENT state: the most recent decision "
                        "that supersedes or updates the others? If they are unrelated, pick "
                        "the most informative one."
                    ),
                    "criteria": dict.fromkeys(keys),
                }
            },
        )
        cur = keys.index(r["answers"]["current"]["choice"])
        cur_text = cluster[cur][1]
        others = [(i, t) for i, (_, t) in enumerate(cluster) if i != cur]

        same_q = {
            "same": {
                "type": "noul",
                "instructions": (
                    "The candidate is an earlier version, a later revision, or a direct "
                    "amendment of the SAME decision recorded in current_fact — not an "
                    "operational ticket, incident, task, or status update that merely "
                    "mentions the same system"
                ),
            }
        }
        same = self._client.decide_many(
            [({"current_fact": cur_text, "candidate": t}, same_q) for _, t in others],
            workers=self._workers,
        )
        labels: list[str | None]
        if self._labeler is not None:
            labels = [
                res["answers"]["label"]["choice"]
                for res in self._client.decide_many(
                    [(t, self._labeler._questions) for _, t in others],
                    workers=self._workers,
                )
            ]
        else:
            labels = [None] * len(others)

        kept = [ids[cur]]
        for (i, _), s, label in zip(others, same, labels, strict=True):
            if float(s["answers"]["same"]["noul"]) < self._same_min:
                continue
            if self._require_label and label != "DECISION":
                continue
            kept.append(ids[i])

        if len(kept) == 1:
            return Fact(text=cur_text, derived_from=kept, cluster_size=1, method="jev_current_v1")
        return Fact(
            text=f"[observed {len(kept)}×] {cur_text}",
            derived_from=kept,
            cluster_size=len(kept),
            method="jev_current_v1",
        )
