"""Write-time enrichment: merken writes the memory, not just filters it.

Queries are phrased at category level ("what monitoring platform do we use?")
while events are instance level ("Migrated from Prometheus/Thanos to
Datadog…"). With no lexical overlap, neither FTS nor a small embedder bridges
the gap, and no reranker can reorder what retrieval never returns
(``experiments/loop_quality/RESULTS.md``, retrieval ceiling, 2026-09-19).

An :class:`Enricher` closes the gap at write time: for an event the write
policy kept, it returns one category-level line ("CI platform: Buildkite
(replaced GitHub Actions)") that ``Memory.remember`` stores *with* the event,
so FTS, the embedder, consolidation and any reranker all see it. The enricher
never sees a query.

The stored text is ``"» <line> «\\n\\n<original text>"``; :func:`strip_enrichment`
removes the line again, which is what the write decider's dedup hydration uses
so a re-submitted raw event is still recognised as a duplicate.

Nothing here is on by default. Enrichment failures never fail a write: on any
error ``Memory.remember`` stores the raw text (fail open).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

ENRICH_PROMPT = (
    "You index engineering notes for later retrieval. Write ONE line, at most 25 words, "
    "that states the general category this note answers and the current choice, so that "
    "a question phrased generically (e.g. 'what monitoring platform do we use?') would "
    "match it. Format: '<Category>: <current choice> (replaced <previous>, if any)'. "
    "Use the note's own facts only. No preamble.\n\nNote:\n"
)

_OPEN, _CLOSE = "»", "«"
_ENRICHED = re.compile(rf"^{_OPEN} [^\n]*? {_CLOSE}\s*")


@runtime_checkable
class Enricher(Protocol):
    """Returns one retrieval line for ``text``, or ``None`` to store it as is."""

    name: str

    def enrich(self, text: str) -> str | None: ...


def apply_enrichment(text: str, line: str | None) -> str:
    """The text ``Memory.remember`` stores: the line on top, the event untouched."""
    line = " ".join((line or "").split()).replace(_CLOSE, "").replace(_OPEN, "")
    if not line:
        return text
    return f"{_OPEN} {line} {_CLOSE}\n\n{text}"


def strip_enrichment(text: str) -> str:
    """Inverse of :func:`apply_enrichment` (no-op on text that was not enriched)."""
    return _ENRICHED.sub("", text, count=1)


class LLMEnricher:
    """One-line enrichment from any OpenAI-compatible ``/chat/completions`` endpoint.

    Defaults to DeepSeek's ``deepseek-flash`` with thinking disabled (a reasoning
    model otherwise spends ``max_tokens`` thinking and returns empty content,
    which silently degrades to no enrichment). The key comes from ``api_key``,
    then ``MERKEN_ENRICH_API_KEY``, then ``DEEPSEEK_API_KEY``.

    ``gate`` decides which events are worth paying for: it receives the raw text
    and returns ``True`` to enrich. :func:`jev_decision_gate` keeps the LLM off
    everything Jev (hosted or local d1) labels NOISE. Without a gate every
    written event is enriched.

    ``complete_fn`` replaces the HTTP call (tests, or a local model): it takes
    the prompt and returns the model's raw text.
    """

    def __init__(
        self,
        *,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        gate: Callable[[str], bool] | None = None,
        extra_body: dict | None = None,
        prompt: str = ENRICH_PROMPT,
        max_tokens: int = 120,
        timeout: float = 30.0,
        complete_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.name = f"llm:{model}"
        self.model = model
        self._gate = gate
        self._prompt = prompt
        self._complete = complete_fn
        if complete_fn is None:
            key = (
                api_key
                or os.environ.get("MERKEN_ENRICH_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
            )
            if not key:
                raise RuntimeError(
                    "LLMEnricher needs api_key=, MERKEN_ENRICH_API_KEY or DEEPSEEK_API_KEY."
                )
            self._key = key
            self._url = base_url.rstrip("/") + "/chat/completions"
            self._extra = (
                {"thinking": {"type": "disabled"}} if extra_body is None else dict(extra_body)
            )
            self._max_tokens = max_tokens
            self._timeout = timeout
        self.stats = {"gated_out": 0, "calls": 0, "empty": 0}

    def _http_complete(self, prompt: str) -> str:
        import urllib.request

        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **self._extra,
        }
        req = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return data["choices"][0]["message"].get("content") or ""

    def enrich(self, text: str) -> str | None:
        if self._gate is not None and not self._gate(text):
            self.stats["gated_out"] += 1
            return None
        complete = self._complete or self._http_complete
        raw = complete(self._prompt + text[:3000])
        self.stats["calls"] += 1
        line = next((ln.strip() for ln in raw.strip().splitlines() if ln.strip()), "")
        if not line:
            self.stats["empty"] += 1
            return None
        return line


def jev_decision_gate(client: object | None = None) -> Callable[[str], bool]:
    """Gate that enriches only what Jev labels DECISION.

    ``client`` is a :class:`merken.classifiers.jev.JevClient` (hosted, or
    ``JevClient.local()`` for the keyless d1 model); ``None`` builds the default
    one. Any Jev error lets the event through: enriching a noise event costs a
    fraction of a cent, losing a decision's line costs a missed recall.
    """
    from merken.classifiers.jev import JevWriteDecider

    decider = JevWriteDecider(client)  # type: ignore[arg-type]

    def gate(text: str) -> bool:
        try:
            return decider.label(text)["choice"] == "DECISION"
        except Exception:  # noqa: BLE001 -- fail open, see docstring
            return True

    return gate
