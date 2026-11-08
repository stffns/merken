"""Oracular labeling of shadow-mode disagreements.

Shadow mode writes a ``shadow_disagree`` tag into every audit row where
the primary decider and the shadow classifier disagreed. Left alone,
those tags are just annotations. This module turns them into labeled
training data by asking an oracle (LLM backend) which side was right.

Flow::

    ShadowWriteDecider      -> audit row ``novel|shadow_disagree:...``
    label_disagreements     -> for each disagreement, backend.label(text)
                            -> label row in ``merken_labels`` collection

Labels are stored per-project in the same vstash DB as the audit log,
under the ``merken_labels`` collection. Re-running ``label_disagreements``
skips events that already have a label -- it is safe to run
repeatedly, and safe to interrupt.

The backend protocol is deliberately thin; any object with
``name: str`` and ``label(text) -> Label`` works. That lets callers
compose oracles (Gemini, Claude, local Gemma, even a human-in-the-loop
wrapper) without touching merken internals.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from merken.memory import Memory


LABEL_COLLECTION = "merken_labels"
LABEL_LAYER = "audit"

_PROMPT = (
    "You are labeling memory events for an agent's write-filter training set.\n"
    "Read the event below and classify it as exactly one of:\n"
    "  DECISION  -- a lasting commitment, design choice, resolved open\n"
    "              question, or reference material worth keeping.\n"
    "  NOISE     -- routine status, attendance, transient update, or\n"
    "              ephemeral log with no lasting value.\n"
    "  UNCERTAIN -- genuinely ambiguous; a reasonable person could go\n"
    "              either way.\n\n"
    "Event:\n"
    "<<<\n"
    "{text}\n"
    ">>>\n\n"
    "Respond on a single line in this exact format (no prose, no code fences):\n"
    "LABEL: <DECISION|NOISE|UNCERTAIN> | CONFIDENCE: <number 0.0-1.0> | "
    "REASON: <one sentence>"
)


@dataclass(frozen=True)
class Label:
    """One oracular judgment on a disagreement."""

    decision: str  # "DECISION" | "NOISE" | "UNCERTAIN"
    confidence: float
    rationale: str
    backend: str


class LabelBackend(Protocol):
    """Protocol for label oracles.

    Implementations should be idempotent and deterministic where
    possible (temperature=0 for LLMs) so re-running produces stable
    training data.
    """

    name: str

    def label(self, text: str) -> Label: ...


def _parse_backend_response(raw: str, backend_name: str) -> Label:
    """Parse ``LABEL: X | CONFIDENCE: 0.8 | REASON: ...`` into a Label.

    Degrades gracefully on malformed responses: returns an UNCERTAIN
    label with confidence 0 and the raw text as the rationale, so a
    misbehaving backend never crashes the labeling loop.
    """
    parts: dict[str, str] = {}
    for piece in raw.splitlines()[0].split("|") if raw.strip() else []:
        if ":" not in piece:
            continue
        key, _, value = piece.partition(":")
        parts[key.strip().upper()] = value.strip()

    decision = parts.get("LABEL", "UNCERTAIN").upper()
    if decision not in ("DECISION", "NOISE", "UNCERTAIN"):
        decision = "UNCERTAIN"

    try:
        confidence = float(parts.get("CONFIDENCE", "0"))
    except ValueError:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = parts.get("REASON", raw.strip())
    return Label(
        decision=decision,
        confidence=confidence,
        rationale=rationale[:500],
        backend=backend_name,
    )


class GeminiLabelBackend:
    """LLM label oracle backed by Google's genai SDK (Gemini)."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        from google import genai

        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "GeminiLabelBackend needs GEMINI_API_KEY or GOOGLE_API_KEY "
                "set in the environment."
            )
        self._client = genai.Client(api_key=resolved_key)
        self.name = model

    def label(self, text: str) -> Label:
        # .replace instead of .format: raw text routinely contains
        # literal `{` / `}` (JSON, code, f-strings). Those would be
        # read as format placeholders and raise KeyError, causing
        # labeling to fail silently for structured content.
        prompt = _PROMPT.replace("{text}", text[:3000])
        resp = self._client.models.generate_content(
            model=self.name, contents=prompt
        )
        return _parse_backend_response(resp.text or "", self.name)


class OpenAICompatLabelBackend:
    """LLM label oracle over any OpenAI-compatible ``/chat/completions`` endpoint.

    DeepSeek, vLLM, LM Studio, Ollama's OpenAI route, OpenAI itself. Standard library only
    (no SDK dependency). ``extra_body`` is merged into the request, e.g.
    ``{"thinking": {"type": "disabled"}}`` for DeepSeek, whose reasoning otherwise eats
    the token budget. The key comes from ``api_key``, then ``MERKEN_LABEL_API_KEY``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        extra_body: dict | None = None,
        timeout: float = 60.0,
        max_tokens: int = 120,
    ) -> None:
        resolved_key = api_key or os.environ.get("MERKEN_LABEL_API_KEY")
        if not resolved_key:
            raise RuntimeError("OpenAICompatLabelBackend needs api_key= or MERKEN_LABEL_API_KEY.")
        self._key = resolved_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._extra = dict(extra_body or {})
        self._timeout = timeout
        self._max_tokens = max_tokens
        self.name = model

    def label(self, text: str) -> Label:
        import urllib.request

        body = {
            "model": self.name,
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": _PROMPT.replace("{text}", text[:3000])}],
            **self._extra,
        }
        req = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            data = json.loads(resp.read())
        raw = data["choices"][0]["message"].get("content") or ""
        return _parse_backend_response(raw, self.name)


class LocalLLMLabelBackend:
    """Offline label oracle via ``LLMWriteDecider`` under the hood.

    Good when: labeling a batch privately (no API call), or when the
    caller wants to measure how much of Gemini's signal a local model
    captures on the same disagreement set. Uses the same chat-template
    + single-turn few-shot scoring that
    :class:`merken.classifiers.llm.LLMWriteDecider` uses, so the
    classifier's production behavior and the oracular comparison stay
    aligned.

    Rationale is synthesized mechanically from the logit scores --
    small local LMs are not reliable at producing useful natural-
    language justifications in a single forward pass, so we trade
    prose for honesty about what the oracle actually did.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ) -> None:
        # Import here so importing merken.labeling does not force
        # torch + transformers on every CLI invocation.
        from merken.classifiers.llm import LLMWriteDecider

        self._decider = LLMWriteDecider(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self.name = f"local-llm:{model_name}"

    def label(self, text: str) -> Label:
        from merken.policies import Event, WriteContext

        decision = self._decider.decide(
            Event(text=text), WriteContext(project="labeling")
        )
        # Map WriteDecider output onto the oracle schema. Local LLMs
        # via two-token scoring never emit UNCERTAIN -- the caller can
        # inspect confidence if they want an uncertainty proxy.
        kind = "DECISION" if decision.write else "NOISE"
        return Label(
            decision=kind,
            confidence=float(decision.confidence),
            rationale=(
                f"local-llm scored {decision.reason}; "
                f"policy={decision.policy}"
            ),
            backend=self.name,
        )


def _shadow_marker(reason: str) -> str | None:
    """Return the shadow marker in a reason string, or None."""
    if "|shadow_" not in reason:
        return None
    _, _, tail = reason.partition("|shadow_")
    marker, _, _ = tail.partition(":")
    return "shadow_" + marker if marker else None


def _parse_audit_body(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _label_title(event_title: str) -> str:
    """Stable title so re-labeling is idempotent per event."""
    return f"label:{event_title}"


def label_disagreements(
    mem: Memory,
    backend: LabelBackend,
    *,
    marker: str = "shadow_disagree",
    limit: int | None = None,
    max_audit_rows: int = 1000,
) -> Iterator[tuple[str, str, Label | None, str]]:
    """Label every disagreement not yet labeled.

    Yields ``(status, event_title, label_or_none, detail)`` tuples so
    the caller (CLI, notebook, whatever) can render progress without
    this function doing any printing itself.

    ``status`` is one of:
        - ``"labeled"`` (new label written)
        - ``"skipped"`` (already labeled, or missing event text)
        - ``"error"`` (backend raised; write not attempted)

    Labels are stored in the project's vstash under
    ``LABEL_COLLECTION``. Re-running is safe: the first pass processes
    all fresh disagreements, subsequent passes skip them.
    """
    existing = mem.search_labels(top_k=max_audit_rows)
    already_labeled: set[str] = set()
    for row in existing:
        title = getattr(row, "title", None) or ""
        if title.startswith("label:"):
            already_labeled.add(title.removeprefix("label:"))

    audit_rows = mem.audit(query=marker, top_k=max_audit_rows, fts_only=True)
    audit_rows = [r for r in audit_rows if marker in (r.text or "")]

    new_count = 0
    for row in audit_rows:
        fields = _parse_audit_body(row.text or "")
        event_title = fields.get("event_title") or ""
        reason = fields.get("reason") or ""
        if _shadow_marker(reason) != marker:
            yield ("skipped", event_title, None, "marker_mismatch")
            continue
        if not event_title:
            yield ("skipped", "", None, "no_event_title")
            continue
        if event_title in already_labeled:
            yield ("skipped", event_title, None, "already_labeled")
            continue

        text = fields.get("event_text_preview") or ""
        if not text:
            yield ("skipped", event_title, None, "no_text")
            continue

        try:
            label = backend.label(text)
        except Exception as exc:
            yield (
                "error",
                event_title,
                None,
                f"{exc.__class__.__name__}: {exc}",
            )
            continue

        body = json.dumps(
            {
                "decision": label.decision,
                "confidence": label.confidence,
                "rationale": label.rationale,
                "backend": label.backend,
                "shadow_marker": marker,
                "event_text_preview": text[:500],
            },
            ensure_ascii=False,
            indent=2,
        )
        mem.remember_label(event_title=event_title, body=body)
        already_labeled.add(event_title)

        yield ("labeled", event_title, label, label.decision)
        new_count += 1
        if limit is not None and new_count >= limit:
            return
