"""Audit log — every decision the loop makes, written through vstash.

CONSTITUTION §4.2: glass box. Every ``should_remember`` (and later
``should_recall``, ``should_consolidate``, ``should_forget``) decision lands
here, with inputs, the policy that fired, and the resulting write.

Storage choice (CONSTITUTION §10 #7): the same vstash backend, but isolated
in its own collection so audit rows can never leak into normal recall.

Audit writes deliberately bypass the merken loop — they call
``vstash.Memory.remember`` directly. Routing audit through the loop would be
recursive and would let a buggy decider silence itself.
"""

from __future__ import annotations

from datetime import datetime, timezone

from merken.policies.should_consolidate import ConsolidationDecision
from merken.policies.should_forget import ForgetDecision
from merken.policies.should_recall import RecallPlan
from merken.policies.types import Decision, Event

AUDIT_COLLECTION = "merken_audit"
AUDIT_LAYER = "audit"

# Tombstones live in their own collection so a user can query
# "what did I forget?" without grepping the general audit log.
# They preserve the full text of the forgotten event for
# unforgetting, which the regular audit rows do not.
TOMBSTONE_COLLECTION = "merken_tombstones"
TOMBSTONE_LAYER = "tombstone"

_PREVIEW_CHARS = 160


def format_recall_audit_row(
    query: str,
    plan: RecallPlan,
    reranker: str | None = None,
) -> tuple[str, str]:
    """Build a (title, body) pair for one recall audit entry.

    ``reranker`` names the post-retrieval reranker that ran on the
    candidates (``rerank_by_recency`` or a ``Reranker`` instance), so
    the audit row explains the final ordering, not just the plan.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    preview = " ".join(query.split())[:_PREVIEW_CHARS]
    layers = ",".join(f"{r.layer}:{r.top_k}" for r in plan.layers)
    body = (
        f"timestamp: {ts}\n"
        f"decision: should_recall\n"
        f"query_preview: {preview}\n"
        f"plan_layers: {layers}\n"
        f"reason: {plan.reason}\n"
        f"policy: {plan.policy}\n"
        f"reranker: {reranker or 'none'}\n"
    )
    title = f"audit:should_recall:{plan.reason}:{ts}"
    return title, body


def format_forget_audit_row(
    event_path: str,
    decision: ForgetDecision,
    derived_in_facts: list[str],
) -> tuple[str, str]:
    """Build a (title, body) pair for one should_forget audit entry.

    This is the *decision* row — every call to the decider writes one
    of these. The full event text is NOT here; it goes in the
    tombstone row (``format_tombstone_row``) when a tombstone fires,
    because audit rows are fire-and-forget summaries and tombstones
    are forgetting-safety nets.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    facts_str = ",".join(derived_in_facts) if derived_in_facts else "(none)"
    body = (
        f"timestamp: {ts}\n"
        f"decision: should_forget\n"
        f"event_path: {event_path}\n"
        f"tombstone: {decision.tombstone}\n"
        f"reason: {decision.reason}\n"
        f"policy: {decision.policy}\n"
        f"derived_in_facts: {facts_str}\n"
    )
    title = f"audit:should_forget:{decision.reason}:{ts}"
    return title, body


def format_tombstone_row(
    event_path: str,
    event_text: str,
    event_title: str | None,
    event_layer: str | None,
    event_tags: str | None,
    derived_in_facts: list[str],
    reason: str,
    policy: str,
) -> tuple[str, str]:
    """Build a (title, body) pair for one tombstone entry.

    The tombstone row preserves the FULL event text so a caller can
    later unforget by re-remembering it. Stored in the
    ``merken_tombstones`` collection, not the audit log.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    facts_str = ",".join(derived_in_facts) if derived_in_facts else "(none)"
    body = (
        f"tombstone_of: {event_path}\n"
        f"tombstoned_at: {ts}\n"
        f"reason: {reason}\n"
        f"policy: {policy}\n"
        f"original_title: {event_title or ''}\n"
        f"original_layer: {event_layer or ''}\n"
        f"original_tags: {event_tags or ''}\n"
        f"derived_in_facts: {facts_str}\n"
        f"---\n"
        f"{event_text}\n"
    )
    title = f"tombstone:{event_path}:{ts}"
    return title, body


def format_consolidate_audit_row(
    decision: ConsolidationDecision,
    n_events: int,
) -> tuple[str, str]:
    """Build a (title, body) pair for one consolidation audit entry."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = (
        f"timestamp: {ts}\n"
        f"decision: should_consolidate\n"
        f"proceed: {decision.proceed}\n"
        f"reason: {decision.reason}\n"
        f"policy: {decision.policy}\n"
        f"n_events: {n_events}\n"
    )
    title = f"audit:should_consolidate:{decision.reason}:{ts}"
    return title, body


def format_midloop_audit_row(observation, decision) -> tuple[str, str]:
    """Build a (title, body) pair for one midloop (should_intervene) audit row.

    Lives here for symmetry with the other format_*_audit_row helpers.
    Imports are deferred so that ``audit.py`` does not depend on the
    midloop module if a caller never uses it -- midloop is the only
    primitive whose dataclasses live in policies/ rather than at the
    package root.

    Body is plain key:value text so vstash FTS can find rows by
    ``task_id:abc123`` or ``step_index:7`` or ``shadow_disagree``.
    """
    import json
    ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    preview = " ".join((observation.output_text or "").split())[:_PREVIEW_CHARS]
    body = (
        f"timestamp: {ts}\n"
        f"decision: should_intervene\n"
        f"task_id: {observation.task_id}\n"
        f"step_id: {observation.step_id}\n"
        f"step_index: {observation.step_index}\n"
        f"task_category: {observation.task_category.value}\n"
        f"intervene: {decision.intervene}\n"
        f"action: {decision.action.value}\n"
        f"state: {decision.state.value}\n"
        f"confidence: {decision.confidence:.4f}\n"
        f"reason: {decision.reason}\n"
        f"policy: {decision.policy}\n"
        f"signals: {json.dumps(decision.signals, sort_keys=True)}\n"
        f"output_preview: {preview}\n"
    )
    title = (
        f"audit:should_intervene:{observation.task_id}:"
        f"step{observation.step_index:04d}:{ts}"
    )
    return title, body


def format_audit_row(event: Event, decision: Decision) -> tuple[str, str]:
    """Build a (title, body) pair for one audit entry.

    The body is plain text on purpose: it has to be human-readable when an
    operator runs ``mem.audit("why was this dropped")``, and it has to be
    indexable by both vstash's vector and FTS paths.

    The title includes microseconds *and* the event title so that two
    audit rows produced within the same second (common in batch
    ingestion) do not collide. Earlier versions used second-precision
    only, which silently dropped audit rows when the same policy
    produced identical reasons in quick succession.
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    preview = " ".join(event.text.split())[:_PREVIEW_CHARS]

    body = (
        f"timestamp: {ts}\n"
        f"decision: should_remember\n"
        f"write: {decision.write}\n"
        f"reason: {decision.reason}\n"
        f"policy: {decision.policy}\n"
        f"confidence: {decision.confidence}\n"
        f"event_layer: {event.layer}\n"
        f"event_title: {event.title or ''}\n"
        f"event_tags: {event.tags or ''}\n"
        f"event_text_preview: {preview}\n"
    )

    title_suffix = f":{event.title}" if event.title else ""
    title = f"audit:should_remember:{decision.reason}:{ts}{title_suffix}"
    return title, body
