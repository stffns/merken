"""Temporal reranking for recall results.

Post-retrieval reranker that boosts newer results within a result set.
See experiments/loop_quality/scenarios/knowledge_update.json for the
scenario that motivated this module.

Formula: reranked_score = score * (1 + temporal_weight * recency_fraction)

- Multiplicative: poor semantic matches don't get promoted just for being recent
- Bounded: max boost is score * temporal_weight (20% at default 0.2)
- Relative: uses min/max of the current result set, adapts to any time range
- Graceful: None timestamps get recency_fraction=0.0 (no penalty, no boost)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vstash import SearchResult


class Reranker(Protocol):
    """Post-retrieval reranker plugged into ``Memory.recall``.

    Receives the merged, deduped candidate list (over-fetched to
    ``top_k * overfetch``) and returns a new ordering. It may drop
    candidates; ``Memory.recall`` truncates to the caller's ``top_k``
    afterwards. Implementations must not mutate the hits.
    """

    name: str

    def rerank(self, query: str, hits: list[SearchResult]) -> list[SearchResult]: ...


def _parse_ts(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def rerank_by_recency(
    results: list[SearchResult],
    *,
    temporal_weight: float = 0.2,
) -> list[SearchResult]:
    """Re-sort results by score boosted with a recency signal.

    Returns a new list sorted by descending reranked score. The
    original SearchResult objects are not mutated — only their
    ordering changes.

    When temporal_weight is 0.0 or all timestamps are None/equal,
    the original ordering is preserved (stable sort).
    """
    if temporal_weight == 0.0 or len(results) <= 1:
        return list(results)

    timestamps = [_parse_ts(getattr(r, "added_at", None)) for r in results]
    valid_ts = [t for t in timestamps if t is not None]

    if len(valid_ts) < 2:
        return list(results)

    t_min = min(valid_ts)
    t_max = max(valid_ts)
    span = (t_max - t_min).total_seconds()

    if span == 0.0:
        return list(results)

    scored: list[tuple[float, int, SearchResult]] = []
    for i, r in enumerate(results):
        ts = timestamps[i]
        if ts is None:
            recency = 0.0
        else:
            recency = (ts - t_min).total_seconds() / span

        original_score = getattr(r, "score", 0.0) or 0.0
        boosted = original_score * (1.0 + temporal_weight * recency)
        # Negate i for stable descending sort (earlier items win ties)
        scored.append((boosted, -i, r))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[2] for s in scored]
