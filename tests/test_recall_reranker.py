"""``Memory.recall`` with a pluggable ``Reranker`` and the over-fetch fix.

Before 2026-09, rerankers only ever saw ``LayeredRecaller``'s fixed
per-layer budgets (5 semantic + 3 episodic) no matter what ``top_k``
was — the over-fetch only widened the *merge* limit. These tests pin
the fixed behaviour: with a reranker configured, every layer fetches
``top_k * recall_overfetch`` candidates, the reranker sees them all,
its ordering wins, and the audit row names it.
"""

from __future__ import annotations

from pathlib import Path

from merken import Memory
from merken.consolidation import Fact
from merken.policies.should_consolidate import PeriodicConsolidator


class SpyReranker:
    name = "spy-reranker"

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self.last_hits: list = []

    def rerank(self, query, hits):
        self.seen.append((query, len(hits)))
        self.last_hits = list(hits)
        return list(reversed(hits))


def _fill(mem: Memory, n: int) -> None:
    for i in range(n):
        mem.remember(
            f"Episodic note number {i} about the payments retry policy and backoff.",
            title=f"ev_{i}",
        )


def test_reranker_sees_overfetched_candidates_and_orders_result(tmp_path: Path) -> None:
    spy = SpyReranker()
    with Memory(
        project="rerank", db=tmp_path / "r.db", reranker=spy, recall_overfetch=4
    ) as mem:
        _fill(mem, 12)
        hits = mem.recall("payments retry policy", top_k=3)

    assert len(hits) == 3
    assert len(spy.seen) == 1
    _, n_candidates = spy.seen[0]
    # 12 episodic docs, top_k=3 * overfetch=4 = 12 requested -> the reranker
    # must see more than the 8 the old fixed budgets would have produced.
    assert n_candidates > 8, n_candidates
    # The reranker's ordering (reversed) is what the caller gets, truncated to top_k.
    expected = [h.path for h in reversed(spy.last_hits)][:3]
    assert [h.path for h in hits] == expected


def test_reranker_audit_row_names_it(tmp_path: Path) -> None:
    spy = SpyReranker()
    with Memory(project="rerank_audit", db=tmp_path / "a.db", reranker=spy) as mem:
        _fill(mem, 3)
        mem.recall("payments", top_k=2)
        rows = mem.audit("should_recall", top_k=5)
    bodies = " ".join((r.text or "") for r in rows)
    assert "reranker: spy-reranker" in bodies


def test_no_reranker_audit_says_none(tmp_path: Path) -> None:
    with Memory(project="rerank_none", db=tmp_path / "n.db") as mem:
        _fill(mem, 2)
        mem.recall("payments", top_k=2)
        rows = mem.audit("should_recall", top_k=5)
    assert "reranker: none" in " ".join((r.text or "") for r in rows)


def test_explicit_layer_bypasses_reranker(tmp_path: Path) -> None:
    spy = SpyReranker()
    with Memory(project="rerank_layer", db=tmp_path / "l.db", reranker=spy) as mem:
        _fill(mem, 3)
        mem.recall("payments", top_k=2, layer="episodic")
    assert spy.seen == []


def test_consolidate_materialize_fn_controls_text_and_provenance(tmp_path: Path) -> None:
    """``materialize_fn`` may expel cluster members; the written fact
    carries only the provenance it returns. Text pair clusters above
    0.70 in both bge-small and multilingual-MiniLM (see
    tests/test_mcp_server.py::test_consolidate_force_builds_fact)."""

    def pick_first_only(cluster: list[tuple[str, str]]) -> Fact:
        id_, text = cluster[0]
        return Fact(text=f"CHOSEN: {text}", derived_from=[id_], cluster_size=1, method="test_v1")

    with Memory(
        project="mat_fn", db=tmp_path / "m.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
    ) as mem:
        mem.remember(
            "PostgreSQL 16 was chosen over SQLite for the analytics warehouse "
            "because of concurrent write requirements.",
            title="a",
        )
        mem.remember(
            "The analytics warehouse runs on PostgreSQL 16; SQLite was ruled "
            "out because of write concurrency concerns.",
            title="b",
        )
        result = mem.consolidate(method="embedding_v1", materialize_fn=pick_first_only)

    assert result.facts_written == 1
    fact = result.facts[0]
    assert fact.method == "test_v1"
    assert fact.text.startswith("CHOSEN: ")
    assert len(fact.derived_from) == 1
