"""Phase 2 tests — ``should_consolidate`` decider + consolidation pipeline.

Split between pure unit tests (no vstash, no network) and integration
tests that exercise the full ``Memory.consolidate`` path.
"""

from __future__ import annotations

from pathlib import Path

from merken import (
    ConsolidateContext,
    ConsolidationResult,
    Fact,
    Memory,
    NeverConsolidate,
    PeriodicConsolidator,
)
from merken.consolidation import (
    _cosine,
    cluster_by_embedding,
    cluster_by_jaccard,
    cluster_by_recall,
    fact_fingerprint,
    jaccard,
    materialize_fact,
)


class _FakeHit:
    """Minimal stand-in for vstash.SearchResult for the pure-unit tests."""

    def __init__(self, path: str) -> None:
        self.path = path


# --------------------------------------------------------------- pure math


def test_jaccard_identical() -> None:
    a = {"hello", "world"}
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_half_overlap() -> None:
    a = {"the", "user", "likes", "teal"}
    b = {"the", "user", "likes", "purple"}
    # intersection=3, union=5
    assert jaccard(a, b) == 0.6


def test_jaccard_empty_both() -> None:
    assert jaccard(set(), set()) == 1.0


def test_jaccard_empty_one_side() -> None:
    assert jaccard({"a"}, set()) == 0.0


# --------------------------------------------------------------- clustering


def test_cluster_empty_input() -> None:
    assert cluster_by_jaccard([]) == []


def test_cluster_single_item() -> None:
    clusters = cluster_by_jaccard([("id1", "alone in the world")])
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_cluster_groups_similar_items() -> None:
    items = [
        ("a", "the user prefers postgres for concurrent writes"),
        ("b", "the user prefers postgres because concurrent writes"),
        ("c", "completely unrelated content about wombats"),
    ]
    clusters = cluster_by_jaccard(items, threshold=0.5)
    # Expect: {a, b} together, {c} alone
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_recall_empty() -> None:
    assert cluster_by_recall([], recall_fn=lambda q, k: []) == []


def test_cluster_by_recall_singleton() -> None:
    clusters = cluster_by_recall(
        [("p1", "alone")],
        recall_fn=lambda q, k: [_FakeHit("p1")],  # only self
    )
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_cluster_by_recall_groups_mutual_neighbors() -> None:
    """Two items that each recall the other should land in one cluster."""
    items = [("p1", "alpha"), ("p2", "beta"), ("p3", "gamma")]

    def fake_recall(query: str, top_k: int) -> list[_FakeHit]:
        # p1 ↔ p2 mutual; p3 alone
        if query == "alpha":
            return [_FakeHit("p1"), _FakeHit("p2")]
        if query == "beta":
            return [_FakeHit("p2"), _FakeHit("p1")]
        return [_FakeHit("p3")]

    clusters = cluster_by_recall(items, recall_fn=fake_recall, top_k=3)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_recall_transitive_link() -> None:
    """A→B and B→C means all three cluster (single-link transitive)."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c")]

    def fake_recall(query: str, top_k: int) -> list[_FakeHit]:
        if query == "a":
            return [_FakeHit("p2")]
        if query == "b":
            return [_FakeHit("p3")]
        return []

    clusters = cluster_by_recall(items, recall_fn=fake_recall, top_k=3)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_by_recall_ignores_unknown_paths() -> None:
    """Hits with paths outside the item set should be dropped."""
    items = [("p1", "text one"), ("p2", "text two")]

    def fake_recall(query: str, top_k: int) -> list[_FakeHit]:
        # p1's only non-self neighbor is 'p99' (not in our items)
        if query == "text one":
            return [_FakeHit("p1"), _FakeHit("p99")]
        return [_FakeHit("p2")]

    clusters = cluster_by_recall(items, recall_fn=fake_recall, top_k=3)
    sizes = sorted(len(c) for c in clusters)
    # p1 and p2 each isolated
    assert sizes == [1, 1]


def test_cosine_identical() -> None:
    assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0


def test_cosine_orthogonal() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_vector() -> None:
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cluster_by_embedding_empty() -> None:
    assert cluster_by_embedding([], embed_fn=lambda ts: []) == []


def test_cluster_by_embedding_groups_above_threshold() -> None:
    """Two items above threshold cluster; a third below stays alone."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c")]
    # Fake vectors: p1 and p2 nearly identical, p3 orthogonal.
    fake_vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.95, 0.31, 0.0],  # cos with [1,0,0] ≈ 0.95
        "c": [0.0, 0.0, 1.0],  # cos with both ≈ 0
    }
    clusters = cluster_by_embedding(
        items,
        embed_fn=lambda ts: [fake_vectors[t] for t in ts],
        threshold=0.7,
    )
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_embedding_respects_threshold() -> None:
    items = [("p1", "a"), ("p2", "b")]
    # cos = 0.6 — below 0.65 default, above 0.5
    vectors = {"a": [1.0, 0.0], "b": [0.6, 0.8]}
    tight = cluster_by_embedding(
        items, embed_fn=lambda ts: [vectors[t] for t in ts], threshold=0.65
    )
    loose = cluster_by_embedding(
        items, embed_fn=lambda ts: [vectors[t] for t in ts], threshold=0.5
    )
    assert len(tight) == 2  # each alone
    assert len(loose) == 1  # together


def test_cluster_by_embedding_single_link_cascades() -> None:
    """single-link: A~B and B~C above threshold groups all three via cascade,
    even when A~C itself is below threshold."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c")]
    # a~b ≈ 0.7, b~c ≈ 0.5, a~c = 0
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.7, 0.7, 0.0],
        "c": [0.0, 0.7, 0.7],
    }
    clusters = cluster_by_embedding(
        items,
        embed_fn=lambda ts: [vectors[t] for t in ts],
        threshold=0.4,
        linkage="single",
    )
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_by_embedding_complete_link_avoids_cascade() -> None:
    """complete-link: A~B and B~C above threshold do NOT group all three,
    because complete-link requires every cross-pair to be above threshold
    and A~C is below."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c")]
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.7, 0.7, 0.0],  # cos(a,b) ≈ 0.7 ✓
        "c": [0.0, 0.7, 0.7],  # cos(b,c) ≈ 0.5, cos(a,c) = 0
    }
    # threshold=0.45: single would cascade; complete should not because
    # a~c is 0 and b~c is only ~0.5 — once {a,b} merges, merging with
    # {c} would require min(cos(a,c), cos(b,c)) >= 0.45, which fails
    # (cos(a,c)=0).
    clusters = cluster_by_embedding(
        items,
        embed_fn=lambda ts: [vectors[t] for t in ts],
        threshold=0.45,
        linkage="complete",
    )
    sizes = sorted(len(c) for c in clusters)
    # {a,b} merges, {c} alone
    assert sizes == [1, 2]


def test_cluster_by_embedding_complete_link_full_merge_when_cohesive() -> None:
    """complete-link merges all three when every pair is above threshold."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c")]
    # All three mutually similar
    vectors = {
        "a": [1.0, 0.1, 0.1],
        "b": [0.9, 0.2, 0.1],
        "c": [0.95, 0.15, 0.05],
    }
    clusters = cluster_by_embedding(
        items,
        embed_fn=lambda ts: [vectors[t] for t in ts],
        threshold=0.9,
        linkage="complete",
    )
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_by_embedding_rejects_unknown_linkage() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown linkage"):
        cluster_by_embedding(
            [("p1", "a"), ("p2", "b")],
            embed_fn=lambda ts: [[1.0, 0.0], [0.9, 0.1]],
            linkage="bogus",
        )


def test_cluster_by_embedding_average_link_accepts_one_outlier() -> None:
    """Average-link's key behavior: a cluster with one weaker
    member still merges if the mean cross-pair is above threshold.
    Complete-link would reject the same merge."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c"), ("p4", "d")]
    # p1, p2, p3 are near-identical. p4 is weaker with all of them
    # but its AVERAGE similarity is still above 0.7.
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.99, 0.14, 0.0],   # cos ≈ 0.99 with a
        "c": [0.98, 0.20, 0.0],   # cos ≈ 0.98 with a
        "d": [0.80, 0.60, 0.0],   # cos ≈ 0.80 with a, slightly less with b/c
    }
    clusters = cluster_by_embedding(
        items,
        embed_fn=lambda ts: [vectors[t] for t in ts],
        threshold=0.75,
        linkage="average",
    )
    # Average-link should merge all 4 if mean cross-pairs stay above 0.75
    assert len(clusters) == 1
    assert len(clusters[0]) == 4


def test_cluster_by_embedding_average_link_still_rejects_cross_topic() -> None:
    """Average-link is not permissive like single-link. A cross-topic
    outlier whose cross-pairs are all clearly below threshold should
    NOT get pulled into a cluster."""
    items = [("p1", "a"), ("p2", "b"), ("p3", "c")]
    # a and b cluster; c is clearly unrelated
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.95, 0.31, 0.0],  # cos(a,b) ≈ 0.95
        "c": [0.0, 0.0, 1.0],    # cos(a,c) = 0, cos(b,c) = 0
    }
    clusters = cluster_by_embedding(
        items,
        embed_fn=lambda ts: [vectors[t] for t in ts],
        threshold=0.70,
        linkage="average",
    )
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_embedding_handles_embedder_failure() -> None:
    """An embedder that raises should leave each item as its own cluster."""
    def angry_embedder(texts: list[str]) -> list:
        raise RuntimeError("boom")

    clusters = cluster_by_embedding(
        [("p1", "x"), ("p2", "y"), ("p3", "z")],
        embed_fn=angry_embedder,
    )
    assert len(clusters) == 3
    assert all(len(c) == 1 for c in clusters)


def test_cluster_by_embedding_mismatched_vector_count_raises() -> None:
    """Defensive check: embedder must return one vector per text."""
    import pytest

    with pytest.raises(ValueError, match="returned"):
        cluster_by_embedding(
            [("p1", "x"), ("p2", "y")],
            embed_fn=lambda ts: [[1.0, 0.0]],  # only one vector
        )


def test_cluster_by_recall_survives_recall_failure() -> None:
    """A recall that raises should not crash clustering — item stays alone."""
    def angry_recall(query: str, top_k: int) -> list[_FakeHit]:
        raise RuntimeError("boom")

    clusters = cluster_by_recall(
        [("p1", "x"), ("p2", "y")],
        recall_fn=angry_recall,
    )
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 1]


def test_cluster_respects_threshold() -> None:
    items = [
        ("a", "alpha beta gamma"),
        ("b", "alpha delta epsilon"),  # only 1/5 overlap
    ]
    tight = cluster_by_jaccard(items, threshold=0.5)
    loose = cluster_by_jaccard(items, threshold=0.1)
    # Tight: each item in its own cluster
    assert len(tight) == 2
    # Loose: both in one cluster
    assert len(loose) == 1
    assert len(loose[0]) == 2


# --------------------------------------------------------------- fact build


def test_materialize_singleton_passthrough() -> None:
    fact = materialize_fact([("id1", "a single observation")])
    assert fact.method == "passthrough"
    assert fact.cluster_size == 1
    assert fact.text == "a single observation"
    assert fact.derived_from == ["id1"]


def test_materialize_cluster_concats() -> None:
    fact = materialize_fact([
        ("id1", "short one"),
        ("id2", "this is a longer observation about the subject"),
        ("id3", "medium length observation"),
    ])
    assert fact.method == "concat_v1"
    assert fact.cluster_size == 3
    assert fact.text.startswith("[observed 3×]")
    assert "longer observation" in fact.text  # anchor = longest
    assert fact.derived_from == ["id1", "id2", "id3"]


def test_fact_fingerprint_is_stable_and_order_independent() -> None:
    f1 = Fact(text="x", derived_from=["a", "b", "c"], cluster_size=3, method="concat_v1")
    f2 = Fact(text="x", derived_from=["c", "a", "b"], cluster_size=3, method="concat_v1")
    assert fact_fingerprint(f1) == fact_fingerprint(f2)
    # Different derived_from → different fingerprint
    f3 = Fact(text="x", derived_from=["a", "b", "d"], cluster_size=3, method="concat_v1")
    assert fact_fingerprint(f1) != fact_fingerprint(f3)


# ------------------------------------------------------- pure policy deciders


def _ctx() -> ConsolidateContext:
    return ConsolidateContext(project="unit")


def test_never_consolidate_always_says_no() -> None:
    d = NeverConsolidate().decide(999, _ctx())
    assert not d.proceed
    assert d.reason == "never_triggers_auto"


def test_periodic_below_threshold() -> None:
    d = PeriodicConsolidator(min_events=10).decide(5, _ctx())
    assert not d.proceed
    assert d.reason.startswith("too_few_events")


def test_periodic_at_threshold() -> None:
    d = PeriodicConsolidator(min_events=10).decide(10, _ctx())
    assert d.proceed
    assert d.reason == "enough_events:10"


def test_periodic_above_threshold() -> None:
    d = PeriodicConsolidator(min_events=10).decide(50, _ctx())
    assert d.proceed


# ---------------------------------------------------- integration with Memory


def test_consolidate_skips_when_below_threshold(tmp_path: Path) -> None:
    with Memory(
        project="consolidate_skip",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=5),
    ) as mem:
        mem.remember("A single observation about wombats.")
        mem.remember("Another note about wallabies.")

        result = mem.consolidate()

    assert isinstance(result, ConsolidationResult)
    assert result.skipped
    assert result.events_examined == 2
    assert result.facts_written == 0
    assert result.reason.startswith("too_few_events")


def test_consolidate_force_bypasses_decider(tmp_path: Path) -> None:
    with Memory(
        project="consolidate_force",
        db=tmp_path / "e.db",
        consolidate_decider=NeverConsolidate(),
    ) as mem:
        mem.remember("The user chose postgres for the analytics project.")
        mem.remember("User chose postgres for the analytics project due to concurrency.")
        mem.remember("Completely unrelated content about photosynthesis in plants.")

        result = mem.consolidate(
            force=True, method="jaccard_v1", jaccard_threshold=0.4
        )

    assert not result.skipped
    assert result.events_examined == 3
    # At least one cluster of size >= 2 should exist → at least one fact
    assert result.facts_written >= 1
    for fact in result.facts:
        assert fact.cluster_size >= 2


def test_consolidate_embedding_v1_clusters_paraphrased_events(tmp_path: Path) -> None:
    """The prueba del vaso: paraphrased sentences about the same topic
    should cluster under embedding_v1, even when Jaccard overlap is low."""
    with Memory(
        project="consolidate_embedding_v1",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
    ) as mem:
        # Two paraphrases of the same fact. Lexically disjoint except
        # for "postgres" and stopwords — Jaccard ≈ 0.22. Should cluster
        # under embedding-based consolidation.
        mem.remember(
            "The team picked postgres as the database for the new analytics service."
        )
        mem.remember(
            "Postgres was the engine chosen to back the analytics service "
            "that the team is building."
        )
        # And an unrelated distractor
        mem.remember(
            "Marketing approved the new brand colors for the summer launch campaign."
        )

        result = mem.consolidate(method="embedding_v1", embedding_threshold=0.65)

    assert not result.skipped
    assert result.method == "embedding_v1"
    assert result.facts_written == 1
    assert result.facts[0].cluster_size == 2


def test_consolidate_invalid_method_raises(tmp_path: Path) -> None:
    with Memory(
        project="consolidate_bad",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=1),
    ) as mem:
        mem.remember("a single event")
        try:
            mem.consolidate(method="bogus_v99", force=True)
        except ValueError as e:
            assert "bogus_v99" in str(e)
        else:
            raise AssertionError("expected ValueError for unknown method")


def test_consolidate_jaccard_v1_emits_fact_findable_in_semantic_layer(
    tmp_path: Path,
) -> None:
    """Near-duplicate text path: jaccard_v1 should produce findable facts."""
    with Memory(
        project="consolidate_real",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
    ) as mem:
        mem.remember("The user prefers the color teal for dashboards and charts.")
        mem.remember("User said teal is preferred for dashboards and charts.")
        mem.remember("The team meeting is scheduled for Tuesday at 3pm in the office.")
        mem.remember("Team meeting happens Tuesday at 3pm in the office.")
        mem.remember("Engineering chose Rust for the new performance module.")

        result = mem.consolidate(method="jaccard_v1", jaccard_threshold=0.4)

    assert not result.skipped
    # Expect 2 clusters of size 2 (teal, meeting) + 1 singleton (Rust skipped).
    assert result.facts_written == 2
    assert all(f.cluster_size == 2 for f in result.facts)

    # And the facts should be findable in the semantic layer.
    with Memory(project="consolidate_real", db=tmp_path / "e.db") as mem2:
        teal_hits = mem2.recall("color preference dashboards", top_k=5, layer="semantic")
        assert teal_hits
        assert any("teal" in (h.text or "").lower() for h in teal_hits)


def test_consolidate_is_idempotent(tmp_path: Path) -> None:
    """Running consolidate twice over the same episodic set doesn't double facts."""
    with Memory(
        project="consolidate_idem",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=2),
    ) as mem:
        mem.remember("The user prefers postgres for the analytics project.")
        mem.remember("User prefers postgres for the analytics project on this team.")

        r1 = mem.consolidate(method="jaccard_v1", jaccard_threshold=0.4)
        r2 = mem.consolidate(method="jaccard_v1", jaccard_threshold=0.4)

        semantic = [
            d for d in mem._vstash.list(collection="default", layer="semantic")
        ]

    assert r1.facts_written == 1
    assert r2.facts_written == 1  # same fact re-written, not a new one
    assert len(semantic) == 1


def test_consolidate_audit_row_recorded(tmp_path: Path) -> None:
    with Memory(
        project="consolidate_audit",
        db=tmp_path / "e.db",
        consolidate_decider=PeriodicConsolidator(min_events=10),
    ) as mem:
        mem.remember("a fact about the project")
        mem.consolidate()

        rows = mem.audit(query="should_consolidate", top_k=10)

    # The audit query searches for "should_consolidate" literal; at least
    # one row should match.
    assert rows
    bodies = " ".join((r.text or "") for r in rows)
    assert "should_consolidate" in bodies
    assert "too_few_events" in bodies
