"""Write-time enrichment (``merken.enrichment``) and its wiring in ``Memory.remember``."""

from __future__ import annotations

from pathlib import Path

import pytest

from merken import LLMEnricher, Memory
from merken.enrichment import (
    Enricher,
    apply_enrichment,
    jev_decision_gate,
    strip_enrichment,
)

EVENT = "Migrated the metrics stack from Prometheus/Thanos to Datadog last sprint."
LINE = "Monitoring platform: Datadog (replaced Prometheus/Thanos)"


class FixedEnricher:
    name = "fixed"

    def __init__(self, line: str | None = LINE) -> None:
        self.line = line
        self.seen: list[str] = []

    def enrich(self, text: str) -> str | None:
        self.seen.append(text)
        return self.line


class BrokenEnricher:
    name = "broken"

    def enrich(self, text: str) -> str | None:
        raise TimeoutError("llm down")


# ------------------------------------------------------------- format


def test_apply_and_strip_round_trip() -> None:
    stored = apply_enrichment(EVENT, LINE)
    assert stored.startswith(f"» {LINE} «")
    assert stored.endswith(EVENT)
    assert strip_enrichment(stored) == EVENT


def test_strip_survives_chunk_join() -> None:
    # Memory's dedup hydration joins chunks with spaces, losing the newlines.
    joined = " ".join(apply_enrichment(EVENT, LINE).split("\n\n"))
    assert strip_enrichment(joined) == EVENT


def test_empty_line_stores_raw_text() -> None:
    assert apply_enrichment(EVENT, None) == EVENT
    assert apply_enrichment(EVENT, "   ") == EVENT


def test_markers_inside_the_line_cannot_break_stripping() -> None:
    stored = apply_enrichment(EVENT, "weird « line » here")
    assert strip_enrichment(stored) == EVENT


def test_strip_is_noop_on_plain_text() -> None:
    assert strip_enrichment(EVENT) == EVENT
    assert strip_enrichment("» not an enrichment") == "» not an enrichment"


# ------------------------------------------------------------- LLMEnricher


def test_llm_enricher_takes_first_nonempty_line_and_satisfies_protocol() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return f"\n  {LINE}  \nsecond line ignored"

    enr = LLMEnricher(complete_fn=complete)
    assert isinstance(enr, Enricher)
    assert enr.enrich(EVENT) == LINE
    assert prompts[0].endswith(EVENT)
    assert enr.stats["calls"] == 1


def test_llm_enricher_empty_content_is_none() -> None:
    enr = LLMEnricher(complete_fn=lambda p: "")
    assert enr.enrich(EVENT) is None
    assert enr.stats["empty"] == 1


def test_llm_enricher_gate_skips_the_call() -> None:
    called = []
    enr = LLMEnricher(complete_fn=lambda p: called.append(p) or LINE, gate=lambda t: False)
    assert enr.enrich(EVENT) is None
    assert called == []
    assert enr.stats["gated_out"] == 1


def test_llm_enricher_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERKEN_ENRICH_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="api_key"):
        LLMEnricher()


def test_llm_enricher_disables_deepseek_thinking_by_default() -> None:
    enr = LLMEnricher(api_key="k")
    assert enr._extra == {"thinking": {"type": "disabled"}}  # noqa: SLF001


class _FakeJev:
    def __init__(self, choice: str | None) -> None:
        self.choice = choice

    def decide(self, state, questions):
        if self.choice is None:
            raise ConnectionError("jev down")
        return {"answers": {"label": {"choice": self.choice}}}


def test_jev_gate_enriches_only_decisions_and_fails_open() -> None:
    assert jev_decision_gate(_FakeJev("DECISION"))(EVENT) is True
    assert jev_decision_gate(_FakeJev("NOISE"))(EVENT) is False
    assert jev_decision_gate(_FakeJev(None))(EVENT) is True


# ------------------------------------------------------------- Memory wiring


def test_remember_stores_line_on_top_and_reports_it(tmp_path: Path) -> None:
    enr = FixedEnricher()
    with Memory(project="enrich", db=tmp_path / "e.db", enricher=enr) as mem:
        r = mem.remember(EVENT, title="ev")
        assert r.written and r.enrichment == LINE
        assert enr.seen == [EVENT]  # the enricher sees the raw event
        hits = mem.recall("what monitoring platform do we use?", top_k=3)
        assert hits and "Monitoring platform: Datadog" in hits[0].text


def test_enricher_not_called_for_skipped_writes(tmp_path: Path) -> None:
    enr = FixedEnricher()
    with Memory(project="enrich", db=tmp_path / "e.db", enricher=enr) as mem:
        assert mem.remember(EVENT).written
        dup = mem.remember(EVENT)
        assert not dup.written
        assert enr.seen == [EVENT]


def test_enricher_failure_stores_raw_text(tmp_path: Path) -> None:
    with Memory(project="enrich", db=tmp_path / "e.db", enricher=BrokenEnricher()) as mem:
        r = mem.remember(EVENT, title="ev")
        assert r.written and r.enrichment is None
        hits = mem.recall("Prometheus Thanos Datadog", top_k=3)
        assert hits and hits[0].text.strip() == EVENT


def test_dedup_hydration_strips_enrichment_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    with Memory(project="enrich", db=db, enricher=FixedEnricher()) as mem:
        assert mem.remember(EVENT).written
    # A fresh instance hydrates its dedup set from the stored (enriched) rows;
    # the raw event must still be recognised as a duplicate.
    with Memory(project="enrich", db=db, enricher=FixedEnricher()) as mem:
        assert not mem.remember(EVENT).written


def test_consolidation_clusters_on_raw_text_but_facts_keep_the_line(tmp_path: Path) -> None:
    # Two unrelated events share one long category line. Clustering on the stored
    # text would merge them (most tokens in common); clustering on the raw events
    # must not. Two near-identical events must still merge, and the fact keeps the line.
    line = " ".join(f"shared{i}" for i in range(60))
    enr = FixedEnricher(line)
    a = "Kafka consumer lag alerting moved to the payments on-call rotation this week."
    b = "Kafka consumer lag alerting moved to the payments on-call rotation, effective now."
    c = "Frontend bundle size budget raised to 300 KB after the design system migration."
    with Memory(project="enrich", db=tmp_path / "e.db", enricher=enr) as mem:
        for t in (a, b, c):
            assert mem.remember(t).written
        res = mem.consolidate(method="jaccard_v1", jaccard_threshold=0.5, force=True)
        assert res.facts_written == 1
        fact = res.facts[0]
        assert len(fact.derived_from) == 2
        assert line in fact.text


def test_materializer_sees_raw_events(tmp_path: Path) -> None:
    from merken.consolidation import materialize_fact

    seen: list[list[str]] = []

    def spy(cluster):
        seen.append([text for _, text in cluster])
        return materialize_fact(cluster)

    a = "Kafka consumer lag alerting moved to the payments on-call rotation this week."
    b = "Kafka consumer lag alerting moved to the payments on-call rotation, effective now."
    with Memory(project="enrich", db=tmp_path / "e.db", enricher=FixedEnricher()) as mem:
        for t in (a, b):
            assert mem.remember(t).written
        res = mem.consolidate(method="jaccard_v1", materialize_fn=spy, force=True)
    assert seen and all("»" not in t for t in seen[0])
    assert res.facts[0].text.count(LINE) == 1
