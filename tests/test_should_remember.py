"""Phase 1 tests — the ``should_remember`` policy and the audit log.

These exercise the loop's first decision primitive end-to-end through the
``Memory`` wrapper. Each test uses an isolated tmp_path SQLite so they can
run in parallel without trampling each other.
"""

from __future__ import annotations

from pathlib import Path

from merken import (
    AlwaysWrite,
    Decision,
    Event,
    HeuristicWriteDecider,
    Memory,
    WriteContext,
)

# ----------------------------------------------------------------- pure policy


def _ctx() -> WriteContext:
    """Minimal WriteContext for tests. Historical name
    ``_ctx_with_recall`` was a leftover from when WriteContext
    exposed a recall callable — removed 2026-04-09 per the
    extending.md review."""
    return WriteContext(project="unit")


def test_heuristic_skips_empty() -> None:
    decider = HeuristicWriteDecider()
    decision = decider.decide(Event(text="   \n  "), _ctx())
    assert not decision.write
    assert decision.reason == "empty"


def test_heuristic_skips_too_short() -> None:
    decider = HeuristicWriteDecider(min_chars=10)
    decision = decider.decide(Event(text="hi there"), _ctx())
    assert not decision.write
    assert decision.reason.startswith("too_short")


def test_heuristic_skips_too_long() -> None:
    decider = HeuristicWriteDecider(max_chars=20)
    decision = decider.decide(
        Event(text="x" * 25),
        _ctx(),
    )
    assert not decision.write
    assert decision.reason.startswith("too_long")


def test_heuristic_writes_novel_when_no_hits() -> None:
    decider = HeuristicWriteDecider()
    decision = decider.decide(
        Event(text="A substantial sentence about geological time."),
        _ctx(),
    )
    assert decision.write
    assert decision.reason == "novel"


def test_heuristic_detects_exact_dup_in_process() -> None:
    text = "The user prefers the color teal in dashboards."

    decider = HeuristicWriteDecider()
    first = decider.decide(Event(text=text), _ctx())
    second = decider.decide(Event(text=text), _ctx())

    assert first.write
    assert first.reason == "novel"
    assert not second.write
    assert second.reason == "dup_exact"


def test_heuristic_normalizes_whitespace_for_dedup() -> None:
    decider = HeuristicWriteDecider()
    first = decider.decide(
        Event(text="the user prefers teal"),
        _ctx(),
    )
    second = decider.decide(
        Event(text="the   user\nprefers\tteal"),
        _ctx(),
    )

    assert first.write
    assert not second.write
    assert second.reason == "dup_exact"


def test_heuristic_state_is_per_instance() -> None:
    """A fresh decider does not see another decider's history."""
    text = "A novel sentence about quartz formation."

    decider_a = HeuristicWriteDecider()
    decider_b = HeuristicWriteDecider()

    a1 = decider_a.decide(Event(text=text), _ctx())
    a2 = decider_a.decide(Event(text=text), _ctx())
    b1 = decider_b.decide(Event(text=text), _ctx())

    assert a1.write
    assert not a2.write  # decider_a remembers
    assert b1.write  # decider_b is fresh


def test_always_write_baseline() -> None:
    decider = AlwaysWrite()
    decision = decider.decide(Event(text=""), _ctx())
    assert decision.write
    assert decision.reason == "always_write"
    assert isinstance(decision, Decision)


# --------------------------------------------------------- integration via Memory


def test_memory_writes_novel_event(tmp_path: Path) -> None:
    with Memory(project="phase1", db=tmp_path / "e.db") as mem:
        result = mem.remember(
            "Phase 1 ingested a novel sentence about marsupial migration."
        )

    assert result.written
    assert result.decision.reason == "novel"
    assert result.ingest is not None


def test_memory_skips_short_event(tmp_path: Path) -> None:
    with Memory(project="phase1", db=tmp_path / "e.db") as mem:
        result = mem.remember("hi")

    assert not result.written
    assert result.decision.reason.startswith("too_short")
    assert result.ingest is None


def test_memory_skips_exact_duplicate(tmp_path: Path) -> None:
    text = "The deployment of 2026-04-08 chose Postgres for concurrency reasons."

    with Memory(project="phase1", db=tmp_path / "e.db") as mem:
        first = mem.remember(text)
        second = mem.remember(text)

    assert first.written
    assert second.written is False
    assert second.decision.reason == "dup_exact"


def test_memory_audit_log_records_skips_and_writes(tmp_path: Path) -> None:
    with Memory(project="phase1", db=tmp_path / "e.db") as mem:
        mem.remember("A real fact worth keeping about the year 2026.")
        mem.remember("hi")  # too short
        mem.remember("   ")  # empty

        rows = mem.audit(query="should_remember", top_k=20)

    assert len(rows) >= 3, f"expected ≥3 audit rows, got {len(rows)}"
    bodies = " ".join((r.text or "") for r in rows)
    assert "novel" in bodies
    assert "too_short" in bodies
    assert "empty" in bodies


def test_memory_recall_does_not_leak_audit_rows(tmp_path: Path) -> None:
    with Memory(project="phase1", db=tmp_path / "e.db") as mem:
        mem.remember("A canary fact about wombats native to Tasmania.")
        # Force several skips to populate the audit collection
        for _ in range(3):
            mem.remember("hi")

        hits = mem.recall("wombats", top_k=10)

    assert hits, "expected the canary to come back"
    for hit in hits:
        body = (hit.text or "").lower()
        assert "decision: should_remember" not in body, (
            f"audit row leaked into recall: {body[:120]}"
        )


def test_memory_always_write_overrides_default(tmp_path: Path) -> None:
    """AlwaysWrite bypasses HeuristicWriteDecider's dedup rule.

    We use a duplicate-write test instead of a too_short test because
    vstash has its own guardrail (rejects texts < ~20 chars with
    status="empty"), and Memory.remember now surfaces that as a
    failed write. AlwaysWrite overriding the decider's "too_short"
    rule is moot when vstash will reject anyway.
    """
    text = (
        "A sentence long enough to pass vstash's ingest guardrail for "
        "the AlwaysWrite override test on 2026-04-09."
    )
    with Memory(
        project="phase1_baseline",
        db=tmp_path / "e.db",
        write_decider=AlwaysWrite(),
    ) as mem:
        r1 = mem.remember(text)
        r2 = mem.remember(text)  # HeuristicWriteDecider would skip as dup_exact

    assert r1.written
    assert r2.written  # AlwaysWrite accepts the duplicate
    assert r1.decision.reason == "always_write"
    assert r2.decision.reason == "always_write"
