"""Tests for the LongMemEval runner — fixture-only, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.retrieval.longmemeval.dataset import Conversation, load_fixture
from experiments.retrieval.longmemeval.runner import (
    _ADAPTERS,
    bootstrap_ci,
    eval_baseline,
    format_result,
)

FIXTURE = (
    Path(__file__).parent.parent
    / "experiments"
    / "retrieval"
    / "longmemeval"
    / "fixtures"
    / "tiny.json"
)


# --------------------------------------------------------------------- dataset


def test_fixture_loads_three_conversations() -> None:
    convs = load_fixture(FIXTURE)
    assert len(convs) == 3
    assert all(isinstance(c, Conversation) for c in convs)
    assert convs[0].question_id == "q1"
    assert convs[0].n_sessions == 3
    assert convs[0].answer_session_ids == ["s1_db"]


def test_fixture_question_types_are_set() -> None:
    convs = load_fixture(FIXTURE)
    for conv in convs:
        assert conv.question_type == "single-session-user"


# ------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_all_hits() -> None:
    lo, hi = bootstrap_ci([True] * 50, n_iter=200)
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_all_misses() -> None:
    lo, hi = bootstrap_ci([False] * 50, n_iter=200)
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_brackets_mean() -> None:
    values = [True] * 30 + [False] * 70
    lo, hi = bootstrap_ci(values, n_iter=500, seed=42)
    mean = 0.30
    assert lo <= mean <= hi
    assert hi - lo > 0  # not collapsed


def test_bootstrap_ci_handles_empty() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


# ----------------------------------------------------------------- adapters available


def test_all_three_baselines_registered() -> None:
    assert set(_ADAPTERS) == {"vstash", "merken-always", "merken-heuristic"}


# --------------------------------------------------------------- end-to-end on fixture


@pytest.mark.parametrize(
    "baseline",
    ["vstash", "merken-always", "merken-heuristic"],
)
def test_baseline_runs_on_fixture(tmp_path: Path, baseline: str) -> None:
    convs = load_fixture(FIXTURE)
    result = eval_baseline(baseline, convs, top_k=5, db_dir=tmp_path)

    assert result.baseline == baseline
    assert result.n_questions == 3
    assert 0.0 <= result.r_at_k <= 1.0
    assert result.ci_low <= result.r_at_k <= result.ci_high
    assert len(result.per_question) == 3
    # Each question should have ingested all turns from its 3 sessions × 2 turns = 6.
    for q in result.per_question:
        assert q.ingest.attempted == 6


def test_merken_heuristic_writes_at_least_as_few_as_always(tmp_path: Path) -> None:
    """The default decider should never write *more* than AlwaysWrite."""
    convs = load_fixture(FIXTURE)
    always = eval_baseline("merken-always", convs, top_k=5, db_dir=tmp_path / "a")
    heur = eval_baseline("merken-heuristic", convs, top_k=5, db_dir=tmp_path / "h")

    written_always = sum(q.ingest.written for q in always.per_question)
    written_heur = sum(q.ingest.written for q in heur.per_question)
    assert written_heur <= written_always


def test_format_result_is_human_readable(tmp_path: Path) -> None:
    convs = load_fixture(FIXTURE)
    result = eval_baseline("merken-heuristic", convs, top_k=5, db_dir=tmp_path)
    line = format_result(result)
    assert "merken-heuristic" in line
    assert "R@5=" in line
    assert "CI=" in line
