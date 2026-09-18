"""Unit tests for ``merken.classifiers.jev`` — no network.

Every test injects a fake ``call_fn`` into ``JevClient`` that maps the
request body to a canned Jev response, so the deciders' logic (thresholds,
fail-open, provenance filtering, current-first ordering) is exercised
without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from merken.classifiers.jev import (
    JevClient,
    JevMaterializer,
    JevReranker,
    JevWriteDecider,
)
from merken.policies.types import Event, WriteContext

CTX = WriteContext(project="t")


def _choice(label: str, p: float, conf: float) -> dict[str, Any]:
    other = "NOISE" if label == "DECISION" else "DECISION"
    return {
        "answers": {
            "label": {
                "type": "choice",
                "choice": label,
                "probabilities": {label: p, other: round(1 - p, 3)},
                "confidence": conf,
            }
        }
    }


# ----------------------------------------------------------------- JevClient


def test_client_requires_key_or_call_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        JevClient()


def test_client_resolves_openrouter_then_typesafe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts")
    c = JevClient()
    assert "openrouter" in c.url and c.model.startswith("~typesafe/")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    c = JevClient()
    assert "typesafe.ai" in c.url and c.model == "jev-latest"


def test_client_decide_many_preserves_order() -> None:
    c = JevClient(call_fn=lambda body: {"echo": body["state"]})
    out = c.decide_many([(f"s{i}", {}) for i in range(20)], workers=4)
    assert [o["echo"] for o in out] == [f"s{i}" for i in range(20)]


# ------------------------------------------------------------ JevWriteDecider


def test_write_decider_confident_noise_skips() -> None:
    c = JevClient(call_fn=lambda body: _choice("NOISE", 0.97, 0.95))
    d = JevWriteDecider(c, confidence_threshold=0.6)
    out = d.decide(Event(text="Standup: nothing to report."), CTX)
    assert out.write is False
    assert out.reason.startswith("noise")
    assert out.confidence == 0.95
    assert out.policy == "jev-classifier"


def test_write_decider_confident_decision_writes() -> None:
    c = JevClient(call_fn=lambda body: _choice("DECISION", 0.99, 0.99))
    out = JevWriteDecider(c).decide(Event(text="Migrated from Kafka to NATS."), CTX)
    assert out.write is True
    assert "P(D)=0.990" in out.reason


def test_write_decider_low_confidence_writes_on_doubt() -> None:
    c = JevClient(call_fn=lambda body: _choice("NOISE", 0.6, 0.3))
    out = JevWriteDecider(c, confidence_threshold=0.6).decide(Event(text="hmm"), CTX)
    assert out.write is True
    assert out.reason.startswith("low_conf:0.300")


def test_write_decider_transport_error_writes() -> None:
    def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("read timed out")

    out = JevWriteDecider(JevClient(call_fn=boom)).decide(Event(text="anything"), CTX)
    assert out.write is True
    assert out.reason == "jev_error:TimeoutError"
    assert out.confidence == 0.0


def test_write_decider_truncates_state() -> None:
    seen: list[str] = []

    def spy(body: dict[str, Any]) -> dict[str, Any]:
        seen.append(body["state"])
        return _choice("DECISION", 0.9, 0.9)

    JevWriteDecider(JevClient(call_fn=spy), max_input_chars=10).decide(Event(text="x" * 50), CTX)
    assert seen == ["x" * 10]


# ---------------------------------------------------------------- JevReranker


@dataclass
class Hit:
    text: str
    path: str = ""


def _reranker_fake(answer_by_text: dict[str, float], current: str | None = None):
    """Fake Jev: per-text noul scores, and a fixed 'current' pick by text."""

    def call(body: dict[str, Any]) -> dict[str, Any]:
        q = body["questions"]
        if "answers" in q:
            return {"answers": {"answers": {"type": "noul", "noul": answer_by_text[body["state"]]}}}
        if "current" in q:
            keys = list(q["current"]["criteria"])
            state = body["state"]
            pick = next(k for k in keys if state[k] == current) if current else keys[0]
            return {"answers": {"current": {"type": "choice", "choice": pick,
                                            "probabilities": {}, "confidence": 1.0}}}
        raise AssertionError(f"unexpected questions {q}")

    return call


def test_reranker_filters_and_puts_current_first() -> None:
    hits = [Hit("noise a"), Hit("we used Redis"), Hit("noise b"), Hit("moved to Caffeine")]
    scores = {"noise a": 0.1, "we used Redis": 0.9, "noise b": 0.2, "moved to Caffeine": 0.9}
    r = JevReranker(JevClient(call_fn=_reranker_fake(scores, current="moved to Caffeine")))
    out = r.rerank("what cache do we use now?", hits)
    assert [h.text for h in out] == ["moved to Caffeine", "we used Redis"]


def test_reranker_keeps_original_order_when_nothing_answers() -> None:
    hits = [Hit("a"), Hit("b")]
    r = JevReranker(JevClient(call_fn=_reranker_fake({"a": 0.1, "b": 0.2})))
    assert [h.text for h in r.rerank("q", hits)] == ["a", "b"]


def test_reranker_transport_error_returns_hits_unchanged() -> None:
    def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise OSError("down")

    hits = [Hit("a"), Hit("b")]
    assert [h.text for h in JevReranker(JevClient(call_fn=boom)).rerank("q", hits)] == ["a", "b"]


def test_reranker_single_hit_no_calls() -> None:
    calls: list[Any] = []
    r = JevReranker(JevClient(call_fn=lambda b: calls.append(b) or {}))
    assert r.rerank("q", [Hit("only")])[0].text == "only"
    assert calls == []


# ------------------------------------------------------------ JevMaterializer


def _materializer_fake(current: str, same: dict[str, float], labels: dict[str, str]):
    def call(body: dict[str, Any]) -> dict[str, Any]:
        q = body["questions"]
        if "current" in q:
            keys = list(q["current"]["criteria"])
            pick = next(k for k in keys if body["state"][k] == current)
            return {"answers": {"current": {"type": "choice", "choice": pick,
                                            "probabilities": {}, "confidence": 1.0}}}
        if "same" in q:
            return {"answers": {"same": {"type": "noul", "noul": same[body["state"]["candidate"]]}}}
        if "label" in q:
            return _choice(labels[body["state"]], 0.9, 0.9)
        raise AssertionError(q)

    return call


def test_materializer_picks_current_and_expels_ticket() -> None:
    cluster = [
        ("p1", "We decided to use Redis for caching."),
        ("p2", "Ticket INFRA-1: Redis memory spike on node 3."),
        ("p3", "Reverted from Redis to Caffeine after p99 regressions."),
    ]
    fake = _materializer_fake(
        current="Reverted from Redis to Caffeine after p99 regressions.",
        same={cluster[0][1]: 0.9, cluster[1][1]: 0.8},  # the ticket even passes "same"
        labels={cluster[0][1]: "DECISION", cluster[1][1]: "NOISE"},
    )
    fact = JevMaterializer(JevClient(call_fn=fake))(cluster)
    assert fact.method == "jev_current_v1"
    assert fact.text == "[observed 2×] Reverted from Redis to Caffeine after p99 regressions."
    assert fact.derived_from == ["p3", "p1"]  # current first, ticket expelled by the DECISION rule
    assert fact.cluster_size == 2


def test_materializer_singleton_passthrough_no_calls() -> None:
    calls: list[Any] = []
    m = JevMaterializer(JevClient(call_fn=lambda b: calls.append(b) or {}))
    fact = m([("p1", "only one")])
    assert fact.method == "passthrough" and fact.derived_from == ["p1"]
    assert calls == []


def test_materializer_error_falls_back_to_longest_text() -> None:
    def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise OSError("down")

    cluster = [("p1", "short"), ("p2", "a much longer text here")]
    fact = JevMaterializer(JevClient(call_fn=boom))(cluster)
    assert fact.method == "concat_v1"
    assert fact.text.endswith("a much longer text here")
    assert fact.derived_from == ["p1", "p2"]


# ------------------------------------------------------------- env wiring


def _reload_shadow(monkeypatch: pytest.MonkeyPatch, **env: str | None):
    import importlib

    import merken._shadow

    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    importlib.reload(merken._shadow)
    return merken._shadow


def test_env_jev_without_key_falls_back_to_heuristic(monkeypatch, tmp_path) -> None:
    from merken import HeuristicWriteDecider, Memory

    _reload_shadow(
        monkeypatch, MERKEN_SHADOW="jev", MERKEN_PRIMARY=None,
        OPENROUTER_API_KEY=None, TYPESAFE_API_KEY=None,
    )
    with Memory(project="t", db=tmp_path / "j.db") as mem:
        assert isinstance(mem._write_decider, HeuristicWriteDecider)


def test_env_jev_primary_chains_after_heuristic(monkeypatch, tmp_path) -> None:
    from merken import ChainedWriteDecider, Memory

    _reload_shadow(
        monkeypatch, MERKEN_PRIMARY="jev", MERKEN_SHADOW=None,
        OPENROUTER_API_KEY="fake", MERKEN_PRIMARY_JEV_THRESHOLD="0.8",
    )
    with Memory(project="t", db=tmp_path / "k.db") as mem:
        dec = mem._write_decider
        assert isinstance(dec, ChainedWriteDecider)
        assert isinstance(dec._classifier, JevWriteDecider)
        assert dec._classifier._threshold == 0.8
        assert dec._classifier._client.model == "~typesafe/jev-latest"


def test_env_jev_shadow_annotates_only(monkeypatch, tmp_path) -> None:
    from merken import Memory, ShadowWriteDecider

    _reload_shadow(
        monkeypatch, MERKEN_SHADOW="jev", MERKEN_PRIMARY=None, TYPESAFE_API_KEY="fake",
        OPENROUTER_API_KEY=None,
    )
    with Memory(project="t", db=tmp_path / "s.db") as mem:
        assert isinstance(mem._write_decider, ShadowWriteDecider)
        assert isinstance(mem._write_decider._shadow, JevWriteDecider)
