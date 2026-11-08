"""D1WriteDecider contract, with the model replaced by a stub (no torch, no download)."""

from __future__ import annotations

from merken.policies.d1_write import D1WriteDecider
from merken.policies.types import Event, WriteContext


class _Stub(D1WriteDecider):
    def __init__(self, p: float, **kw) -> None:
        super().__init__(**kw)
        self.p = p
        self.seen: list[str] = []

    def probabilities(self, texts: list[str]) -> list[float]:
        self.seen += texts
        return [self.p] * len(texts)


CTX = WriteContext(project="t")


def test_writes_above_threshold_with_its_probability_in_the_audit() -> None:
    d = _Stub(0.83).decide(Event(text="We moved the cache to Redis because of stale reads."), CTX)
    assert d.write and d.policy == "D1WriteDecider"
    assert d.reason == "d1_keep:0.830" and abs(d.confidence - 0.83) < 1e-9


def test_skips_below_threshold_with_confidence_in_the_skip() -> None:
    d = _Stub(0.2).decide(Event(text="pytest: 47 passed in 3.2s"), CTX)
    assert not d.write and abs(d.confidence - 0.8) < 1e-9


def test_threshold_is_configurable() -> None:
    assert _Stub(0.6, threshold=0.5).decide(Event(text="something specific"), CTX).write
    assert not _Stub(0.6).decide(Event(text="something specific"), CTX).write  # default 0.7


def test_empty_never_reaches_the_model() -> None:
    stub = _Stub(0.99)
    d = stub.decide(Event(text="   "), CTX)
    assert not d.write and d.reason == "empty" and stub.seen == []


def test_no_model_loaded_at_construction() -> None:
    assert D1WriteDecider()._model is None
