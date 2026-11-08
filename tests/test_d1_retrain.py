"""Eval-gated retrain loop for the d1 write filter: split, gate, promote, teacher cache.

No model is loaded: everything here is the bookkeeping around training.
"""

from __future__ import annotations

import json

import pytest

from experiments.d1_write_filter import retrain as rt
from merken.labeling import Label, OpenAICompatLabelBackend


def _metrics(bacc: float, real: float = 1.0, guard: float = 0.9) -> dict:
    sc = {n: {"agreement": real, "n": 20} for n in rt.REAL}
    sc |= {n: {"agreement": guard, "n": 50} for n in rt.GUARD}
    return {"holdout": {"bacc": bacc}, "scenarios": sc}


# --------------------------------------------------------------------------- split


def test_split_is_stable_as_labels_arrive() -> None:
    rows = [{"text": f"event number {i}", "label": "NOISE", "keep": 0.1} for i in range(400)]
    _, _, hold = rt.split(rows)
    more = rows + [{"text": f"new event {i}", "label": "NOISE", "keep": 0.1} for i in range(400)]
    _, _, hold2 = rt.split(more)
    assert {r["text"] for r in hold} <= {r["text"] for r in hold2}


def test_split_buckets_are_disjoint_and_cover_everything() -> None:
    rows = [{"text": f"e{i}", "label": "DECISION", "keep": 0.9} for i in range(1000)]
    train, valid, hold = rt.split(rows)
    texts = [r["text"] for r in train + valid + hold]
    assert len(texts) == len(set(texts)) == 1000
    assert 0.10 < len(hold) / 1000 < 0.20
    assert 0.05 < len(valid) / 1000 < 0.15


def test_bucket_ignores_whitespace_and_case() -> None:
    assert rt.bucket("We  chose Redis") == rt.bucket("we chose redis")


# --------------------------------------------------------------------------- gate


def test_gate_promotes_a_strict_improvement() -> None:
    g = rt.gate(_metrics(0.80), _metrics(0.85))
    assert g.promote


def test_gate_rejects_a_holdout_regression() -> None:
    g = rt.gate(_metrics(0.85), _metrics(0.84))
    assert not g.promote and "holdout" in g.reasons[0]


def test_gate_min_gain() -> None:
    assert not rt.gate(_metrics(0.80), _metrics(0.805), min_gain=0.01).promote


def test_gate_never_trades_a_real_scenario() -> None:
    cur, cand = _metrics(0.80), _metrics(0.95)
    cand["scenarios"]["jay_vstash_snapshot"]["agreement"] = 0.95
    g = rt.gate(cur, cand)
    assert not g.promote and any("REAL jay_vstash_snapshot" in r for r in g.reasons)


def test_gate_tolerates_one_event_in_a_small_guard_scenario() -> None:
    cur, cand = _metrics(0.80), _metrics(0.90)
    cur["scenarios"]["markdown_tables_held_out"] |= {"agreement": 11 / 12, "n": 12}
    cand["scenarios"]["markdown_tables_held_out"] |= {"agreement": 10 / 12, "n": 12}
    g = rt.gate(cur, cand)
    assert g.promote and any("tolerated" in r for r in g.reasons)


def test_gate_rejects_two_lost_events_in_one_guard_scenario() -> None:
    cur, cand = _metrics(0.80), _metrics(0.90)
    cur["scenarios"]["markdown_tables_held_out"] |= {"agreement": 11 / 12, "n": 12}
    cand["scenarios"]["markdown_tables_held_out"] |= {"agreement": 9 / 12, "n": 12}
    assert not rt.gate(cur, cand).promote


def test_gate_rejects_a_pooled_guard_drop() -> None:
    cur, cand = _metrics(0.80, guard=0.90), _metrics(0.90, guard=0.88)  # -1 event each, -2 pts
    cand["scenarios"]["noisy_agent_stream"]["agreement"] = 0.86
    g = rt.gate(cur, cand)
    assert not g.promote and any("pooled" in r for r in g.reasons)


# --------------------------------------------------------------------------- promote


def _adapter(path, tag: str):
    path.mkdir(parents=True)
    (path / "adapter.safetensors").write_text(tag)
    (path / "lora.json").write_text("{}")
    return path


def test_promote_keeps_the_previous_adapter(tmp_path) -> None:
    final = _adapter(tmp_path / "current", "old")
    rt.promote(_adapter(tmp_path / "cand", "new"), final)
    assert (final / "adapter.safetensors").read_text() == "new"
    assert (tmp_path / "current.old" / "adapter.safetensors").read_text() == "old"


def test_promote_rolls_back_on_failure(tmp_path, monkeypatch) -> None:
    final = _adapter(tmp_path / "current", "old")
    cand = _adapter(tmp_path / "cand", "new")

    def boom(src, dst):
        (dst / "partial").mkdir(parents=True)  # a half-finished copy
        raise OSError("disk full")

    monkeypatch.setattr(rt.shutil, "copytree", boom)
    with pytest.raises(OSError, match="disk full"):
        rt.promote(cand, final)
    assert (final / "adapter.safetensors").read_text() == "old"
    assert not (final / "partial").exists()


def test_promote_into_empty_slot(tmp_path) -> None:
    rt.promote(_adapter(tmp_path / "cand", "new"), tmp_path / "current")
    assert (tmp_path / "current" / "adapter.safetensors").read_text() == "new"
    assert not (tmp_path / "current.old").exists()


# --------------------------------------------------------------------------- teacher


class _Counting:
    name = "fake-teacher"

    def __init__(self) -> None:
        self.calls = 0

    def label(self, text: str) -> Label:
        self.calls += 1
        return Label("DECISION", 0.9, "because", self.name)


def test_cached_backend_never_pays_twice(tmp_path) -> None:
    inner = _Counting()
    path = tmp_path / "cache.jsonl"
    a = rt.CachedBackend(inner, path)
    a.label("we moved to Redis")
    a.label("we moved to Redis")
    b = rt.CachedBackend(inner, path)  # a new process reads the cache back
    lab = b.label("we moved to Redis")
    assert inner.calls == 1 and b.hits == 1
    assert lab.decision == "DECISION" and lab.confidence == 0.9


def test_openai_compat_backend_parses_and_sends_extra_body(monkeypatch) -> None:
    sent = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            body = {
                "choices": [
                    {"message": {"content": "LABEL: NOISE | CONFIDENCE: 0.8 | REASON: status"}}
                ]
            }
            return json.dumps(body).encode()

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        sent["auth"] = req.headers["Authorization"]
        return _Resp()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    be = OpenAICompatLabelBackend(
        model="deepseek-flash",
        base_url="https://api.example.com/",
        api_key="k",
        extra_body={"thinking": {"type": "disabled"}},
    )
    lab = be.label("pytest: 47 passed")
    assert lab.decision == "NOISE" and lab.confidence == 0.8 and lab.backend == "deepseek-flash"
    assert sent["url"] == "https://api.example.com/chat/completions"
    assert sent["body"]["thinking"] == {"type": "disabled"} and sent["body"]["temperature"] == 0
    assert sent["auth"] == "Bearer k"


def test_openai_compat_backend_requires_a_key(monkeypatch) -> None:
    monkeypatch.delenv("MERKEN_LABEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="api_key"):
        OpenAICompatLabelBackend(model="m", base_url="http://x")
