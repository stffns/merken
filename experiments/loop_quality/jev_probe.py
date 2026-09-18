"""Jev at merken's three decision points — ablation driver.

Runs a loop_quality scenario four ways, cumulatively, and reports the
runner's strict ``query_pass_rate`` plus ``hit@1`` (what an agent that
reads only the first hit would get):

    baseline   merken defaults (Heuristic write, longest-text fact, no rerank)
    write      + ChainedWriteDecider(Heuristic -> JevWriteDecider)
    write+cons + JevMaterializer as ``materialize_fn`` (current member = fact,
                 provenance restricted to DECISION-class versions of it)
    all        + JevReranker on recall (over-fetch, answers-the-query filter,
                 current state first)

Jev is reached through ``merken.classifiers.jev.JevClient`` (OpenRouter or
TypeSafe key from the environment). Every response is cached on disk under
``.jev_cache/`` keyed by request hash, so reruns are free and deterministic
— delete the directory to re-query. This is an experiment: network is
opt-in (CONSTITUTION §4.1), nothing here changes a default.

Usage::

    python -m experiments.loop_quality.jev_probe \\
        --scenario experiments/loop_quality/scenarios/knowledge_update_50topics.json

Results (merken@3953c818, jev-1.13-20260917, 2026-09-18) are recorded in
experiments/loop_quality/RESULTS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from experiments.loop_quality.runner import (
    _compute_cluster_purity,
    _fact_path_to_topic,
    _ingest_scenario,
    load_scenario,
)
from merken import Memory
from merken.classifiers.jev import (
    JevClient,
    JevMaterializer,
    JevReranker,
    JevWriteDecider,
)
from merken.policies.should_consolidate import PeriodicConsolidator
from merken.policies.should_recall import LayeredRecaller
from merken.policies.should_remember import ChainedWriteDecider, HeuristicWriteDecider

CACHE_DIR = Path(".jev_cache")
MODES = ("baseline", "write", "write+cons", "all")


class CachingJevClient(JevClient):
    """JevClient with an on-disk response cache and a cost/usage tally."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        super().__init__()
        self._cache = cache_dir
        self._cache.mkdir(exist_ok=True)
        self.stats = {"calls": 0, "cached": 0, "cost": 0.0, "tokens": 0, "model": ""}

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps(
            {"model": self.model, "state": state, "questions": questions}, sort_keys=True
        ).encode()
        key = self._cache / (hashlib.sha1(body).hexdigest() + ".json")  # noqa: S324
        if key.exists():
            self.stats["cached"] += 1
            return json.loads(key.read_text())
        out = super().decide(state, questions)
        self.stats["calls"] += 1
        self.stats["cost"] += out.get("usage", {}).get("cost", 0.0)
        self.stats["tokens"] += out.get("usage", {}).get("input_tokens", 0)
        self.stats["model"] = out.get("model", self.stats["model"])
        key.write_text(json.dumps(out))
        return out


def _contains(h: Any, q: Any) -> bool:
    return all(s.lower() in (h.text or "").lower() for s in q.expect_contains)


def _ok_strict(h: Any, q: Any, uni: dict[str, str]) -> bool:
    """The runner's criterion: topic-pure provenance + expected substrings."""
    return uni.get(h.path) == q.expect_topic and _contains(h, q)


def _ok_answer(h: Any, q: Any, p2t: dict[str, str], prov: dict[str, list[str]]) -> bool:
    """Lenient: right text, and provenance touches the expected topic
    (a fact from a mixed cluster still answers the user)."""
    if not _contains(h, q):
        return False
    if p2t.get(h.path) == q.expect_topic:
        return True
    return any(p2t.get(p) == q.expect_topic for p in prov.get(h.text or "", []))


def run(
    client: CachingJevClient,
    scenario_path: Path,
    mode: str,
    *,
    top_k: int = 5,
    threshold: float = 0.70,
    overfetch: int = 4,
) -> dict[str, Any]:
    sc = load_scenario(scenario_path)
    use_write = mode in ("write", "write+cons", "all")
    use_cons = mode in ("write+cons", "all")
    use_recall = mode == "all"

    write_dec = None
    if use_write:
        jev_write = JevWriteDecider(client)
        # Warm the cache in parallel; the sequential ingest then hits disk.
        client.decide_many([(e.text, jev_write._questions) for e in sc.events])
        write_dec = ChainedWriteDecider(HeuristicWriteDecider(), jev_write)
    reranker = JevReranker(client) if use_recall else None
    recall_dec = (
        LayeredRecaller(top_k_semantic=top_k * overfetch, top_k_episodic=top_k * overfetch)
        if use_recall
        else None
    )
    materialize_fn = JevMaterializer(client) if use_cons else None

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td, Memory(
        project=f"jev_{sc.name}",
        db=Path(td) / "m.db",
        write_decider=write_dec,
        recall_decider=recall_dec,
        consolidate_decider=PeriodicConsolidator(min_events=2),
        reranker=reranker,
        recall_overfetch=overfetch,
    ) as mem:
        p2t = _ingest_scenario(mem, sc)
        cons = mem.consolidate(
            method="embedding_v1",
            embedding_threshold=threshold,
            embedding_linkage="complete",
            materialize_fn=materialize_fn,
        )
        uni = {**p2t, **_fact_path_to_topic(cons.facts, p2t)}
        prov_by_text = {f.text: f.derived_from for f in cons.facts}

        strict = strict1 = answer = answer1 = 0
        fails: list[str] = []
        for q in sc.queries:
            hits = mem.recall(q.question, top_k=top_k)
            s_ok = [_ok_strict(h, q, uni) for h in hits]
            a_ok = [_ok_answer(h, q, p2t, prov_by_text) for h in hits]
            strict += any(s_ok)
            strict1 += bool(s_ok and s_ok[0])
            answer += any(a_ok)
            answer1 += bool(a_ok and a_ok[0])
            if not any(a_ok):
                fails.append(q.expect_topic)
        purity = _compute_cluster_purity(cons.facts, p2t)

    return {
        "scenario": sc.name,
        "mode": mode,
        "events": len(sc.events),
        "written": len(p2t),
        "noise_written": sum(1 for t in p2t.values() if t == "noise"),
        "facts": cons.facts_written,
        "purity": purity,
        "strict": strict,
        "strict_hit1": strict1,
        "answer": answer,
        "answer_hit1": answer1,
        "queries": len(sc.queries),
        "fails": fails,
        "elapsed_s": time.perf_counter() - t0,
    }


def format_row(r: dict[str, Any]) -> str:
    n = max(1, r["queries"])
    return (
        f"{r['scenario']:40} {r['mode']:11} written {r['written']:4}/{r['events']:4} "
        f"(noise {r['noise_written']:3})  facts {r['facts']:3}  purity {r['purity']:.0%}  "
        f"strict {r['strict']}/{r['queries']} ({r['strict'] / n:.0%}) hit@1 {r['strict_hit1']}  |  "
        f"answer {r['answer']}/{r['queries']} ({r['answer'] / n:.0%}) hit@1 {r['answer_hit1']}  "
        f"{r['elapsed_s']:.0f}s"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.loop_quality.jev_probe", description=__doc__)
    p.add_argument("--scenario", type=Path, action="append", required=True)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--overfetch", type=int, default=4)
    p.add_argument("--out", type=Path, default=None, help="write results JSON here")
    a = p.parse_args(argv)

    client = CachingJevClient()
    results = []
    for sp in a.scenario:
        for mode in a.modes.split(","):
            if mode not in MODES:
                print(f"unknown mode {mode!r}; expected one of {MODES}", file=sys.stderr)
                return 2
            r = run(client, sp, mode, top_k=a.top_k, overfetch=a.overfetch)
            results.append(r)
            print(format_row(r), flush=True)
            if r["fails"]:
                shown = ", ".join(r["fails"][:12]) + (" …" if len(r["fails"]) > 12 else "")
                print(f"{'':52} fails: {shown}")
    st = client.stats
    print(
        f"\njev={st['model'] or '(cached)'}  calls {st['calls']} (cached {st['cached']})  "
        f"tokens {st['tokens']}  cost ${st['cost']:.4f}"
    )
    if a.out:
        a.out.write_text(json.dumps({"results": results, "stats": st}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
