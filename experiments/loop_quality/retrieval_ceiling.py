"""Where does the correct event sit in vstash's ranking? (retrieval ceiling probe)

Silt's rule: before proposing an algorithm, look at the distribution.
After the Jev ablation (RESULTS.md, 2026-09-18) the failures left on
``knowledge_update_50topics`` were all "the right event is not in the
candidates". This probe isolates retrieval: no consolidation, no
reranker, just ``vstash.Memory.search`` on the episodic layer, and for
every query the rank of the *current* event (topic match +
``expect_contains``) under a grid of search knobs vstash already has:
``retrieval_mode``, ``vec_weight``/``fts_weight``, ``mmr_lambda``.

Reports recall@5/20/40/100 and the median rank per config, on two stores:
everything written (``AlwaysWrite``) and the Jev-filtered store
(``Chained(Heuristic -> JevWriteDecider)``, cached responses).

Usage::

    python -m experiments.loop_quality.retrieval_ceiling \\
        --scenario experiments/loop_quality/scenarios/knowledge_update_50topics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from experiments.loop_quality.jev_probe import CachingJevClient
from experiments.loop_quality.runner import _ingest_scenario, load_scenario
from merken import Memory
from merken.classifiers.jev import JevWriteDecider
from merken.policies.should_remember import (
    AlwaysWrite,
    ChainedWriteDecider,
    HeuristicWriteDecider,
)

KS = (5, 20, 40, 100)

CONFIGS: dict[str, dict[str, Any]] = {
    "hybrid (default, mmr=0.5)": {},
    "hybrid mmr=1.0": {"mmr_lambda": 1.0},
    "hybrid mmr=0.0": {"mmr_lambda": 0.0},
    "vec_only": {"retrieval_mode": "vec_only"},
    "vec_only mmr=1.0": {"retrieval_mode": "vec_only", "mmr_lambda": 1.0},
    "fts_only": {"retrieval_mode": "fts_only"},
    "hybrid vec=0.8 fts=0.2 mmr=1.0": {"vec_weight": 0.8, "fts_weight": 0.2, "mmr_lambda": 1.0},
    "hybrid vec=0.5 fts=0.5 mmr=1.0": {"vec_weight": 0.5, "fts_weight": 0.5, "mmr_lambda": 1.0},
    "hybrid vec=0.2 fts=0.8 mmr=1.0": {"vec_weight": 0.2, "fts_weight": 0.8, "mmr_lambda": 1.0},
}


def _rank_of_answer(hits: list[Any], q: Any, p2t: dict[str, str]) -> int | None:
    for i, h in enumerate(hits):
        if p2t.get(h.path) != q.expect_topic:
            continue
        if all(s.lower() in (h.text or "").lower() for s in q.expect_contains):
            return i + 1
    return None


def probe(mem: Memory, sc: Any, p2t: dict[str, str], *, top_k: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, kw in CONFIGS.items():
        ranks: list[int | None] = []
        for q in sc.queries:
            try:
                hits = mem._vstash.search(  # noqa: SLF001 — probe, not production
                    q.question, top_k=top_k, collection=mem.collection, layer="episodic", **kw
                )
            except TypeError as e:  # knob not supported by this vstash version
                out[name] = {"error": str(e)}
                break
            ranks.append(_rank_of_answer(list(hits), q, p2t))
        else:
            found = [r for r in ranks if r is not None]
            out[name] = {
                "recall": {k: sum(1 for r in found if r <= k) / len(ranks) for k in KS},
                "median_rank": statistics.median(found) if found else None,
                "missing": len(ranks) - len(found),
                "ranks": ranks,
            }
    return out


def fmt(name: str, r: dict[str, Any], n: int) -> str:
    if "error" in r:
        return f"{name:34} n/a ({r['error'][:60]})"
    rec = "  ".join(f"@{k} {r['recall'][k]:.0%}" for k in KS)
    med = "—" if r["median_rank"] is None else f"{r['median_rank']:.0f}"
    return f"{name:34} {rec}   median rank {med:>3}   not in top-{n}: {r['missing']}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.loop_quality.retrieval_ceiling")
    p.add_argument("--scenario", type=Path, action="append", required=True)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--no-jev", action="store_true", help="skip the Jev-filtered store")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    results: dict[str, Any] = {}
    stores = [("all-written", lambda: AlwaysWrite())]
    if not a.no_jev:
        client = CachingJevClient()
        def jev_chain() -> ChainedWriteDecider:
            return ChainedWriteDecider(HeuristicWriteDecider(), JevWriteDecider(client))

        stores.append(("jev-filtered", jev_chain))
    for sp in a.scenario:
        sc = load_scenario(sp)
        for store_name, make_decider in stores:
            with tempfile.TemporaryDirectory() as td, Memory(
                project=f"ceiling_{sc.name}", db=Path(td) / "m.db", write_decider=make_decider()
            ) as mem:
                p2t = _ingest_scenario(mem, sc)
                n_noise = sum(1 for t in p2t.values() if t == "noise")
                print(
                    f"\n== {sc.name} · store={store_name} · {len(p2t)} docs "
                    f"({n_noise} noise) · {len(sc.queries)} queries · top_k={a.top_k}"
                )
                r = probe(mem, sc, p2t, top_k=a.top_k)
                for name, rr in r.items():
                    print(fmt(name, rr, a.top_k))
                results[f"{sc.name}/{store_name}"] = r
    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
