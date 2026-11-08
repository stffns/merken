"""Write-time enrichment through the production path: ``Memory(enricher=LLMEnricher(...))``.

``write_enrich_probe.py`` (2026-09-19) enriched events *outside* merken, before
``remember``. This driver runs the same loop with the enrichment inside
``Memory.remember`` (``merken.enrichment``), so the number it prints is the one a
user of the library gets:

- write: ``ChainedWriteDecider(Heuristic -> JevWriteDecider)``
- enrich: ``LLMEnricher`` (DeepSeek ``deepseek-flash`` by default, thinking off),
  gated by ``jev_decision_gate`` so only DECISION events pay for a line
- consolidate: ``embedding_v1`` complete-link 0.70, ``JevMaterializer``
- recall: ``JevReranker``, over-fetch 20, top_k 5

Jev runs locally (``--local LiquidAI/d1-2B``, the default; ``--local ''`` for the
hosted API). Jev and LLM responses are cached under ``.jev_cache/``.

    DEEPSEEK_API_KEY=... python -m experiments.loop_quality.enrich_prod \\
        --scenario experiments/loop_quality/scenarios/knowledge_update_50topics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from experiments.loop_quality.jev_probe import CACHE_DIR, CachingJevClient, _ok_answer, _ok_strict
from experiments.loop_quality.retrieval_ceiling import KS, _rank_of_answer
from experiments.loop_quality.runner import (
    _compute_cluster_purity,
    _fact_path_to_topic,
    load_scenario,
)
from merken import LLMEnricher, Memory, jev_decision_gate
from merken.classifiers.jev import JevMaterializer, JevReranker, JevWriteDecider
from merken.policies.should_consolidate import PeriodicConsolidator
from merken.policies.should_recall import LayeredRecaller
from merken.policies.should_remember import ChainedWriteDecider, HeuristicWriteDecider


def cached_complete(enr: LLMEnricher, cache: Path = CACHE_DIR) -> Any:
    """Wrap the enricher's HTTP call with an on-disk cache keyed by (model, prompt)."""
    http = enr._http_complete  # noqa: SLF001 -- experiment harness

    def complete(prompt: str) -> str:
        key = cache / ("enrich-" + hashlib.sha1(f"{enr.model}\0{prompt}".encode()).hexdigest() + ".json")  # noqa: S324,E501
        if key.exists():
            return json.loads(key.read_text())["raw"]
        for attempt in range(4):
            try:
                raw = http(prompt)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * 2**attempt)
        key.write_text(json.dumps({"raw": raw}))
        return raw

    return complete


def run(sc: Any, client: CachingJevClient, enricher: LLMEnricher | None, *, top_k: int,
        overfetch: int) -> dict:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td, Memory(
        project=f"enrich_prod_{sc.name}", db=Path(td) / "m.db",
        write_decider=ChainedWriteDecider(HeuristicWriteDecider(), JevWriteDecider(client)),
        recall_decider=LayeredRecaller(
            top_k_semantic=top_k * overfetch, top_k_episodic=top_k * overfetch
        ),
        consolidate_decider=PeriodicConsolidator(min_events=2),
        reranker=JevReranker(client), recall_overfetch=overfetch,
        enricher=enricher,
    ) as mem:
        p2t: dict[str, str] = {}
        n_enriched = 0
        for e in sc.events:
            r = mem.remember(e.text, title=f"event_{e.id}", tags=f"topic:{e.topic}")
            if r.written and r.ingest is not None:
                p2t[r.ingest.source] = e.topic
                n_enriched += r.enrichment is not None
        cons = mem.consolidate(method="embedding_v1", embedding_threshold=0.70,
                               embedding_linkage="complete", materialize_fn=JevMaterializer(client))
        uni = {**p2t, **_fact_path_to_topic(cons.facts, p2t)}
        prov = {f.text: f.derived_from for f in cons.facts}
        ranks = []
        for q in sc.queries:
            hits = mem._vstash.search(  # noqa: SLF001 -- probe, not production
                q.question, top_k=200, collection=mem.collection, layer="episodic"
            )
            ranks.append(_rank_of_answer(list(hits), q, p2t))
        found = [r for r in ranks if r]
        ceiling = {k: sum(1 for r in found if r <= k) / max(1, len(ranks)) for k in KS}
        strict = strict1 = answer = answer1 = 0
        fails = []
        for q in sc.queries:
            hits = mem.recall(q.question, top_k=top_k)
            s_ok = [_ok_strict(h, q, uni) for h in hits]
            a_ok = [_ok_answer(h, q, p2t, prov) for h in hits]
            strict += any(s_ok)
            strict1 += bool(s_ok and s_ok[0])
            answer += any(a_ok)
            answer1 += bool(a_ok and a_ok[0])
            if not any(a_ok):
                fails.append(q.expect_topic)
        purity = _compute_cluster_purity(cons.facts, p2t)
    return {
        "written": len(p2t), "enriched": n_enriched, "facts": cons.facts_written,
        "purity": purity, "ceiling": ceiling, "missing": len(ranks) - len(found),
        "strict": strict, "strict_hit1": strict1, "answer": answer, "answer_hit1": answer1,
        "queries": len(sc.queries), "fails": fails, "elapsed_s": time.perf_counter() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.loop_quality.enrich_prod")
    p.add_argument("--scenario", type=Path, action="append", required=True)
    p.add_argument("--local", default="LiquidAI/d1-2B", help="'' = hosted Jev API")
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--overfetch", type=int, default=20)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("MERKEN_ENRICH_API_KEY"):
        print("DEEPSEEK_API_KEY not set", file=sys.stderr)
        return 2
    client = CachingJevClient(local=a.local or None)
    enricher = LLMEnricher(model=a.model, base_url=a.base_url, gate=jev_decision_gate(client))
    enricher._complete = cached_complete(enricher)  # noqa: SLF001
    results = {}
    for sp in a.scenario:
        sc = load_scenario(sp)
        variants = [("jev-loop", None)] if not a.no_baseline else []
        variants.append((f"jev-loop + enrich[{a.model}]", enricher))
        for label, enr in variants:
            r = run(sc, client, enr, top_k=a.top_k, overfetch=a.overfetch)
            results[f"{sc.name}/{label}"] = r
            n = max(1, r["queries"])
            ceil = "  ".join(f"@{k} {r['ceiling'][k]:.0%}" for k in KS)
            print(
                f"{sc.name:28} {label:34} written {r['written']} enriched {r['enriched']}  "
                f"purity {r['purity']:.0%}  ceiling {ceil} (missing {r['missing']})  "
                f"strict {r['strict']}/{r['queries']} ({r['strict'] / n:.0%}) hit@1 "
                f"{r['strict_hit1']}  answer {r['answer']}/{r['queries']} hit@1 "
                f"{r['answer_hit1']}  {r['elapsed_s']:.0f}s",
                flush=True,
            )
            if r["fails"]:
                print(f"{'':64} fails: {', '.join(r['fails'])}", flush=True)
    print(f"\nenricher {enricher.stats}   jev calls {client.stats['calls']} "
          f"(cached {client.stats['cached']})")
    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
