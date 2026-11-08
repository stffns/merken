"""The other half of the idea: merken *writes* the memory, not just filters it.

The retrieval-ceiling probe showed the remaining ``knowledge_update``
failures are a vocabulary gap: queries are phrased at category level
("what monitoring platform do we use?") while events are instance level
("Migrated from Prometheus/Thanos to Datadog…"). No lexical overlap, and
small embedders do not bridge it (bge-base, nomic: +0–2pp). A retriever
cannot fix what the writer never said.

This probe enriches events **at write time**: Jev decides which events
are DECISION-class (so the LLM is only paid for signal), and a cheap LLM
writes one category-level line — "<category>: <current choice>
(<what it replaced>)" — that is stored *with* the event. The LLM never
sees a query. Then the full Jev loop (write filter, materializer,
reranker) runs as in ``jev_probe.py`` and we compare against it.

LLM transport: OpenRouter chat completions (``OPENROUTER_API_KEY``),
model from ``--llm`` (default ``openai/gpt-4.1-mini``; reasoning models
return empty content under a small ``max_tokens``). Responses are
cached under ``.jev_cache/`` like Jev's.

Usage::

    python -m experiments.loop_quality.write_enrich_probe \\
        --scenario experiments/loop_quality/scenarios/knowledge_update_50topics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.loop_quality.jev_probe import (
    CACHE_DIR,
    CachingJevClient,
    _ok_answer,
    _ok_strict,
)
from experiments.loop_quality.retrieval_ceiling import KS, _rank_of_answer
from experiments.loop_quality.runner import (
    _compute_cluster_purity,
    _fact_path_to_topic,
    load_scenario,
)
from merken import Memory
from merken.classifiers.jev import JevMaterializer, JevReranker, JevWriteDecider
from merken.policies.should_consolidate import PeriodicConsolidator
from merken.policies.should_recall import LayeredRecaller
from merken.policies.should_remember import ChainedWriteDecider, HeuristicWriteDecider

OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"

ENRICH_PROMPT = (
    "You index engineering notes for later retrieval. Write ONE line, at most 25 words, "
    "that states the general category this note answers and the current choice, so that "
    "a question phrased generically (e.g. 'what monitoring platform do we use?') would "
    "match it. Format: '<Category>: <current choice> (replaced <previous>, if any)'. "
    "Use the note's own facts only. No preamble.\n\nNote:\n"
)


class Enricher:
    def __init__(self, api_key: str, model: str, cache_dir: Path = CACHE_DIR) -> None:
        self._key = api_key
        self.model = model
        self._cache = cache_dir
        self.stats = {"calls": 0, "cached": 0, "cost": 0.0}

    def one_line(self, text: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": ENRICH_PROMPT + text}],
            "max_tokens": 120,
            "temperature": 0,
            # Reasoning models (DeepSeek v4.x, gpt-oss) otherwise spend the whole
            # budget thinking and return content=None. OpenRouter-level switch.
            "reasoning": {"enabled": False},
        }
        raw = json.dumps(body, sort_keys=True).encode()
        key = self._cache / ("llm-" + hashlib.sha1(raw).hexdigest() + ".json")  # noqa: S324
        if key.exists():
            self.stats["cached"] += 1
            return json.loads(key.read_text())["line"]
        req = urllib.request.Request(
            OPENROUTER_CHAT, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                    out = json.load(resp)
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * 2**attempt)
        content = out["choices"][0]["message"].get("content") or ""
        line = (content.strip().splitlines() or [""])[0].strip()
        cost = float(out.get("usage", {}).get("cost", 0.0) or 0.0)
        self.stats["calls"] += 1
        self.stats["cost"] += cost
        key.write_text(json.dumps({"line": line, "cost": cost}))
        return line


def run(
    sc: Any,
    client: CachingJevClient,
    enricher: Enricher | None,
    *,
    top_k: int,
    overfetch: int,
) -> dict:
    jev_write = JevWriteDecider(client)
    client.decide_many([(e.text, jev_write._questions) for e in sc.events])  # warm cache

    # Enrich only what Jev calls DECISION (that is the "decide when to spend" part).
    enriched: dict[str, str] = {}
    n_llm = 0
    if enricher is not None:
        todo = [e for e in sc.events if jev_write.label(e.text)["choice"] == "DECISION"]
        with ThreadPoolExecutor(max_workers=8) as ex:
            lines = list(ex.map(lambda e: enricher.one_line(e.text), todo))
        for e, line in zip(todo, lines, strict=True):
            enriched[e.id] = f"{line}\n{e.text}"
        n_llm = len(todo)

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td, Memory(
        project=f"enrich_{sc.name}", db=Path(td) / "m.db",
        write_decider=ChainedWriteDecider(HeuristicWriteDecider(), jev_write),
        recall_decider=LayeredRecaller(
            top_k_semantic=top_k * overfetch, top_k_episodic=top_k * overfetch
        ),
        consolidate_decider=PeriodicConsolidator(min_events=2),
        reranker=JevReranker(client), recall_overfetch=overfetch,
    ) as mem:
        p2t: dict[str, str] = {}
        for e in sc.events:
            r = mem.remember(
                enriched.get(e.id, e.text), title=f"event_{e.id}", tags=f"topic:{e.topic}"
            )
            if r.written and r.ingest is not None:
                p2t[r.ingest.source] = e.topic
        cons = mem.consolidate(method="embedding_v1", embedding_threshold=0.70,
                               embedding_linkage="complete", materialize_fn=JevMaterializer(client))
        uni = {**p2t, **_fact_path_to_topic(cons.facts, p2t)}
        prov = {f.text: f.derived_from for f in cons.facts}
        # retrieval ceiling on this store (episodic, hybrid default, top-200)
        ranks = []
        for q in sc.queries:
            hits = mem._vstash.search(  # noqa: SLF001 — probe, not production
                q.question, top_k=200, collection=mem.collection, layer="episodic"
            )
            ranks.append(_rank_of_answer(list(hits), q, p2t))
        found = [r for r in ranks if r]
        ceiling = {k: sum(1 for r in found if r <= k) / len(ranks) for k in KS}
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
        "n_llm": n_llm,
        "written": len(p2t),
        "facts": cons.facts_written,
        "purity": purity,
        "ceiling": ceiling,
        "missing": len(ranks) - len(found),
        "strict": strict,
        "strict_hit1": strict1,
        "answer": answer,
        "answer_hit1": answer1,
        "queries": len(sc.queries),
        "fails": fails,
        "elapsed_s": time.perf_counter() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    import os

    p = argparse.ArgumentParser(prog="experiments.loop_quality.write_enrich_probe")
    p.add_argument("--scenario", type=Path, action="append", required=True)
    p.add_argument("--llm", default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--overfetch", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    client = CachingJevClient()
    enricher = Enricher(key, a.llm)
    results = {}
    for sp in a.scenario:
        sc = load_scenario(sp)
        variants = (("jev-loop (no enrich)", None), (f"jev-loop + enrich[{a.llm}]", enricher))
        for label, enr in variants:
            r = run(sc, client, enr, top_k=a.top_k, overfetch=a.overfetch)
            results[f"{sc.name}/{label}"] = r
            n = max(1, r["queries"])
            ceil = "  ".join(f"@{k} {r['ceiling'][k]:.0%}" for k in KS)
            print(
                f"{sc.name:28} {label:38} llm-calls {r['n_llm']:3}  purity {r['purity']:.0%}  "
                f"ceiling {ceil} (missing {r['missing']})  "
                f"strict {r['strict']}/{r['queries']} ({r['strict'] / n:.0%}) "
                f"hit@1 {r['strict_hit1']}  answer {r['answer']}/{r['queries']} "
                f"hit@1 {r['answer_hit1']}  {r['elapsed_s']:.0f}s",
                flush=True,
            )
            if r["fails"]:
                print(f"{'':68} fails: {', '.join(r['fails'])}")
    es, js = enricher.stats, client.stats
    print(
        f"\nllm calls {es['calls']} (cached {es['cached']}) cost ${es['cost']:.4f}   "
        f"jev calls {js['calls']} (cached {js['cached']}) cost ${js['cost']:.4f}"
    )
    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
