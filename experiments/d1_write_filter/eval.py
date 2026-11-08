"""D1WriteDecider (zero-shot, Liquid AI d1-2B) vs nanoGPT v7 on the loop_quality write filter.

Same labels and metrics as ``experiments/eval_v7_vs_v6.py``: an event is DECISION (should be
written) unless its ``topic`` is ``"noise"``, except ``noisy_agent_stream``, whose events carry a
``content_type`` (decision / tool_echo / ack / status / ambient_chat) and whose 16 noise events the
v7 eval scored as DECISION; it is labelled by content_type here and has no comparable v7
number. v7's numbers are read from the committed ``experiments/nanogpt/eval_v6_to_v9.json`` and
``disjoint_holdout_eval.json`` rather than re-run, since the v7 checkpoint is not in the repo.

d1 has seen none of these scenarios and none of merken's 1026 oracled labels: every number here is
zero-shot. ``knowledge_update_50topics`` is v7's *training* set, so it is out-of-sample only for d1.

    python -m experiments.d1_write_filter.eval            # writes results.json next to this file
"""

from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCEN = ROOT / "experiments" / "loop_quality" / "scenarios"
NANO = ROOT / "experiments" / "nanogpt"
OUT = Path(__file__).with_name("results.json")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

SCENARIOS = {
    # name in this report: (file, key in eval_v6_to_v9.json or None)
    "markdown_tables_held_out": ("markdown_tables_held_out.json", "markdown_tables_held_out"),
    "organic_val_held_out": ("organic_val_held_out.json", "organic_val_held_out"),
    "jay_vstash_snapshot": ("jay_vstash_2026_04_09_snapshot.json", "jay_vstash_snapshot"),
    "jay_vstash_decontam": (
        "jay_vstash_2026_04_09_snapshot_decontam.json",
        "jay_vstash_snapshot_decontam",
    ),
    "analytics_project": ("analytics_project.json", "analytics_project"),
    "session_2026_04_09": ("session_2026_04_09.json", "session_2026_04_09"),
    "bilingual_es_en": ("bilingual_es_en_2026_04_14.json", "bilingual_es_en_2026_04_14"),
    # v7's eval scored this scenario's noise_* events as DEC, so it has no comparable v7 number
    "noisy_agent_stream": ("noisy_agent_stream.json", None),
    "disjoint_noise_heavy_holdout": ("disjoint_noise_heavy_holdout.json", None),
    "knowledge_update_hard": ("knowledge_update_hard.json", "knowledge_update_hard"),
    "knowledge_update_20topics (d1 DEV: prompt tuned here)": (
        "knowledge_update_20topics.json",
        "knowledge_update_20t",
    ),
    "knowledge_update_50topics (v7 TRAINING)": (
        "knowledge_update_50topics.json",
        "knowledge_update_50t (TRAINING)",
    ),
}


def pairs(path: Path) -> list[tuple[str, bool]]:
    data = json.loads(path.read_text())
    out = []
    for e in data.get("events", []):
        text = (e.get("text") or "").strip()
        if text:
            if "content_type" in e:  # noisy_agent_stream: echoes, acks, status, chatter = noise
                out.append((text, e["content_type"] == "decision"))
            else:
                out.append((text, e.get("topic", "") != "noise"))
    return out


def metrics(labels: list[bool], writes: list[bool]) -> dict:
    tp = sum(y and w for y, w in zip(labels, writes, strict=True))
    tn = sum(not y and not w for y, w in zip(labels, writes, strict=True))
    fp = sum(not y and w for y, w in zip(labels, writes, strict=True))
    fn = sum(y and not w for y, w in zip(labels, writes, strict=True))
    n = len(labels)
    return {
        "n": n,
        "n_dec": tp + fn,
        "n_noi": tn + fp,
        "agreement": (tp + tn) / n if n else None,
        "dec_recall": tp / (tp + fn) if tp + fn else None,
        "noi_recall": tn / (tn + fp) if tn + fp else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
    }


def v7_reference() -> dict[str, dict]:
    ref = {}
    table = json.loads((NANO / "eval_v6_to_v9.json").read_text())["scenarios"]
    for name, (_, key) in SCENARIOS.items():
        if key and key in table:
            ref[name] = table[key]["per_version"]["v7"]
    disjoint = json.loads((NANO / "disjoint_holdout_eval.json").read_text())["versions"]["v7"]
    ref["disjoint_noise_heavy_holdout"] = {
        "agreement": disjoint["acc"],
        "dec_recall": disjoint["dec_recall"],
        "noi_recall": disjoint["noi_recall"],
        "fpr": disjoint["fpr"],
    }
    return ref


def main() -> None:
    import argparse

    from merken.policies.d1_write import D1WriteDecider

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA directory from train.py")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    decider = D1WriteDecider(adapter=a.adapter, threshold=a.threshold)
    out = Path(a.out)
    ref = v7_reference()
    results: dict = {
        "model": decider.model_name,
        "adapter": a.adapter,
        "zero_shot": a.adapter is None,
        "threshold": decider.threshold,
        "instructions": decider.instructions,
        "scenarios": {},
    }
    latencies: list[float] = []
    for name, (fname, _) in SCENARIOS.items():
        ps_labels = pairs(SCEN / fname)
        texts = [t for t, _ in ps_labels]
        labels = [y for _, y in ps_labels]
        probs = []
        for i in range(0, len(texts), 16):
            t = time.perf_counter()
            probs += decider.probabilities(texts[i : i + 16])
            latencies.append((time.perf_counter() - t) * 1000 / len(texts[i : i + 16]))
        by_th = {str(th): metrics(labels, [p >= th for p in probs]) for th in THRESHOLDS}
        results["scenarios"][name] = {
            "d1": by_th,
            "v7": ref.get(name),
            "p_dec_mean": st.mean([p for p, y in zip(probs, labels, strict=True) if y])
            if any(labels)
            else None,
            "p_noise_mean": st.mean([p for p, y in zip(probs, labels, strict=True) if not y])
            if not all(labels)
            else None,
            "misses": [
                {"p": round(p, 3), "label": "DEC" if y else "NOISE", "text": t[:160]}
                for t, y, p in zip(texts, labels, probs, strict=True)
                if (p >= decider.threshold) != y
            ][:6],
        }
        m = by_th[str(decider.threshold)]
        v7 = ref.get(name) or {}

        def f(x):
            return "  -  " if x is None else f"{x:5.0%}"

        print(
            f"{name:40s} n={m['n']:4d} dec={m['n_dec']:4d} | "
            f"d1@{decider.threshold} acc {f(m['agreement'])} "
            f"DECrec {f(m['dec_recall'])} FPR {f(m['fpr'])} | v7 acc {f(v7.get('agreement'))} "
            f"DECrec {f(v7.get('dec_recall'))} FPR {f(v7.get('fpr'))}",
            flush=True,
        )
    results["ms_per_event_median"] = st.median(latencies)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nlatencia: {st.median(latencies):.0f} ms por evento (lotes de 16) -> {out}")


if __name__ == "__main__":
    main()
