"""Eval-gated retrain for the d1 write filter: vstash's ``retrain`` loop, applied to a decider.

vstash fine-tunes its embedding model from signal it finds in the user's own corpus, measures the
candidate on a held-out split, and refuses to promote one that regresses. This does the same for
``D1WriteDecider``'s LoRA:

1. **Teach** (optional, ``--teach``): every shadow disagreement in ``~/.merken/*.db`` that has no
   label yet goes to an OpenAI-compatible oracle (DeepSeek by default) through merken's own
   ``label_disagreements``; the label lands in that project's ``merken_labels`` collection.
   Answers are cached on disk by (text, prompt, model), so a rerun never pays twice.
2. **Collect**: ``data/merken_labels_*.jsonl`` plus every ``merken_labels`` row in
   ``~/.merken/*.db``. DECISION / NOISE at confidence >= 0.6, deduplicated on text.
3. **Split by hash**: a label is in the gate's holdout iff sha1(text) falls in the first 15%. The
   split never reshuffles as labels arrive, so successive runs are judged on a growing but
   stable set. Early stopping uses a second hash bucket, never the holdout.
4. **Train** a candidate LoRA (``train.fit``), optionally with synthetic scenarios as v7 did.
5. **Gate** the candidate against the current adapter (or zero-shot d1 if there is none):
   - balanced accuracy on the holdout must improve by at least ``--min-gain``;
   - no real-content scenario may lose accuracy (merken's rule: real notes are never traded);
   - the noise-guard scenarios, pooled, may lose at most ``--guard-tol``, and each one at most
     ``--guard-events`` events (they are small: one event can be 8 points).
6. **Promote** atomically: ``adapters/current`` -> ``adapters/current.old``, candidate ->
   ``adapters/current``, rolled back on failure. Every run appends to ``adapters/history.jsonl``.

``--candidate DIR`` skips training and gates an existing adapter (e.g. to decide between two runs).

    uv run --extra d1 python -m experiments.d1_write_filter.retrain --synthetic \\
        knowledge_update_50topics.json                       # train, gate, promote
    uv run --extra d1 python -m experiments.d1_write_filter.retrain \\
        --candidate adapters/d1-write-synth --current adapters/d1-write --dry-run
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.d1_write_filter import eval as ev

ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = ROOT / "adapters"
CACHE = ROOT / "data" / "teacher_cache.jsonl"

# All-DECISION scenarios of real notes and sessions: a candidate may never lose accuracy here.
REAL = (
    "organic_val_held_out",
    "jay_vstash_snapshot",
    "jay_vstash_decontam",
    "analytics_project",
    "session_2026_04_09",
    "bilingual_es_en",
)
# Scenarios with noise the filter must keep out. ``knowledge_update_20topics`` / ``50topics`` are
# left out: 50topics is a training option and 20topics comes from the same generator.
GUARD = (
    "disjoint_noise_heavy_holdout",
    "knowledge_update_hard",
    "noisy_agent_stream",
    "markdown_tables_held_out",
)


def norm(t: str) -> str:
    return " ".join(t.split()).lower()


def bucket(text: str) -> int:
    """Stable 0-99 bucket from the text, so splits never move as data grows."""
    return int(hashlib.sha1(norm(text).encode()).hexdigest()[:8], 16) % 100


# --------------------------------------------------------------------------- teach


class CachedBackend:
    """Wrap a label backend with an on-disk cache keyed by (text, prompt, model)."""

    def __init__(self, inner, path: Path = CACHE) -> None:
        from merken.labeling import _PROMPT

        self.inner, self.path, self.name = inner, path, inner.name
        self._ph = hashlib.sha1(_PROMPT.encode()).hexdigest()[:12]
        self._cache: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self._cache[row["key"]] = row
        self.hits = 0

    def _key(self, text: str) -> str:
        h = hashlib.sha1(text.encode()).hexdigest()
        return f"{h}:{self._ph}:{self.name}"

    def label(self, text: str):
        from merken.labeling import Label

        k = self._key(text)
        if k in self._cache:
            self.hits += 1
            r = self._cache[k]
            return Label(r["decision"], r["confidence"], r["rationale"], self.name)
        lab = self.inner.label(text)
        row = {"key": k, **asdict(lab)}
        self._cache[k] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return lab


def deepseek_key() -> str:
    """The key from the macOS Keychain (service ``deepseek-api``); never printed or stored."""
    out = subprocess.run(
        ["security", "find-generic-password", "-s", "deepseek-api", "-w"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def teach(model: str, base_url: str, limit: int | None) -> dict[str, int]:
    from merken import Memory
    from merken.labeling import OpenAICompatLabelBackend, label_disagreements

    backend = CachedBackend(
        OpenAICompatLabelBackend(
            model=model,
            base_url=base_url,
            api_key=deepseek_key(),
            extra_body={"thinking": {"type": "disabled"}},
        )
    )
    counts: dict[str, int] = {}
    for db in sorted((Path.home() / ".merken").glob("*.db")):
        mem = Memory(project=db.stem, db=db)
        for status, _title, _label, _detail in label_disagreements(mem, backend, limit=limit):
            counts[status] = counts.get(status, 0) + 1
    counts["cache_hits"] = backend.hits
    return counts


# --------------------------------------------------------------------------- collect


def collect(min_conf: float = 0.6) -> list[dict]:
    from experiments.extract_labels_to_jsonl import iter_labels_from_db, parse_label_body

    raw: list[dict] = []
    for f in sorted((ROOT / "data").glob("merken_labels_*.jsonl")):
        raw += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    for db in sorted((Path.home() / ".merken").glob("*.db")):
        for _title, body, _project in iter_labels_from_db(db):
            obj = parse_label_body(body)
            if obj:
                raw.append(obj)
    rows, seen = [], set()
    for r in raw:
        text = (r.get("text") or r.get("event_text_preview") or "").strip()
        label = r.get("label") or r.get("decision")
        conf = float(r.get("confidence") or 0)
        if not text or label not in ("DECISION", "NOISE") or conf < min_conf:
            continue
        if norm(text) in seen:
            continue
        seen.add(norm(text))
        keep = conf if label == "DECISION" else 1 - conf
        rows.append({"text": text, "keep": keep, "label": label})
    return rows


def split(rows: list[dict], holdout_pct: int = 15, valid_pct: int = 10):
    """(train, valid, holdout): holdout feeds only the gate, valid only early stopping."""
    hold = [r for r in rows if bucket(r["text"]) < holdout_pct]
    valid = [r for r in rows if holdout_pct <= bucket(r["text"]) < holdout_pct + valid_pct]
    train = [r for r in rows if bucket(r["text"]) >= holdout_pct + valid_pct]
    return train, valid, hold


# --------------------------------------------------------------------------- score + gate


def score(adapter: str | None, holdout: list[dict], threshold: float) -> dict:
    from merken.policies.d1_write import D1WriteDecider

    d = D1WriteDecider(adapter=adapter, threshold=threshold)

    def probs(texts: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), 16):
            out += d.probabilities(texts[i : i + 16])
        return out

    ps = probs([r["text"] for r in holdout])
    y = [r["label"] == "DECISION" for r in holdout]
    m = ev.metrics(y, [p >= threshold for p in ps])
    res = {
        "holdout": {
            **m,
            "bacc": ((m["dec_recall"] or 0) + (m["noi_recall"] or 0)) / 2,
        },
        "scenarios": {},
    }
    for name in REAL + GUARD:
        fname = ev.SCENARIOS[name][0]
        pairs = ev.pairs(ev.SCEN / fname)
        sp = probs([t for t, _ in pairs])
        res["scenarios"][name] = ev.metrics([y for _, y in pairs], [p >= threshold for p in sp])
    d._model = None  # release the weights before the next model loads
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    return res


@dataclass
class GateResult:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def _pooled(m: dict) -> float:
    n = sum(m["scenarios"][g].get("n", 1) for g in GUARD)
    return sum(m["scenarios"][g]["agreement"] * m["scenarios"][g].get("n", 1) for g in GUARD) / n


def gate(
    cur: dict,
    cand: dict,
    min_gain: float = 0.0,
    guard_tol: float = 0.02,
    guard_events: int = 1,
) -> GateResult:
    """Promote only a candidate that is better on the holdout and safe on the scenarios.

    - Holdout balanced accuracy must improve by ``min_gain``.
    - REAL scenarios: no loss at all (merken never trades real notes).
    - GUARD scenarios are small (12-108 events), so one event can be 8 points: they are judged
      pooled (event-weighted accuracy may drop at most ``guard_tol``), and each one may lose at
      most ``guard_events`` events.
    """
    reasons, ok = [], True
    dc, dn = cur["holdout"]["bacc"], cand["holdout"]["bacc"]
    if dn < dc + min_gain:
        ok = False
        reasons.append(f"holdout bacc {dn:.3f} < {dc:.3f} + {min_gain}")
    else:
        reasons.append(f"holdout bacc {dc:.3f} -> {dn:.3f}")
    for name in REAL:
        a, b = cur["scenarios"][name]["agreement"], cand["scenarios"][name]["agreement"]
        if b < a:
            ok = False
            reasons.append(f"REAL {name}: {a:.0%} -> {b:.0%}")
    pa, pb = _pooled(cur), _pooled(cand)
    if pb < pa - guard_tol:
        ok = False
        reasons.append(f"GUARD pooled {pa:.1%} -> {pb:.1%}")
    else:
        reasons.append(f"guard pooled {pa:.1%} -> {pb:.1%}")
    for name in GUARD:
        sa, sb = cur["scenarios"][name], cand["scenarios"][name]
        lost = round((sa["agreement"] - sb["agreement"]) * sa.get("n", 1))
        if lost > guard_events:
            ok = False
            reasons.append(f"GUARD {name}: {sa['agreement']:.0%} -> {sb['agreement']:.0%}")
        elif lost > 0:
            reasons.append(f"guard {name} -{lost} event(s), tolerated")
    return GateResult(ok, reasons)


# --------------------------------------------------------------------------- promote


def promote(candidate: Path, final: Path) -> None:
    """Move ``candidate`` into ``final``, keeping the previous one at ``final.old``.

    Same contract as vstash's ``_promote_candidate``: the old adapter is renamed aside first,
    a partial copy is removed before restoring, and the rollback cannot mask the original error.
    """
    backup = final.with_name(final.name + ".old")
    final.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        shutil.rmtree(backup)
    existed, backed_up = final.exists(), False
    try:
        if existed:
            final.rename(backup)
            backed_up = True
        shutil.copytree(candidate, final)
    except Exception:
        try:
            if final.exists():
                shutil.rmtree(final)
            if backed_up:
                backup.rename(final)
        except Exception:  # noqa: BLE001, S110 -- never mask the original error
            pass
        raise


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teach", action="store_true", help="label new shadow disagreements first")
    ap.add_argument("--teacher-model", default="deepseek-flash")
    ap.add_argument("--teacher-url", default="https://api.deepseek.com")
    ap.add_argument("--teach-limit", type=int)
    ap.add_argument("--candidate", help="gate this adapter instead of training one")
    ap.add_argument("--current", help="adapter to beat (default: adapters/current, else zero-shot)")
    ap.add_argument("--synthetic", nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--guard-tol", type=float, default=0.02)
    ap.add_argument("--guard-events", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true", help="gate but never promote")
    a = ap.parse_args()
    t0 = time.time()
    record: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(a)}

    if a.teach:
        record["teach"] = teach(a.teacher_model, a.teacher_url, a.teach_limit)
        print("teach:", record["teach"], flush=True)

    rows = collect()
    train_rows, valid_rows, hold = split(rows)
    record["data"] = {
        "labels": len(rows),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "holdout": len(hold),
    }
    print("datos:", record["data"], flush=True)

    final = ADAPTERS / "current"
    current = a.current or (str(final) if final.exists() else None)
    record["current"] = current or "zero-shot"

    if a.candidate:
        cand_dir = Path(a.candidate)
    else:
        from experiments.d1_write_filter import train as tr

        cand_dir = ADAPTERS / f"candidate-{time.strftime('%Y%m%d-%H%M%S')}"
        synth = tuple(a.synthetic)
        if synth:
            known = {norm(r["text"]) for r in rows}
            extra = [r for r in tr.synthetic_rows(synth, 0.95) if norm(r["text"]) not in known]
            extra = [r for r in extra if bucket(r["text"]) >= 25]  # never in holdout/valid buckets
            train_rows = train_rows + extra
            record["data"]["synthetic"] = len(extra)
        record["best_valid_loss"] = tr.fit(train_rows, valid_rows, cand_dir, epochs=a.epochs)
    record["candidate"] = str(cand_dir)

    print(f"midiendo actual ({record['current']})…", flush=True)
    cur = score(current, hold, a.threshold)
    print("midiendo candidato…", flush=True)
    cand = score(str(cand_dir), hold, a.threshold)
    g = gate(cur, cand, a.min_gain, a.guard_tol, a.guard_events)
    record.update(
        {
            "current_metrics": cur,
            "candidate_metrics": cand,
            "promote": g.promote,
            "reasons": g.reasons,
        }
    )
    if g.promote and not a.dry_run:
        promote(cand_dir, final)
        record["promoted_to"] = str(final)
    record["minutes"] = round((time.time() - t0) / 60, 1)
    ADAPTERS.mkdir(exist_ok=True)
    with (ADAPTERS / "history.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    def f(x):
        return "  -  " if x is None else f"{x:5.0%}"

    print(f"\n{'':32s} {'actual':>8s} {'candidato':>10s}")
    print(f"{'holdout bacc':32s} {f(cur['holdout']['bacc']):>8s} {f(cand['holdout']['bacc']):>10s}")
    for name in REAL + GUARD:
        tag = "REAL " if name in REAL else "GUARD"
        print(
            f"{tag} {name[:26]:26s} {f(cur['scenarios'][name]['agreement']):>8s} "
            f"{f(cand['scenarios'][name]['agreement']):>10s}"
        )
    verdict = (
        "PROMOVIDO"
        if record.get("promoted_to")
        else ("pasa el gate (dry-run, no promovido)" if g.promote else "RECHAZADO")
    )
    print(f"\n{verdict}: " + "; ".join(g.reasons))


if __name__ == "__main__":
    main()
