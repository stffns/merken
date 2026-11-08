"""LoRA on d1-2B from merken's oracled write labels: the zero-shot D1WriteDecider,
taught merken's own definition.

Same recipe and code as jev-local's ``train_d1.py`` (d1's slot readout with gradients,
cross-entropy against a soft target), imported from ``~/code/jev-local``. The question is
``merken.policies.d1_write.INSTRUCTIONS`` verbatim, so the adapter drops into
``D1WriteDecider`` unchanged. Target per label: P(keep) = confidence for DECISION and
1 - confidence for NOISE; UNCERTAIN rows are dropped.

Training data is ``data/merken_labels_*.jsonl`` (private, gitignored). Any label whose text
also appears in a ``loop_quality`` scenario is dropped first, so ``eval.py --adapter`` stays
out-of-sample. A stratified 15% is held out for validation.

Run with the destila environment (torch, transformers, peft):
    ~/code/destila/.venv/bin/python -m experiments.d1_write_filter.train --out adapters/d1-write
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import random
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path.home() / "code" / "jev-local"))

import ast  # noqa: E402

import torch  # noqa: E402
from train_d1 import batches, evaluate, load_model, loss_of, save  # noqa: E402


def _instructions() -> str:
    """The decider's question, read from its source.

    Importing merken would pull vstash into this (training-only) environment.
    """
    tree = ast.parse((ROOT / "merken" / "policies" / "d1_write.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "INSTRUCTIONS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("INSTRUCTIONS not found in d1_write.py")


INSTRUCTIONS = _instructions()
SCEN = ROOT / "experiments" / "loop_quality" / "scenarios"


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def scenario_texts(skip: tuple[str, ...] = ()) -> set[str]:
    seen = set()
    for f in SCEN.glob("*.json"):
        if f.name in skip:
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for e in data.get("events", []) if isinstance(data, dict) else []:
            if e.get("text"):
                seen.add(norm(e["text"])[:200])
    return seen


def load_rows(min_conf: float, synthetic: tuple[str, ...] = ()) -> list[dict]:
    blocked, rows, dropped = scenario_texts(synthetic), [], 0
    for f in sorted(glob.glob(str(ROOT / "data" / "merken_labels_*.jsonl"))):
        for line in pathlib.Path(f).read_text().splitlines():
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            if (
                not text
                or r.get("label") not in ("DECISION", "NOISE")
                or (r.get("confidence") or 0) < min_conf
            ):
                continue
            if norm(text)[:200] in blocked:
                dropped += 1
                continue
            c = float(r["confidence"])
            keep = c if r["label"] == "DECISION" else 1 - c
            rows.append(
                {"text": text, "keep": keep, "label": r["label"], "file": pathlib.Path(f).name}
            )
    # the same event can be labelled in more than one export: keep the first
    uniq = {}
    for r in rows:
        uniq.setdefault(norm(r["text"]), r)
    print(f"{len(uniq):,} etiquetas únicas | {dropped} descartadas (aparecen en un escenario)")
    return list(uniq.values())


def synthetic_rows(files: tuple[str, ...], soft: float) -> list[dict]:
    """Scenario events used as training data, the way v7 trained on knowledge_update_50topics.

    Labelled like ``eval.py`` (DECISION unless ``topic == "noise"``). An event whose text also
    appears in any other scenario is dropped, so the held-out scenarios stay out-of-sample.
    """
    blocked, out, dropped = scenario_texts(files), [], 0
    for name in files:
        for e in json.loads((SCEN / name).read_text()).get("events", []):
            text = (e.get("text") or "").strip()
            if not text:
                continue
            if norm(text)[:200] in blocked:
                dropped += 1
                continue
            dec = e.get("topic", "") != "noise"
            out.append(
                {
                    "text": text,
                    "keep": soft if dec else 1 - soft,
                    "label": "DECISION" if dec else "NOISE",
                    "file": name,
                }
            )
    uniq = {}
    for r in out:
        uniq.setdefault(norm(r["text"]), r)
    print(
        f"sintéticos: {len(uniq):,} únicos de {', '.join(files)} | "
        f"{dropped} descartados (aparecen en otro escenario)"
    )
    return list(uniq.values())


def n_dec(rows: list[dict]) -> int:
    return sum(r["label"] == "DECISION" for r in rows)


def to_examples(engine, rows: list[dict], max_chars: int) -> list[dict]:
    q = engine.named({"keep": {"type": "noul", "instructions": INSTRUCTIONS}})["keep"]
    out = []
    for r in rows:
        state = r["text"][:max_chars]
        out.append(
            {
                "state": state,
                "q": q,
                "target": [r["keep"], 1 - r["keep"]],
                "row": r,
                "n": engine.tokens(state, [q]),
            }
        )
    return out


def fit(
    train_rows: list[dict],
    valid_rows: list[dict],
    out: pathlib.Path,
    *,
    model: str = "LiquidAI/d1-2B",
    batch: int = 8,
    accum: int = 1,
    lr: float = 1e-4,
    warmup: int = 30,
    epochs: float = 3.0,
    eval_every: int = 50,
    max_chars: int = 12_000,
    seed: int = 0,
) -> float:
    """Train a LoRA on ``train_rows``; save the best-on-``valid_rows`` adapter to ``out``.

    Returns the best validation loss. The model is local to this call, so it is released
    when the call returns and callers can evaluate right after on the same GPU.
    """
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    m, engine, cfg, dev = load_model(model)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    train, valid = (
        to_examples(engine, train_rows, max_chars),
        to_examples(engine, valid_rows, max_chars),
    )
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = int(math.ceil(len(train) / batch / accum) * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)),
    )

    def report(tag: str) -> float:
        m.eval()
        v = evaluate(engine, valid, batch)
        print(
            f"{tag}: validación pérdida {v['loss']:.4f} | |Δp| {v['dp']:.3f} | "
            f"acuerdo {v['agree']:.1%}",
            flush=True,
        )
        return v["loss"]

    best, step, t0, run_loss = report("paso 0"), 0, time.time(), []
    while step < total:
        for group in batches(train, batch, rnd):
            m.train()
            loss = loss_of(engine, group) / accum
            loss.backward()
            run_loss.append(float(loss.detach()) * accum)
            if len(run_loss) % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if dev == "mps":
                torch.mps.empty_cache()
            if step % 25 == 0:
                el = time.time() - t0
                print(
                    f"paso {step:>5}/{total} | "
                    f"pérdida {sum(run_loss[-50:]) / len(run_loss[-50:]):.4f} | "
                    f"quedan {(total - step) * el / step / 60:.0f} min",
                    flush=True,
                )
            if step % eval_every == 0 or step == total:
                v = report(f"paso {step}")
                if v < best:
                    best = v
                    save(m, cfg, out)
                    print("  ← mejor, guardado", flush=True)
            if step >= total:
                break
    if dev == "mps":
        torch.mps.empty_cache()
    print(
        f"listo: {step} pasos en {(time.time() - t0) / 60:.0f} min, "
        f"mejor pérdida de validación {best:.4f}"
    )

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/d1-2B")
    ap.add_argument("--out", default="adapters/d1-write")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--max-chars", type=int, default=12_000)
    ap.add_argument("--valid-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--synthetic",
        nargs="*",
        default=[],
        help="loop_quality scenario files added to training only",
    )
    ap.add_argument("--synthetic-soft", type=float, default=0.95)
    a = ap.parse_args()
    synth = tuple(a.synthetic)
    torch.manual_seed(a.seed)
    rnd = random.Random(a.seed)

    rows = load_rows(a.min_conf, synth)
    rnd.shuffle(rows)
    train_rows, valid_rows = [], []
    for lab in ("DECISION", "NOISE"):  # stratified split
        part = [r for r in rows if r["label"] == lab]
        k = int(len(part) * a.valid_share)
        valid_rows += part[:k]
        train_rows += part[k:]
    if synth:  # training only: validation stays the real held-out labels
        known = {norm(r["text"]) for r in rows}
        train_rows += [
            r for r in synthetic_rows(synth, a.synthetic_soft) if norm(r["text"]) not in known
        ]
    print(
        f"entrenamiento {len(train_rows):,} ({n_dec(train_rows)} DECISION) | "
        f"validación {len(valid_rows):,} ({n_dec(valid_rows)} DECISION)"
    )

    fit(
        train_rows,
        valid_rows,
        ROOT / a.out,
        model=a.model,
        batch=a.batch,
        accum=a.accum,
        lr=a.lr,
        warmup=a.warmup,
        epochs=a.epochs,
        eval_every=a.eval_every,
        max_chars=a.max_chars,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
