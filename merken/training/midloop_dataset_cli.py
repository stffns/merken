"""CLI for `merken-midloop-dataset` -- Phase 1 dataset builder.

Subcommands:

  generate-cases     -- large LLM -> (prompt, truth) pairs from
                        protocols. Default backend: anthropic Claude
                        Sonnet 4.6. Reads protocols JSONL, writes
                        cases JSONL.
  generate-responses -- small LM -> model_response per case prompt.
                        Default backend: HF transformers Gemma 3 1B.
                        Reads cases JSONL, writes cases-with-response
                        JSONL.
  align              -- alignment + semantic filter; emits intervention
                        labels. No external model.
  build              -- end-to-end orchestrator: protocols -> cases ->
                        responses -> aligned. Reuses the three steps
                        above sequentially.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from merken.training.case_generator import (
    CaseGenerator,
    GeneratedCase,
    LLMClient,
    ProtocolClause,
    default_anthropic_client,
)
from merken.training.midloop_dataset import Aligner, Case, default_embed_fn
from merken.training.response_generator import (
    GenerateFn,
    ResponseGenerator,
    default_hf_client,
)


def _stable_case_id(prompt: str, truth: str, model_response: str) -> str:
    """Deterministic id when the input JSONL omits case_id."""
    h = hashlib.sha1()
    for s in (prompt, truth, model_response):
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return f"case_{h.hexdigest()[:12]}"


def _load_cases(path: Path):
    """Yield Case from a JSONL where each line has at least
    ``truth`` and ``model_response``. ``prompt``, ``case_id``, and
    ``metadata`` are optional.
    """
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"warn: skipping line {lineno}: {e}", file=sys.stderr)
                continue
            truth = row.get("truth")
            model_response = row.get("model_response")
            if truth is None or model_response is None:
                print(
                    f"warn: skipping line {lineno}: missing truth or model_response",
                    file=sys.stderr,
                )
                continue
            yield Case(
                truth=truth,
                model_response=model_response,
                prompt=row.get("prompt", ""),
                case_id=row.get("case_id") or _stable_case_id(
                    row.get("prompt", ""), truth, model_response,
                ),
                metadata=row.get("metadata", {}),
            )


def _serialize_result(result, *, drop_subseq_text: bool) -> dict:
    """Build the JSON output row for one aligned case.

    ``drop_subseq_text=True`` omits truth_text / model_text from the
    intervention list -- useful when the dataset is large and the
    full text is recoverable from (case_id, token_idx). Default
    keeps everything for inspection.
    """
    interventions = []
    for r in result.interventions:
        item = {
            "model_token_start": r.model_start,
            "model_token_end": r.model_end,
            "truth_token_start": r.truth_start,
            "truth_token_end": r.truth_end,
            "cosine_sim": r.cosine_sim,  # None for pure insert/delete
        }
        if not drop_subseq_text:
            item["truth_text"] = r.truth_text
            item["model_text"] = r.model_text
        interventions.append(item)
    return {
        "case_id": result.case.case_id,
        "prompt": result.case.prompt,
        "truth": result.case.truth,
        "model_response": result.case.model_response,
        "metadata": result.case.metadata,
        "n_divergent": len(result.divergent_regions),
        "n_dropped_semantic": len(result.semantic_drops),
        "n_interventions": len(result.interventions),
        "interventions": interventions,
    }


def cmd_align(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"error: input file not found: {in_path}", file=sys.stderr)
        return 2

    if args.no_embed:
        embed_fn = None
        print(
            "--no-embed: aligner will skip semantic filter; "
            "all divergent regions become interventions",
            file=sys.stderr,
        )
    else:
        embed_fn = default_embed_fn(model_name=args.embed_model)

    aligner = Aligner(embed_fn=embed_fn, similarity_threshold=args.threshold)
    n_cases = 0
    n_div_total = 0
    n_drop_total = 0
    n_int_total = 0

    with out_path.open("w") as fout:
        for case in _load_cases(in_path):
            try:
                if args.no_embed:
                    result = aligner.align_only(case)
                else:
                    result = aligner.process(case)
            except Exception as e:
                print(
                    f"warn: case {case.case_id}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue
            row = _serialize_result(result, drop_subseq_text=args.compact)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_cases += 1
            n_div_total += row["n_divergent"]
            n_drop_total += row["n_dropped_semantic"]
            n_int_total += row["n_interventions"]
            if n_cases % 100 == 0:
                print(
                    f"  processed {n_cases} cases "
                    f"({n_int_total} interventions, "
                    f"{n_drop_total} semantic drops)",
                    file=sys.stderr,
                )

    print(
        f"\n=== Phase 1 dataset built ===\n"
        f"  cases:                  {n_cases}\n"
        f"  divergent regions:      {n_div_total}\n"
        f"  semantic drops:         {n_drop_total} "
        f"({100 * n_drop_total / max(n_div_total, 1):.1f}%)\n"
        f"  intervention labels:    {n_int_total}\n"
        f"  output:                 {out_path}",
        file=sys.stderr,
    )
    return 0


def _load_protocols(path: Path):
    """Yield ProtocolClause from a JSONL where each line has at
    least ``protocol_id`` and ``text``. ``metadata`` is optional.
    """
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"warn: skipping line {lineno}: {e}", file=sys.stderr)
                continue
            pid = row.get("protocol_id")
            text = row.get("text")
            if not pid or not text:
                print(
                    f"warn: skipping line {lineno}: missing protocol_id or text",
                    file=sys.stderr,
                )
                continue
            yield ProtocolClause(
                protocol_id=str(pid),
                text=str(text),
                metadata=row.get("metadata", {}),
            )


def _load_generated_cases(path: Path):
    """Yield GeneratedCase from JSONL emitted by `generate-cases`.

    Skips malformed rows with a stderr warning (consistent with
    ``_load_cases`` and ``_load_protocols``); a single bad line must
    not abort the whole ``generate-responses`` batch.
    """
    with path.open() as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"warn: skipping line {lineno}: {e}", file=sys.stderr)
                continue
            missing = [k for k in ("case_id", "prompt", "truth") if not row.get(k)]
            if missing:
                print(
                    f"warn: skipping line {lineno}: missing required keys {missing}",
                    file=sys.stderr,
                )
                continue
            yield GeneratedCase(
                case_id=row["case_id"],
                prompt=row["prompt"],
                truth=row["truth"],
                metadata=row.get("metadata", {}),
            )


def _resolve_llm_client(args) -> LLMClient:
    """Build the case-generation LLM client per CLI args.

    Lookup order:
      1. ANTHROPIC_API_KEY env (with --large-model override).
      2. Error if missing -- there is no offline fallback for
         generation (the case generator requires a real LLM).
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "error: ANTHROPIC_API_KEY env var not set; required for "
            "generate-cases. Tests use a fake client; the CLI does not."
        )
    return default_anthropic_client(api_key=api_key, model=args.large_model)


def _resolve_generate_fn(args) -> GenerateFn:
    """Build the response-generation small-model GenerateFn per args."""
    return default_hf_client(
        model_name=args.small_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


def cmd_generate_cases(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"error: protocols file not found: {in_path}", file=sys.stderr)
        return 2

    llm = _resolve_llm_client(args)
    generator = CaseGenerator(llm)

    n_protocols = 0
    n_cases = 0
    with out_path.open("w") as fout:
        for clause in _load_protocols(in_path):
            n_protocols += 1
            try:
                cases = generator.generate(clause, n=args.n)
            except Exception as e:
                print(
                    f"warn: protocol {clause.protocol_id}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue
            for case in cases:
                fout.write(json.dumps({
                    "case_id": case.case_id,
                    "prompt": case.prompt,
                    "truth": case.truth,
                    "metadata": case.metadata,
                }, ensure_ascii=False) + "\n")
                n_cases += 1
            if n_protocols % 10 == 0:
                print(
                    f"  processed {n_protocols} protocols -> {n_cases} cases",
                    file=sys.stderr,
                )

    print(
        f"\n=== generate-cases complete ===\n"
        f"  protocols: {n_protocols}\n"
        f"  cases:     {n_cases}\n"
        f"  output:    {out_path}",
        file=sys.stderr,
    )
    return 0


def cmd_generate_responses(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"error: cases file not found: {in_path}", file=sys.stderr)
        return 2

    print(f"loading small model {args.small_model!r}...", file=sys.stderr)
    gen_fn = _resolve_generate_fn(args)
    generator = ResponseGenerator(gen_fn)

    n = 0
    with out_path.open("w") as fout:
        for case in _load_generated_cases(in_path):
            try:
                cwr = generator.respond(case)
            except Exception as e:
                print(
                    f"warn: case {case.case_id}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue
            fout.write(json.dumps({
                "case_id": cwr.case_id,
                "prompt": cwr.prompt,
                "truth": cwr.truth,
                "model_response": cwr.model_response,
                "metadata": cwr.metadata,
            }, ensure_ascii=False) + "\n")
            n += 1
            if n % 10 == 0:
                print(f"  responded {n} cases", file=sys.stderr)

    print(
        f"\n=== generate-responses complete ===\n"
        f"  cases responded: {n}\n"
        f"  output:          {out_path}",
        file=sys.stderr,
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """End-to-end: protocols -> cases -> responses -> aligned.

    Writes intermediate artifacts to ``<out_dir>/cases.jsonl``,
    ``<out_dir>/responses.jsonl``, and ``<out_dir>/aligned.jsonl``.

    Each step OVERWRITES its target artifact, so a re-run produces
    a fresh dataset end-to-end (no skip-if-exists logic). To skip
    early steps -- e.g. if cases.jsonl is already correct and you
    only want to re-run responses + align -- invoke the subcommands
    individually instead of ``build``. A future revision may add a
    ``--skip-existing`` flag if the manual orchestration becomes
    common; today the explicit subcommand path is the supported
    way to partial-rebuild.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "cases.jsonl"
    responses_path = out_dir / "responses.jsonl"
    aligned_path = out_dir / "aligned.jsonl"

    print("=== Phase 1 build ===", file=sys.stderr)
    print(f"step 1/3: generate-cases -> {cases_path}", file=sys.stderr)
    args.input = args.protocols
    args.output = str(cases_path)
    rc = cmd_generate_cases(args)
    if rc != 0:
        return rc

    print(f"\nstep 2/3: generate-responses -> {responses_path}", file=sys.stderr)
    args.input = str(cases_path)
    args.output = str(responses_path)
    rc = cmd_generate_responses(args)
    if rc != 0:
        return rc

    print(f"\nstep 3/3: align -> {aligned_path}", file=sys.stderr)
    args.input = str(responses_path)
    args.output = str(aligned_path)
    rc = cmd_align(args)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="merken-midloop-dataset",
        description=(
            "Build the midloop training dataset from "
            "(truth, model_response) case pairs. See "
            "notes/midloop-spec.md for the design."
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    aln = sub.add_parser(
        "align",
        help="align truth vs model + filter semantically equivalent divergences",
    )
    aln.add_argument(
        "--input", "--in", required=True,
        help="JSONL input. Each line needs `truth` and `model_response` "
             "(plus optional `prompt`, `case_id`, `metadata`).",
    )
    aln.add_argument(
        "--output", "--out", required=True,
        help="JSONL output, one alignment result per case.",
    )
    aln.add_argument(
        "--threshold", type=float, default=0.90,
        help="cosine-sim threshold; >= keeps as semantic drop, < keeps as "
             "intervention. Default 0.90 (Jay's spec; calibrate empirically).",
    )
    aln.add_argument(
        "--embed-model", default="BAAI/bge-small-en-v1.5",
        help="embedder model name (default: bge-small)",
    )
    aln.add_argument(
        "--no-embed", action="store_true",
        help="skip semantic filter (all divergences become interventions). "
             "Useful for fast iteration / debugging the diff alone.",
    )
    aln.add_argument(
        "--compact", action="store_true",
        help="omit truth_text / model_text from interventions to "
             "save disk; recoverable via (case_id, token_idx).",
    )
    aln.set_defaults(func=cmd_align)

    # ----- generate-cases
    gc = sub.add_parser(
        "generate-cases",
        help="large-LLM expansion of protocol clauses into clinical cases",
    )
    gc.add_argument(
        "--input", "--in", required=True,
        help="JSONL of protocols. Each line: {protocol_id, text, metadata?}.",
    )
    gc.add_argument(
        "--output", "--out", required=True,
        help="JSONL output of generated cases.",
    )
    gc.add_argument(
        "--n", type=int, default=5,
        help="cases per protocol clause (default 5).",
    )
    gc.add_argument(
        "--large-model", default="claude-sonnet-4-6",
        help="anthropic model id (default claude-sonnet-4-6).",
    )
    gc.set_defaults(func=cmd_generate_cases)

    # ----- generate-responses
    gr = sub.add_parser(
        "generate-responses",
        help="small-model responses to each case prompt",
    )
    gr.add_argument(
        "--input", "--in", required=True,
        help="JSONL of cases (output of generate-cases).",
    )
    gr.add_argument(
        "--output", "--out", required=True,
        help="JSONL output of cases-with-response.",
    )
    gr.add_argument(
        "--small-model", default="google/gemma-3-1b-it",
        help="HF transformers model id. Override per spec; the default "
             "is conservative -- Gemma 3 1B fits in MPS without GPU.",
    )
    gr.add_argument(
        "--device", default="auto",
        help="HF device map (auto / cpu / cuda / mps).",
    )
    gr.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="generation length cap.",
    )
    gr.set_defaults(func=cmd_generate_responses)

    # ----- build (end-to-end orchestrator)
    bd = sub.add_parser(
        "build",
        help="end-to-end pipeline: protocols -> cases -> responses -> aligned",
    )
    bd.add_argument(
        "--protocols", required=True,
        help="JSONL of source protocols.",
    )
    bd.add_argument(
        "--out-dir", required=True,
        help="output directory; writes cases.jsonl, responses.jsonl, aligned.jsonl.",
    )
    bd.add_argument("--n", type=int, default=5)
    bd.add_argument("--large-model", default="claude-sonnet-4-6")
    bd.add_argument("--small-model", default="google/gemma-3-1b-it")
    bd.add_argument("--device", default="auto")
    bd.add_argument("--max-new-tokens", type=int, default=512)
    bd.add_argument("--threshold", type=float, default=0.90)
    bd.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    bd.add_argument(
        "--no-embed", action="store_true",
        help="skip semantic filter at the alignment step.",
    )
    bd.add_argument("--compact", action="store_true")
    bd.set_defaults(func=cmd_build)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
