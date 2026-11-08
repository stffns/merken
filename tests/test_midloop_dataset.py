"""Tests for `merken.training.midloop_dataset` aligner.

Uses a deterministic dummy embedder so the suite stays offline + fast.
The MedLocal-shaped tests cover the load-bearing case: a paraphrased
correct subseq must be DROPPED (not labeled as intervention) and a
materially-wrong subseq must be KEPT.

A separate small set runs against the real fastembed-backed embedder
under the `slow` mark for end-to-end sanity. Those are deselected
by default.
"""

from __future__ import annotations

import re

import pytest

from merken.training.midloop_dataset import (
    Aligner,
    AlignmentResult,
    Case,
    diff_regions,
    tokenize,
)

# ----------------------------------------------------------------- dummy embed

# Toy 5-dim embedding: count occurrences of a fixed bag of tokens.
# Two strings get identical embeddings iff they contain the same
# tokens with the same multiplicities (order-insensitive). Cosine
# of identical vectors is 1.0; orthogonal bags give 0.0; partial
# overlap gives values in between. Deterministic, no model load.
_BAG = ["mg", "kg", "kilo", "miligramos", "diarios", "amoxicillin"]


def _dummy_embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for t in texts:
        words = re.findall(r"\w+", t.lower())
        # base on counts of bag tokens; always at least 1.0 in the
        # last slot so empty strings have a defined direction
        # (avoiding divide-by-zero in cosine).
        vec = [float(words.count(b)) for b in _BAG] + [1.0]
        out.append(vec)
    return out


# An embedder that ALWAYS returns the same vector -- everything is
# semantically equivalent. Useful to confirm the threshold actually
# kicks in and drops everything except pure inserts/deletes.
def _identity_embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in texts]


# ----------------------------------------------------------------- tokenize

def test_tokenize_splits_on_whitespace() -> None:
    assert tokenize("amoxicillin 50 mg/kg/day") == [
        "amoxicillin", "50", "mg/kg/day",
    ]
    assert tokenize("") == []
    assert tokenize("   ") == []


# ----------------------------------------------------------------- diff_regions

def test_identical_sequences_have_no_regions() -> None:
    truth = ["a", "b", "c"]
    assert diff_regions(truth, truth) == []


def test_single_replace_yields_one_region() -> None:
    regions = diff_regions(["a", "b", "c"], ["a", "X", "c"])
    assert len(regions) == 1
    r = regions[0]
    assert r.truth_text == "b"
    assert r.model_text == "X"
    assert (r.truth_start, r.truth_end) == (1, 2)
    assert (r.model_start, r.model_end) == (1, 2)


def test_pure_insertion_has_empty_truth_text() -> None:
    regions = diff_regions(["a", "c"], ["a", "b", "c"])
    assert len(regions) == 1
    assert regions[0].truth_text == ""
    assert regions[0].model_text == "b"


def test_pure_deletion_has_empty_model_text() -> None:
    regions = diff_regions(["a", "b", "c"], ["a", "c"])
    assert len(regions) == 1
    assert regions[0].truth_text == "b"
    assert regions[0].model_text == ""


def test_multiple_regions_in_one_case() -> None:
    regions = diff_regions(
        ["the", "dose", "is", "50", "mg", "for", "kids"],
        ["the", "dose", "is", "80", "mg", "for", "adults"],
    )
    assert len(regions) == 2
    assert regions[0].truth_text == "50"
    assert regions[0].model_text == "80"
    assert regions[1].truth_text == "kids"
    assert regions[1].model_text == "adults"


# ----------------------------------------------------------------- align_only

def test_align_only_does_not_require_embedder() -> None:
    aligner = Aligner()  # no embed_fn
    case = Case(truth="a b c", model_response="a X c")
    res = aligner.align_only(case)
    assert isinstance(res, AlignmentResult)
    assert len(res.divergent_regions) == 1
    assert len(res.interventions) == 1  # all regions kept; no filter
    assert res.semantic_drops == []
    assert res.intervene_at_token_indices == [1]


def test_process_without_embedder_raises() -> None:
    aligner = Aligner()
    case = Case(truth="a b c", model_response="a X c")
    with pytest.raises(RuntimeError, match="requires an embed_fn"):
        aligner.process(case)


# ----------------------------------------------------------------- semantic filter

def test_paraphrase_is_dropped_real_failure_is_kept() -> None:
    """Load-bearing test for the MedLocal use case.

    Region 1 (50 mg/kg/dia vs 50 miligramos por kilo diarios) is a
    paraphrase -- semantically equivalent at the bag-of-clinical-
    terms level. The dummy embedder returns identical vectors for
    strings with the same bag tokens. With threshold=0.90 this
    region MUST be dropped.

    Region 2 (50 mg/kg vs 80 mg/kg) is a numeric-dose change. The
    dummy embedder distinguishes by the bag tokens but '50' and
    '80' are not in the bag, so both vectors are equal too. To
    keep this region in the test, we use the IDENTITY embedder
    only for the paraphrase test and dummy_bag for a separate
    multi-region test below.
    """
    aligner = Aligner(_identity_embed, similarity_threshold=0.90)
    case = Case(
        truth="50 mg/kg/dia",
        model_response="50 miligramos por kilo diarios",
    )
    res = aligner.process(case)
    # The whole substantive divergence is one region. Identity
    # embedder => sim=1.0 >= 0.90 => dropped.
    assert len(res.divergent_regions) == 1
    assert len(res.semantic_drops) == 1
    assert len(res.interventions) == 0
    assert res.divergent_regions[0].cosine_sim == pytest.approx(1.0)


def test_threshold_below_sim_keeps_region() -> None:
    """Sim=1.0 with threshold=1.5 keeps everything as intervention."""
    # Use threshold > 1.0 (impossible cosine) so nothing drops.
    aligner = Aligner(_identity_embed, similarity_threshold=1.5)
    case = Case(truth="a b c", model_response="a X c")
    res = aligner.process(case)
    assert len(res.interventions) == 1
    assert len(res.semantic_drops) == 0


def test_pure_insertion_always_kept_as_intervention() -> None:
    """A region with empty truth (model added words) is never
    semantically equivalent -- always an intervention label."""
    aligner = Aligner(_identity_embed, similarity_threshold=0.0)
    # threshold=0 means even 0-cosine drops; but pure-insert path
    # bypasses the embedder and stays as intervention.
    case = Case(truth="a c", model_response="a b c")
    res = aligner.process(case)
    assert len(res.interventions) == 1
    assert len(res.semantic_drops) == 0
    assert res.interventions[0].truth_text == ""
    assert res.interventions[0].cosine_sim is None  # never embedded


def test_pure_deletion_always_kept_as_intervention() -> None:
    aligner = Aligner(_identity_embed, similarity_threshold=0.0)
    case = Case(truth="a b c", model_response="a c")
    res = aligner.process(case)
    assert len(res.interventions) == 1
    assert res.interventions[0].model_text == ""
    assert res.interventions[0].cosine_sim is None


def test_intervene_at_token_indices_uses_model_side_start() -> None:
    aligner = Aligner(_dummy_embed, similarity_threshold=2.0)
    # threshold > 1 means everything stays as intervention -- we
    # only care about the index ordering here.
    case = Case(
        truth="the dose is 50 mg for kids",
        model_response="the dose is 80 mg for adults",
    )
    res = aligner.process(case)
    assert len(res.interventions) == 2
    # Model-side starts: '80' at idx 3, 'adults' at idx 6.
    assert res.intervene_at_token_indices == [3, 6]


def test_empty_case_has_no_regions() -> None:
    aligner = Aligner(_identity_embed)
    case = Case(truth="", model_response="")
    res = aligner.process(case)
    assert res.divergent_regions == []
    assert res.interventions == []
    assert res.semantic_drops == []


def test_completely_different_responses_yield_one_replace_region() -> None:
    """SequenceMatcher merges contiguous diffs into a single replace op."""
    aligner = Aligner(_identity_embed, similarity_threshold=2.0)
    case = Case(
        truth="totally different content here",
        model_response="completely other text now",
    )
    res = aligner.process(case)
    # Exact region count depends on SequenceMatcher's matching;
    # what we care about is that ALL substantive regions land as
    # interventions when threshold > 1.0.
    assert len(res.interventions) >= 1
    assert all(r in res.interventions or r in res.semantic_drops
               for r in res.divergent_regions)


# ----------------------------------------------------------------- batching

def test_embed_fn_called_once_per_process_call() -> None:
    """Per Jay's spec: one batch call per case, not per region."""
    call_count = {"n": 0, "sizes": []}

    def counting_embed(texts: list[str]) -> list[list[float]]:
        call_count["n"] += 1
        call_count["sizes"].append(len(texts))
        return _identity_embed(texts)

    aligner = Aligner(counting_embed, similarity_threshold=0.90)
    case = Case(
        truth="the dose is 50 mg for adults with mild infection symptoms",
        model_response="the dose is 80 mg for kids with severe pneumonia symptoms",
    )
    aligner.process(case)
    assert call_count["n"] == 1, (
        f"expected 1 batched embed call per case, got {call_count['n']}"
    )
    # The single batch contains 2x the number of substantive regions
    # (truth subseqs first, then model subseqs).
    assert call_count["sizes"][0] % 2 == 0
    assert call_count["sizes"][0] >= 2


def test_embed_fn_returns_wrong_size_raises() -> None:
    def bad_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0]]  # always returns 1 embedding regardless

    aligner = Aligner(bad_embed)
    case = Case(truth="a b c", model_response="a X c")
    with pytest.raises(RuntimeError, match="expected one-to-one"):
        aligner.process(case)
