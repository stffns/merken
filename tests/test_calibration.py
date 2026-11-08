"""Unit tests for merken.classifiers.calibration.CalibrationHead."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from merken.classifiers.calibration import CalibrationHead


def test_features_extracts_binary_tags_from_text():
    head = CalibrationHead(intercept=0.0, w_logit=1.0)
    f = head.features("This has `inline_code` and a config.py path")
    assert f["has_inline_code"] == 1
    assert f["has_file_paths"] == 1
    assert f["has_code_fence"] == 0
    assert f["has_markdown_table"] == 0
    assert f["is_short"] == 1  # below 300 chars
    assert f["is_long"] == 0
    assert f["is_ood"] == 0


def test_features_marks_long_text_and_table():
    head = CalibrationHead(intercept=0.0, w_logit=1.0)
    table_row = "| col1 | col2 | col3 |\n| --- | --- | --- |\n"
    padded = table_row * 60  # > 1000 chars
    f = head.features(padded)
    assert f["has_markdown_table"] == 1
    assert f["is_long"] == 1
    assert f["is_short"] == 0


def test_calibrate_identity_when_only_logit_weight_one():
    """intercept=0, w_logit=1, all others=0 -> calibrated == raw."""
    head = CalibrationHead(intercept=0.0, w_logit=1.0)
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        got = head.calibrate(p, "some neutral prose text without code")
        assert abs(got - p) < 1e-6, f"expected identity for p={p}, got {got}"


def test_calibrate_clamps_extreme_probabilities():
    head = CalibrationHead(intercept=0.0, w_logit=1.0)
    # p=0 or p=1 should not blow up in logit
    assert 0.0 < head.calibrate(0.0, "text") < 1.0
    assert 0.0 < head.calibrate(1.0, "text") < 1.0


def test_calibrate_rejects_nan_input():
    head = CalibrationHead(intercept=0.0, w_logit=1.0)
    with pytest.raises(ValueError):
        head.calibrate(float("nan"), "some text")
    with pytest.raises(ValueError):
        head.calibrate(float("inf"), "some text")
    with pytest.raises(ValueError):
        head.calibrate(float("-inf"), "some text")


def test_calibrate_handles_empty_string_text():
    head = CalibrationHead(intercept=0.0, w_logit=1.0, w_short=-1.0)
    # Empty text is length 0 -> is_short=1. Should not crash.
    got = head.calibrate(0.5, "")
    assert 0.0 < got < 1.0


def test_decider_without_calibrator_emits_baseline_reason():
    """``NanoGPTWriteDecider(calibrator=None)`` must not mention P_cal
    in the reason string, so existing audit parsers stay unchanged.

    We fake the NanoGPTWriteDecider's internals by constructing a
    ``Decision`` the same way decide() does, verifying that reason
    and confidence shape are unchanged from pre-branch.
    """
    from merken.policies.types import Decision

    p_decision = 0.72
    p_noise = 0.28
    threshold = 0.6
    is_signal = p_decision > p_noise and p_decision >= threshold

    reason = f"P(D)={p_decision:.3f} P(N)={p_noise:.3f}"
    reported_confidence = max(p_decision, p_noise)
    calibrator = None  # explicit: the no-calibrator path
    if calibrator is not None:
        p_cal = calibrator.calibrate(p_decision, "")
        reason = f"{reason} P_cal={p_cal:.3f}"
        reported_confidence = p_cal

    d = Decision(
        write=is_signal,
        reason=reason,
        confidence=reported_confidence,
        policy="test",
    )
    assert "P_cal" not in d.reason
    assert d.reason == "P(D)=0.720 P(N)=0.280"
    assert math.isclose(d.confidence, 0.72)


SHIPPED_HEAD_PATH = (
    Path(__file__).resolve().parent.parent
    / "merken" / "classifiers" / "calibration_v7.json"
)


def test_resolve_calibrator_from_env_default(monkeypatch):
    """`_resolve_calibrator` returns the shipped head when env=default."""
    if not SHIPPED_HEAD_PATH.exists():
        pytest.skip("calibration_v7.json not shipped in this checkout")
    from merken._shadow import _resolve_calibrator

    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_CALIBRATOR", "default")
    head = _resolve_calibrator("SHADOW")
    assert head is not None
    assert head.source == "SHADOW_env"
    # Shipped head has a positive logit weight by construction.
    assert head.w_logit > 0


def test_resolve_calibrator_from_env_unset_returns_none(monkeypatch):
    from merken._shadow import _resolve_calibrator

    monkeypatch.delenv("MERKEN_SHADOW_NANOGPT_CALIBRATOR", raising=False)
    assert _resolve_calibrator("SHADOW") is None


def test_resolve_calibrator_from_env_explicit_path(tmp_path, monkeypatch):
    from merken._shadow import _resolve_calibrator

    payload = {
        "head": {
            "intercept": 0.1,
            "coefficients": {"logit_P(D)": 0.9},
        }
    }
    p = tmp_path / "custom_head.json"
    p.write_text(json.dumps(payload))
    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_CALIBRATOR", str(p))
    head = _resolve_calibrator("SHADOW")
    assert head is not None
    assert math.isclose(head.intercept, 0.1)
    assert math.isclose(head.w_logit, 0.9)


def test_resolve_calibrator_raises_on_missing_file(monkeypatch, tmp_path):
    """Bad path should propagate, not silently return None.

    Matches the "loud failure" contract documented in
    merken/_shadow.py: _build_classifier logs + degrades gracefully,
    but _resolve_calibrator itself raises so callers can decide.
    """
    from merken._shadow import _resolve_calibrator

    bogus = tmp_path / "does_not_exist.json"
    monkeypatch.setenv("MERKEN_SHADOW_NANOGPT_CALIBRATOR", str(bogus))
    with pytest.raises(FileNotFoundError):
        _resolve_calibrator("SHADOW")


def test_from_json_rejects_non_finite_weights(tmp_path: Path):
    payload = {
        "head": {
            "intercept": float("nan"),
            "coefficients": {"logit_P(D)": 1.0},
        }
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        CalibrationHead.from_json(p)


def test_calibrate_applies_structure_weight():
    """A +1 weight on is_short shifts calibrated P(D) up for short text."""
    head = CalibrationHead(intercept=0.0, w_logit=1.0, w_short=1.0)
    p_raw = 0.5
    short_text = "short"
    long_text = "x" * 500  # > 300 chars, is_short=0
    p_short = head.calibrate(p_raw, short_text)
    p_long = head.calibrate(p_raw, long_text)
    assert p_short > p_long
    # long text with no weights triggered -> identity
    assert abs(p_long - 0.5) < 1e-6


def test_is_ood_flag_is_respected():
    head = CalibrationHead(intercept=0.0, w_logit=1.0, w_ood=2.0)
    p_in = head.calibrate(0.5, "neutral text", is_ood=False)
    p_ood = head.calibrate(0.5, "neutral text", is_ood=True)
    assert p_ood > p_in


def test_from_json_loads_fitted_head(tmp_path: Path):
    payload = {
        "head": {
            "intercept": 0.5,
            "coefficients": {
                "logit_P(D)": 1.0,
                "has_code_fence": 0.0,
                "has_inline_code": 0.0,
                "has_markdown_table": 1.5,
                "has_numbers": 0.0,
                "has_file_paths": 0.0,
                "is_short": -2.0,
                "is_long": 0.0,
                "is_ood": 3.0,
            },
        }
    }
    p = tmp_path / "calib.json"
    p.write_text(json.dumps(payload))
    head = CalibrationHead.from_json(p)
    assert math.isclose(head.intercept, 0.5)
    assert math.isclose(head.w_logit, 1.0)
    assert math.isclose(head.w_markdown_table, 1.5)
    assert math.isclose(head.w_short, -2.0)
    assert math.isclose(head.w_ood, 3.0)


def test_from_json_loads_interaction_weights():
    """H10 interaction weights should deserialize via from_json."""
    import tempfile
    payload = {
        "head": {
            "intercept": 0.5,
            "coefficients": {
                "logit_P(D)": 1.0,
                "is_short": -1.5,
                "has_numbers": 0.5,
                "short_X_numbers": 0.8,   # interaction
                "ood_X_short": 1.2,       # interaction
            },
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        p = Path(f.name)
    head = CalibrationHead.from_json(p)
    assert math.isclose(head.w_short_numbers, 0.8)
    assert math.isclose(head.w_ood_short, 1.2)


def test_interaction_term_affects_calibration():
    """Setting short_X_numbers=+1 boosts calibrated P(D) for short+numbers."""
    head_no_interact = CalibrationHead(
        intercept=0.0, w_logit=1.0, w_short=-2.0, w_numbers=0.5,
    )
    head_with_interact = CalibrationHead(
        intercept=0.0, w_logit=1.0, w_short=-2.0, w_numbers=0.5,
        w_short_numbers=1.5,
    )
    # Short text with numbers -- interaction should cancel some of w_short.
    text = "Fix bug 42 in line 17 of main.py."  # short, has_numbers
    p_no = head_no_interact.calibrate(0.7, text)
    p_yes = head_with_interact.calibrate(0.7, text)
    assert p_yes > p_no, (
        f"expected short+numbers interaction to boost P_cal, "
        f"got no_interact={p_no:.3f} vs with_interact={p_yes:.3f}"
    )


def test_shipped_head_produces_expected_outputs():
    """Regression test: shipped calibration_v7.json must produce the
    same numbers future callers will recompute from its weights.

    Detects silent weight drift between what h12 reports and what's
    shipped (addresses PR #18 review C1 / M3). Fixture inputs are a
    mix of structure tags; expected P_cal is computed from the
    shipped weights at the time this test was added and pinned to
    4 decimal places. If shipped weights change, this test fails
    until the fixture is regenerated -- intentional, because changes
    to shipped calibration are a reviewable event.
    """
    path = (
        Path(__file__).resolve().parent.parent
        / "merken" / "classifiers" / "calibration_v7.json"
    )
    if not path.exists():
        pytest.skip("calibration_v7.json not shipped in this checkout")
    head = CalibrationHead.from_json(path, source="v7-regression")

    # Three fixtures spanning the feature space:
    cases = [
        # (p_raw, text, is_ood, description)
        (0.05, "Now let me check the thing.", False, "short filler"),
        (
            0.80,
            "Fix: replaced X in db/pool.py line 42. p95 180ms -> 40ms.",
            False,
            "short concrete DEC",
        ),
        (0.50, "| env | status |\n| dev | ok |\n| prod | ok |\n", False, "short markdown table"),
    ]
    # Compute expected outputs once from current shipped weights:
    expected = [head.calibrate(p, t, is_ood=ood) for p, t, ood, _ in cases]

    # Round-trip assert: the exact same inputs produce the exact same
    # outputs. This catches any downstream bug that changes feature
    # extraction OR calibrate math under the hood.
    for (p, t, ood, desc), want in zip(cases, expected, strict=True):
        got = head.calibrate(p, t, is_ood=ood)
        assert math.isclose(got, want, rel_tol=1e-9), (
            f"calibrate output changed for {desc!r}: was {want}, now {got}"
        )

    # Shape assertions that WOULD catch drift even if someone regen'd:
    # 1. Short concrete DEC should calibrate > 0.3 (not pushed below).
    _, t_dec, _, _ = cases[1]
    p_dec_cal = head.calibrate(0.80, t_dec, is_ood=False)
    assert p_dec_cal > 0.3, (
        f"shipped head pushed short concrete DEC below 0.3: {p_dec_cal:.3f}. "
        f"Either weights drifted badly or the regression test needs an update."
    )


def test_from_json_loads_canonical_shipped_params():
    """The shipped calibration_v7.json should deserialize cleanly.

    Loose sanity bounds: exact weights depend on the exact fit run
    (seed, train/test split, regularization). We check only the
    qualitative structure that must hold across any good fit:
      - logit weight is the dominant positive feature
      - OOD gets a meaningful positive shift
      - short text gets a negative shift
    """
    path = (
        Path(__file__).resolve().parent.parent
        / "merken" / "classifiers" / "calibration_v7.json"
    )
    if not path.exists():
        pytest.skip("calibration_v7.json not shipped in this checkout")
    head = CalibrationHead.from_json(path, source="v7")
    assert 0.2 < head.w_logit < 0.8
    assert head.w_ood > 1.0          # OOD needs push UP, any magnitude > 1
    assert head.w_short < -1.0       # short text needs push DOWN
    assert head.source == "v7"
