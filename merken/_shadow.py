"""Env-var activation for shadow mode and graduation.

Two roles for a classifier (nanoGPT / LLM) via env:

- ``MERKEN_SHADOW=nanogpt|llm`` -- auditing-only. The default
  ``HeuristicWriteDecider`` stays primary and controls writes; the
  classifier only annotates the audit reason with its prediction and
  an agree/disagree tag. Bootstrap mode.

- ``MERKEN_PRIMARY=nanogpt|llm`` -- graduation. The classifier becomes
  authoritative, chained *after* ``HeuristicWriteDecider`` so that the
  hygiene gates (empty, too-short, too-long, exact-dup) still fire
  before the classifier sees the event. Only use this once the shadow
  logs + oracular labels have shown the classifier beats the gate-only
  baseline on held-out content.

Only one of ``MERKEN_SHADOW`` / ``MERKEN_PRIMARY`` should be set at a
time. If both are set the primary role wins (ignores SHADOW) -- a
deliberately conservative tie-breaker: whoever typed ``PRIMARY`` meant
it.

Torch is imported **at module evaluation** (not inside a function) when
either role is enabled, so that ``from merken import ...`` causes torch
to load BEFORE vstash/fastembed. That's the Mistake #10 order
requirement (notes/nanogpt-training-log.md): the first native library
to load wins the macOS process; later ones can segfault. Keep this
module as the first import in ``merken/__init__.py``.

Supported backends (shared between SHADOW and PRIMARY):

- ``nanogpt``
    + ``MERKEN_<role>_NANOGPT_CKPT=/path/to/ckpt.pt``
    + ``MERKEN_<role>_NANOGPT_META=/path/to/meta.pkl``
    + ``MERKEN_<role>_NANOGPT_CALIBRATOR=default`` (optional)
      Wraps the decider with a post-hoc CalibrationHead so
      ``Decision.reason`` includes ``P_cal=x.xxx`` and
      ``Decision.confidence`` reports the calibrated value. Does NOT
      change pass/fail -- threshold stays on raw P(D). Accepts:
        * ``default`` or ``shipped`` -> use shipped
          ``merken/classifiers/calibration_v7.json``.
        * any filesystem path -> load that JSON instead.
        * unset -> no calibration (raw P(D) shown).
      Calibrator load errors (bad path, malformed JSON) print a
      warning to stderr; the classifier still loads without
      calibration.
- ``llm``
    + ``MERKEN_<role>_LLM_MODEL=google/gemma-3-270m-it``
    + ``MERKEN_<role>_LLM_DEVICE=cpu`` (optional, default ``cpu``)
- ``jev`` (network, opt-in; no torch)
    + ``OPENROUTER_API_KEY`` or ``TYPESAFE_API_KEY`` in the environment
    + ``MERKEN_<role>_JEV_THRESHOLD=0.6`` (optional; below it, write)

where ``<role>`` is ``SHADOW`` or ``PRIMARY``.

Anything else (empty / unset / unknown) -> no classifier wired.
"""

from __future__ import annotations

import os

_ENABLED_VALUES = {"nanogpt", "llm", "jev"}
# Only these backends need torch loaded before vstash (Mistake #10).
_TORCH_VALUES = {"nanogpt", "llm"}
_SHADOW_KIND = os.environ.get("MERKEN_SHADOW", "").strip().lower()
_PRIMARY_KIND = os.environ.get("MERKEN_PRIMARY", "").strip().lower()
# Local-oracle labeling also needs torch loaded first (Mistake #10):
# ``merken audit --label-with local`` constructs an LLMWriteDecider
# which imports transformers + torch. If vstash has already loaded
# fastembed, the process segfaults on first model forward.
_LABEL_LLM_MODEL = os.environ.get("MERKEN_LABEL_LLM_MODEL", "").strip()

# Eagerly import torch if any role that needs torch is enabled. This
# must happen before any other merken module that transitively imports
# vstash (fastembed/ONNX). See notes/nanogpt-training-log.md
# Mistake #10.
if (
    _SHADOW_KIND in _TORCH_VALUES
    or _PRIMARY_KIND in _TORCH_VALUES
    or _LABEL_LLM_MODEL
):
    import torch  # noqa: F401


def _resolve_calibrator(role: str):
    """Return a CalibrationHead or None based on env.

    ``MERKEN_<role>_NANOGPT_CALIBRATOR`` values:
      - unset / empty -> None (no calibration)
      - ``default`` or ``shipped`` -> use
        ``merken/classifiers/calibration_v7.json``
      - any other value -> treat as a filesystem path to a JSON head

    Errors loading the JSON raise; the caller can decide whether to
    propagate or degrade gracefully. ``_build_classifier`` explicitly
    catches + logs so a typoed calibrator path produces a visible
    stderr warning rather than silently disabling calibration.
    """
    from pathlib import Path

    raw = (os.environ.get(f"MERKEN_{role}_NANOGPT_CALIBRATOR") or "").strip()
    if not raw:
        return None
    from merken.classifiers.calibration import CalibrationHead

    if raw.lower() in ("default", "shipped"):
        path = Path(__file__).resolve().parent / "classifiers" / "calibration_v7.json"
    else:
        path = Path(raw)
    return CalibrationHead.from_json(path, source=f"{role}_env")


def _build_classifier(kind: str, role: str):
    """Construct the classifier for ``role`` (``SHADOW`` / ``PRIMARY``).

    Calibrator load errors are caught here and printed to stderr so
    the classifier itself still loads. ``Memory._default_write_decider``
    catches every exception from this function as a fail-safe (model
    missing -> heuristic), so a calibrator misconfig would otherwise
    silently disable the whole classifier -- not what we want.
    """
    if kind == "nanogpt":
        from merken.classifiers.nanogpt import NanoGPTWriteDecider

        ckpt = os.environ.get(f"MERKEN_{role}_NANOGPT_CKPT")
        meta = os.environ.get(f"MERKEN_{role}_NANOGPT_META")
        if not (ckpt and meta):
            raise RuntimeError(
                f"MERKEN_{role}=nanogpt requires MERKEN_{role}_NANOGPT_CKPT "
                f"and MERKEN_{role}_NANOGPT_META env vars."
            )
        try:
            calibrator = _resolve_calibrator(role)
        except Exception as e:
            import sys
            print(
                f"merken: failed to load {role} calibrator "
                f"(MERKEN_{role}_NANOGPT_CALIBRATOR): "
                f"{type(e).__name__}: {e}. "
                f"Continuing without calibration.",
                file=sys.stderr,
            )
            calibrator = None
        return NanoGPTWriteDecider(ckpt, meta, calibrator=calibrator)

    if kind == "llm":
        from merken.classifiers.llm import LLMWriteDecider

        model = os.environ.get(f"MERKEN_{role}_LLM_MODEL")
        if not model:
            raise RuntimeError(
                f"MERKEN_{role}=llm requires MERKEN_{role}_LLM_MODEL env var."
            )
        device = os.environ.get(f"MERKEN_{role}_LLM_DEVICE", "cpu")
        return LLMWriteDecider(model_name=model, device=device)

    if kind == "jev":
        from merken.classifiers.jev import JevWriteDecider

        raw = os.environ.get(f"MERKEN_{role}_JEV_THRESHOLD", "").strip()
        threshold = float(raw) if raw else 0.6
        # JevClient resolves OPENROUTER_API_KEY / TYPESAFE_API_KEY itself
        # and raises if neither is set; Memory catches that and falls
        # back to the heuristic, like every other misconfigured backend.
        return JevWriteDecider(confidence_threshold=threshold)

    raise RuntimeError(f"unknown MERKEN_{role} backend: {kind!r}")


def load_shadow_from_env():
    """Return a shadow ``WriteDecider`` constructed from env, or ``None``.

    Only returns a classifier when ``MERKEN_SHADOW`` is set AND
    ``MERKEN_PRIMARY`` is NOT set. When PRIMARY wins, there is no
    "shadow" decider -- graduation mode is strictly about putting the
    classifier on the write path.
    """
    if _PRIMARY_KIND in _ENABLED_VALUES:
        return None
    if _SHADOW_KIND not in _ENABLED_VALUES:
        return None
    return _build_classifier(_SHADOW_KIND, "SHADOW")


def load_primary_from_env():
    """Return the graduated classifier constructed from env, or ``None``.

    When this returns a non-None value, ``Memory`` will chain it AFTER
    a ``HeuristicWriteDecider`` gate so hygiene rules still fire, and
    will NOT wrap it in shadow mode.
    """
    if _PRIMARY_KIND not in _ENABLED_VALUES:
        return None
    return _build_classifier(_PRIMARY_KIND, "PRIMARY")
