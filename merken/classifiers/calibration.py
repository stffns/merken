"""Post-hoc calibration head for nanoGPT write deciders.

The head is a 9-parameter logistic regression fitted in H1
(see engram experiments/nanogpt/HYPOTHESES.md):

    P_cal = sigmoid(
        intercept
      + w_logit * logit(P_raw)
      + w_code_fence   * has_code_fence
      + w_inline_code  * has_inline_code
      + w_mdtable      * has_markdown_table
      + w_numbers      * has_numbers
      + w_filepaths    * has_file_paths
      + w_short        * is_short       # len < 300
      + w_long         * is_long        # len >= 1000
      + w_ood          * is_ood         # OOD-like, see note
    )

``is_ood`` is a caller-provided hint. Production callers should
default to ``False`` (treat content as in-distribution); the flag is
primarily useful for offline evaluation against public datasets.

The head does NOT change the write/skip decision. It only rewrites
the displayed probability so users and downstream code see an
honestly-calibrated confidence value.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-6

CODE_FENCE = re.compile(r"```")
INLINE_CODE = re.compile(r"`[^`\n]{2,}`")
TABLE_ROW = re.compile(r"\|[^\n]*\|[^\n]*\|")
FILE_PATH = re.compile(r"\b\S+\.(?:py|js|ts|md|json|toml|yml|yaml|go|rs|sh|sql)\b")
NUMBER = re.compile(r"\d")


def _logit(p: float) -> float:
    """Safe logit.

    NaN propagates through ``min`` / ``max`` in CPython, so a naive
    clamp would silently produce NaN logits. Detect non-finite input
    and reject it so callers see the problem immediately instead of
    poisoning audit rows and downstream ``>= threshold`` checks.
    """
    if not math.isfinite(p):
        raise ValueError(f"calibrator received non-finite probability: {p!r}")
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class CalibrationHead:
    """Structure-conditional post-hoc calibrator.

    Weights are intended to be loaded from a JSON produced by the
    H1 / H10 / H11 calibration sweep scripts. The JSON schema looks
    like::

        {
          "head": {
            "intercept": 1.05,
            "coefficients": {
              "logit_P(D)": 0.44,
              "has_code_fence": 0.20,
              ...
              "short_X_numbers": 0.12,        # interaction, optional
              "ood_X_short": 1.02,            # interaction, optional
              ...
            }
          }
        }

    Interaction weights default to 0.0 when absent so older JSON
    artifacts (H1 / H11) still load correctly and produce the same
    numbers as before.
    """

    intercept: float
    w_logit: float
    w_code_fence: float = 0.0
    w_inline_code: float = 0.0
    w_markdown_table: float = 0.0
    w_numbers: float = 0.0
    w_file_paths: float = 0.0
    w_short: float = 0.0
    w_long: float = 0.0
    w_ood: float = 0.0
    # Interaction terms (H10). Default 0 so old JSON artifacts load.
    w_short_numbers: float = 0.0
    w_short_file_paths: float = 0.0
    w_short_inline_code: float = 0.0
    w_short_markdown_table: float = 0.0
    w_ood_short: float = 0.0
    # Included so downstream code can tell which experiment produced
    # this head; reliable for logging, unreliable as version control.
    source: str = field(default="unknown")

    @classmethod
    def from_json(cls, path: str | Path, *, source: str | None = None) -> CalibrationHead:
        p = Path(path)
        data = json.loads(p.read_text())
        head = data.get("head") or data  # accept raw head dict too
        coefs = head.get("coefficients") or {}

        def _get(*keys: str) -> float:
            """Pull a coefficient and verify it is finite."""
            for k in keys:
                if k in coefs:
                    v = float(coefs[k])
                    break
            else:
                return 0.0
            if not math.isfinite(v):
                raise ValueError(
                    f"non-finite weight {keys[0]!r}={v!r} in {p}"
                )
            return v

        intercept = float(head.get("intercept", 0.0))
        if not math.isfinite(intercept):
            raise ValueError(f"non-finite intercept in {p}")

        return cls(
            intercept=intercept,
            w_logit=_get("logit_P(D)", "logit_p_d"),
            w_code_fence=_get("has_code_fence"),
            w_inline_code=_get("has_inline_code"),
            w_markdown_table=_get("has_markdown_table"),
            w_numbers=_get("has_numbers"),
            w_file_paths=_get("has_file_paths"),
            w_short=_get("is_short"),
            w_long=_get("is_long"),
            w_ood=_get("is_ood"),
            w_short_numbers=_get("short_X_numbers"),
            w_short_file_paths=_get("short_X_file_paths"),
            w_short_inline_code=_get("short_X_inline_code"),
            w_short_markdown_table=_get("short_X_markdown_table"),
            w_ood_short=_get("ood_X_short"),
            source=source or str(p),
        )

    def features(self, text: str, *, is_ood: bool = False) -> dict[str, int]:
        """Extract the 8 binary tags the head uses (logit is external)."""
        return {
            "has_code_fence": 1 if CODE_FENCE.search(text) else 0,
            "has_inline_code": 1 if INLINE_CODE.search(text) else 0,
            "has_markdown_table": 1 if TABLE_ROW.search(text) else 0,
            "has_numbers": 1 if len(NUMBER.findall(text)) >= 3 else 0,
            "has_file_paths": 1 if FILE_PATH.search(text) else 0,
            "is_short": 1 if len(text) < 300 else 0,
            "is_long": 1 if len(text) >= 1000 else 0,
            "is_ood": 1 if is_ood else 0,
        }

    def calibrate(self, p_raw: float, text: str, *, is_ood: bool = False) -> float:
        """Apply the head to a raw P(D), returning calibrated P(D).

        ``p_raw`` must be finite and will be clamped into ``[EPS, 1-EPS]``
        before the logit transform. Non-finite ``p_raw`` (NaN / inf)
        raises ``ValueError`` rather than silently producing NaN.
        """
        f = self.features(text, is_ood=is_ood)
        z = (
            self.intercept
            + self.w_logit * _logit(p_raw)
            + self.w_code_fence * f["has_code_fence"]
            + self.w_inline_code * f["has_inline_code"]
            + self.w_markdown_table * f["has_markdown_table"]
            + self.w_numbers * f["has_numbers"]
            + self.w_file_paths * f["has_file_paths"]
            + self.w_short * f["is_short"]
            + self.w_long * f["is_long"]
            + self.w_ood * f["is_ood"]
            # Interaction terms (H10). Defaults are 0 for old heads.
            + self.w_short_numbers * f["is_short"] * f["has_numbers"]
            + self.w_short_file_paths * f["is_short"] * f["has_file_paths"]
            + self.w_short_inline_code * f["is_short"] * f["has_inline_code"]
            + self.w_short_markdown_table * f["is_short"] * f["has_markdown_table"]
            + self.w_ood_short * f["is_ood"] * f["is_short"]
        )
        return _sigmoid(z)
