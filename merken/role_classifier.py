"""Few-shot semantic role classifier for merken events.

Production wrapper around the V2 STATE_CHANGE_REPORT taxonomy validated
in #39b (PR #42). Classifies an event text into one of four roles via
mean-cosine similarity to per-role prototypes.

Roles
-----
- ``state_change_report`` (SCR): a confirmed change to system state,
  framed as a factual report of what is now in production. Past tense
  + concrete state-change verbs + specific entity. No "X over Y"
  comparative structure.
- ``investigation``: an open inquiry being actively diagnosed.
- ``observation``: a factual report of state or events without
  selecting between options or actively inquiring.
- ``preference``: a subjective stance, recommendation, or positional
  preference without commitment.

See ``experiments/role_markers/prototype_style_guide.md`` (V2 section)
for full role definitions and authoring rules.

Validation status
-----------------
3-seed run on the canonical V2 eval set (#39b, 2026-04-28):

    macro-F1 mean = 0.929 +- 0.007 across seeds 42, 43, 44
    SCR recall    = 100% pooled (60/60)
    INVESTIGATION = 90.0% pooled
    OBSERVATION   = 86.7% pooled
    PREFERENCE    = 95.0% pooled

Pre-registered STRONG gate met on all 3 seeds.

Quick start
-----------

>>> from merken.role_classifier import RoleClassifier
>>> clf = RoleClassifier.default()  # loads bundled prototypes + vstash embedder
>>> result = clf.classify("Production now runs Vespa as of 2026-04-15.")
>>> result.role
'state_change_report'
>>> result.confidence  # margin between top-1 and top-2 mean similarities
0.067

For batch classification:

>>> results = clf.classify_batch([text1, text2, text3])
>>> [r.role for r in results]
['state_change_report', 'preference', 'investigation']

Limitations
-----------
- Only validated on BGE-small-en-v1.5 and multilingual-MiniLM-L12-v2
  (#38b cross-embedder check). Other embedders may shift the cosine
  thresholds.
- Validated on synthetic events authored to match the V2 style guide.
  Real organic content may have different surface lexical patterns.
  Top-1 confidence margin is reported on every classification so a
  caller can apply a reject threshold (e.g. ``confidence < 0.05``)
  for ambiguous events.
- The taxonomy explicitly excludes events outside the 4 roles
  (e.g. "scheduled", "deferred", "blocked"). Out-of-taxonomy events
  produce a low-confidence prediction; callers should handle them
  rather than treating the predicted role as ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

# Pre-registered V2 role taxonomy. Frozen with the prototypes it ships
# with -- changing this tuple requires a new prototype set + revalidation.
ROLES = ("state_change_report", "investigation", "observation", "preference")


@dataclass(frozen=True)
class RoleClassification:
    """Result of classifying a single event."""

    role: str
    """Predicted role: one of ROLES."""

    confidence: float
    """Margin between top-1 and top-2 mean similarities. Higher means
    the classifier was more decisive. Use this for reject-option
    filtering on ambiguous events (e.g. ``< 0.05`` is "uncertain")."""

    role_similarities: dict[str, float]
    """Mean cosine similarity to each role's prototypes. Iteration order
    matches the classifier's taxonomy (the order of keys in the
    ``prototypes`` dict, defaulting to ROLES for the bundled set)."""


def _load_prototypes_default() -> dict[str, list[dict]]:
    """Load the bundled V2 prototype set."""
    try:
        text = (resources.files("merken.data") / "role_prototypes_v2.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        # Fallback for editable installs where importlib.resources may
        # not see the data file.
        path = Path(__file__).parent / "data" / "role_prototypes_v2.json"
        text = path.read_text()
    raw = json.loads(text)
    if "roles" not in raw:
        raise RuntimeError("prototype JSON missing 'roles' key")
    out: dict[str, list[dict]] = {}
    for role in ROLES:
        if role not in raw["roles"]:
            raise RuntimeError(
                f"prototype JSON missing role {role!r} -- bundled "
                f"data is corrupt or stale"
            )
        out[role] = list(raw["roles"][role])
        if not out[role]:
            raise RuntimeError(f"prototype JSON has empty pool for {role!r}")
    return out


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


class RoleClassifier:
    """Prototype-based few-shot classifier for the V2 role taxonomy.

    Lazy embedding: prototypes are embedded on first ``classify`` /
    ``classify_batch`` call, not at construction time. This keeps the
    constructor cheap and makes the classifier safe to construct in
    contexts where vstash is not yet configured.

    Parameters
    ----------
    embed_fn:
        Callable that takes a list of texts and returns a 2-D array of
        embeddings (one row per input). For production use, the default
        constructor wires this to ``vstash.embed.embed_texts``.
    model_name:
        Embedding model identifier passed to ``embed_fn``.
    prototypes:
        Optional override of the bundled prototype set. Use only when
        validating a new taxonomy; the bundled set is the one validated
        in #39b. The taxonomy (set of valid role labels) is derived
        from this dict's keys -- the global ``ROLES`` constant only
        describes the bundled default. The classifier supports
        arbitrary role names and any number of roles >= 1.
    """

    def __init__(
        self,
        *,
        embed_fn: Callable[[list[str], str], np.ndarray],
        model_name: str,
        prototypes: dict[str, list[dict]] | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._model_name = model_name
        self._prototypes = prototypes if prototypes is not None else _load_prototypes_default()
        self._roles: tuple[str, ...] = tuple(self._prototypes.keys())
        if not self._roles:
            raise ValueError("prototypes must define at least one role")
        for role in self._roles:
            if not self._prototypes[role]:
                raise ValueError(f"prototypes for role {role!r} is empty")
        self._proto_unit_by_role: dict[str, np.ndarray] | None = None  # lazy

    @classmethod
    def default(cls, vstash_memory: object | None = None) -> RoleClassifier:
        """Construct a classifier wired to vstash's default embedder.

        Parameters
        ----------
        vstash_memory:
            Optional ``vstash.Memory`` instance used to resolve the
            store-pinned embedding model (avoids vector-space drift
            against an existing store). When omitted, falls back to
            ``vstash.config.EmbeddingsConfig().model``.
        """
        from vstash.embed import embed_texts as _embed_texts

        if vstash_memory is not None:
            from merken.memory import _resolve_vstash_embed_model
            model_name = _resolve_vstash_embed_model(vstash_memory)
        else:
            from vstash.config import EmbeddingsConfig
            model_name = EmbeddingsConfig().model
            if not model_name:
                raise RuntimeError(
                    "vstash.config.EmbeddingsConfig().model is empty; "
                    "cannot construct a default RoleClassifier"
                )

        def _embed(texts: list[str], model: str) -> np.ndarray:
            arr = np.asarray(_embed_texts(texts, model_name=model), dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] != len(texts):
                raise RuntimeError(
                    f"embed_texts returned shape {arr.shape!r} for "
                    f"{len(texts)} inputs"
                )
            return arr

        return cls(embed_fn=_embed, model_name=model_name)

    def _ensure_prototypes_embedded(self) -> dict[str, np.ndarray]:
        """Embed all prototypes once and cache them."""
        if self._proto_unit_by_role is not None:
            return self._proto_unit_by_role
        proto_texts: list[str] = []
        cursor: list[tuple[str, int]] = []  # (role, len) markers
        for role in self._roles:
            texts = [p["text"] for p in self._prototypes[role]]
            cursor.append((role, len(texts)))
            proto_texts.extend(texts)
        proto_vec = self._embed_fn(proto_texts, self._model_name)
        proto_unit = _l2_normalize(proto_vec)
        out: dict[str, np.ndarray] = {}
        idx = 0
        for role, count in cursor:
            out[role] = proto_unit[idx:idx + count]
            idx += count
        self._proto_unit_by_role = out
        return out

    def classify(self, text: str) -> RoleClassification:
        """Classify a single event text. Convenience wrapper over ``classify_batch``."""
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: Sequence[str]) -> list[RoleClassification]:
        """Classify a batch of event texts.

        Returns one RoleClassification per input, in input order.

        Cosine similarity to each role is computed as the mean cosine
        across that role's prototypes (Snell, Swersky, Zemel 2017).
        Predicted role is the argmax. Confidence is the margin between
        top-1 and top-2 mean similarities.
        """
        if not texts:
            return []
        proto_unit_by_role = self._ensure_prototypes_embedded()

        eval_vec = self._embed_fn(list(texts), self._model_name)
        eval_unit = _l2_normalize(eval_vec)

        n = eval_unit.shape[0]
        n_roles = len(self._roles)
        role_means = np.zeros((n, n_roles), dtype=np.float64)
        for r_idx, role in enumerate(self._roles):
            sub = proto_unit_by_role[role]
            sims = eval_unit @ sub.T  # (n, P_role)
            role_means[:, r_idx] = sims.mean(axis=1)

        pred_idx = np.argmax(role_means, axis=1)
        sorted_means = -np.sort(-role_means, axis=1)
        top1 = sorted_means[:, 0]
        if n_roles >= 2:
            top2 = sorted_means[:, 1]
            margin = top1 - top2
        else:
            # Single-role taxonomy: no alternative to compete with, so
            # the prediction is trivially decisive. +inf is the natural
            # margin for "nothing to subtract".
            margin = np.full_like(top1, np.inf)

        out: list[RoleClassification] = []
        for i in range(n):
            out.append(
                RoleClassification(
                    role=self._roles[int(pred_idx[i])],
                    confidence=float(margin[i]),
                    role_similarities={
                        role: float(role_means[i, r_idx])
                        for r_idx, role in enumerate(self._roles)
                    },
                )
            )
        return out

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def roles(self) -> tuple[str, ...]:
        """Roles in this classifier's taxonomy (order matches the
        ``prototypes`` dict's key order)."""
        return self._roles

    @property
    def prototype_count(self) -> dict[str, int]:
        """Number of prototypes per role."""
        return {role: len(self._prototypes[role]) for role in self._roles}
