"""Unit tests for merken.role_classifier.RoleClassifier.

Uses a fake embed_fn to keep the unit tests deterministic and offline.
For end-to-end validation against the real vstash embedder, see
experiments/role_markers/few_shot_classifier_v2.py which produced the
canonical 3-seed numbers (macro-F1 0.929, SCR recall 100%).
"""

from __future__ import annotations

import numpy as np
import pytest

from merken.role_classifier import (
    ROLES,
    RoleClassification,
    RoleClassifier,
    _load_prototypes_default,
)


@pytest.fixture
def fake_embed_fn():
    """Return an embed_fn that maps text -> deterministic vector.

    Each role gets a distinct anchor direction in 8-D space:
        state_change_report -> [1, 0, 0, 0, 0, 0, 0, 0]
        investigation       -> [0, 1, 0, 0, 0, 0, 0, 0]
        observation         -> [0, 0, 1, 0, 0, 0, 0, 0]
        preference          -> [0, 0, 0, 1, 0, 0, 0, 0]

    Texts containing role-specific keywords get a vector close to that
    role's anchor (with a small noise component). This keeps the
    classifier deterministic in tests.
    """
    role_anchors: dict[str, np.ndarray] = {}
    for i, role in enumerate(ROLES):
        v = np.zeros(8, dtype=np.float64)
        v[i] = 1.0
        role_anchors[role] = v

    role_keywords = {
        "state_change_report": ("has been running", "decommissioned", "rolled out", "since 2026"),
        "investigation": ("investigating", "looking into", "diagnosing", "root-causing"),
        "observation": ("shows", "noticed", "observed", "metrics indicate"),
        "preference": ("prefer", "lean toward", "would rather", "recommend"),
    }

    def _embed(texts: list[str], model_name: str) -> np.ndarray:
        vectors = []
        for t in texts:
            tl = t.lower()
            best_role = "state_change_report"
            best_score = 0
            for role, keywords in role_keywords.items():
                score = sum(1 for kw in keywords if kw in tl)
                if score > best_score:
                    best_score = score
                    best_role = role
            anchor = role_anchors[best_role].copy()
            # Add small per-text noise based on length so rerun is stable.
            noise = np.zeros(8, dtype=np.float64)
            noise[4] = (len(t) % 7) * 0.01
            noise[5] = (sum(ord(c) for c in t[:4]) % 11) * 0.01
            vectors.append(anchor + noise)
        return np.asarray(vectors, dtype=np.float64)

    return _embed


def test_load_prototypes_default_has_all_roles():
    protos = _load_prototypes_default()
    assert set(protos.keys()) == set(ROLES)
    for role in ROLES:
        assert len(protos[role]) >= 7, f"role {role!r} has fewer than 7 prototypes"


def test_load_prototypes_each_proto_has_required_fields():
    protos = _load_prototypes_default()
    for role in ROLES:
        for p in protos[role]:
            assert "id" in p
            assert "text" in p
            assert "topic" in p
            assert isinstance(p["text"], str) and p["text"]


def test_classifier_construction_does_not_embed(fake_embed_fn):
    """Lazy embedding: building a classifier should not call embed_fn."""
    call_log = []

    def _embed(texts, model_name):
        call_log.append(texts)
        return fake_embed_fn(texts, model_name)

    clf = RoleClassifier(embed_fn=_embed, model_name="fake-model")
    assert call_log == [], "constructor must not embed prototypes"
    assert clf.model_name == "fake-model"
    counts = clf.prototype_count
    for role in ROLES:
        assert counts[role] >= 7


def test_classify_single_scr(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    text = "Vespa has been running in production since 2026-04-15."
    result = clf.classify(text)
    assert isinstance(result, RoleClassification)
    assert result.role == "state_change_report"
    assert result.confidence > 0
    assert set(result.role_similarities.keys()) == set(ROLES)


def test_classify_single_preference(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    text = "Personally I would prefer keeping Stripe and absorbing the fee."
    result = clf.classify(text)
    assert result.role == "preference"
    assert result.confidence > 0


def test_classify_single_investigation(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    text = "Investigating why the Kafka consumer lag started this morning."
    result = clf.classify(text)
    assert result.role == "investigation"


def test_classify_single_observation(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    text = "metrics indicate Search QPS at a 40% week-over-week climb."
    result = clf.classify(text)
    assert result.role == "observation"


def test_classify_batch_returns_one_per_input(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    texts = [
        "Vespa has been running in production since 2026-04-15.",
        "Personally I would prefer keeping Stripe.",
        "Investigating the Kafka consumer lag.",
        "Logs observed a 40% week-over-week climb.",
    ]
    results = clf.classify_batch(texts)
    assert len(results) == 4
    assert results[0].role == "state_change_report"
    assert results[1].role == "preference"
    assert results[2].role == "investigation"
    assert results[3].role == "observation"


def test_classify_batch_empty_input(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    assert clf.classify_batch([]) == []


def test_prototypes_embedded_once(fake_embed_fn):
    """Repeated classify calls should not re-embed prototypes."""
    call_count = [0]

    def _embed(texts, model_name):
        call_count[0] += 1
        return fake_embed_fn(texts, model_name)

    clf = RoleClassifier(embed_fn=_embed, model_name="fake-model")
    clf.classify("first call")  # call 1: embed protos, call 2: embed input
    n_after_first = call_count[0]
    clf.classify("second call")  # call 3 only: input
    n_after_second = call_count[0]
    assert n_after_first == 2, "first classify should embed protos + input"
    assert n_after_second == 3, "second classify should reuse cached protos"


def test_role_similarities_in_valid_range(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    result = clf.classify("Vespa has been running in production since 2026-04-15.")
    for role, sim in result.role_similarities.items():
        assert -1.001 <= sim <= 1.001, (
            f"sim for {role!r} = {sim} outside [-1, 1] range"
        )


def test_confidence_is_top1_minus_top2(fake_embed_fn):
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    result = clf.classify("Vespa has been running in production since 2026-04-15.")
    sims = sorted(result.role_similarities.values(), reverse=True)
    expected_margin = sims[0] - sims[1]
    assert result.confidence == pytest.approx(expected_margin, abs=1e-9)


def test_custom_prototypes_override(fake_embed_fn):
    """Caller can pass a custom prototype set (e.g. for taxonomy validation)."""
    custom = {
        role: [
            {"id": f"custom_{role}_{i}", "text": f"custom prototype {role} {i}", "topic": "test"}
            for i in range(7)
        ]
        for role in ROLES
    }
    clf = RoleClassifier(
        embed_fn=fake_embed_fn,
        model_name="fake-model",
        prototypes=custom,
    )
    counts = clf.prototype_count
    for role in ROLES:
        assert counts[role] == 7


def test_empty_prototypes_raises_at_construction(fake_embed_fn):
    """Empty prototype dict is not a valid taxonomy."""
    with pytest.raises(ValueError, match="at least one role"):
        RoleClassifier(
            embed_fn=fake_embed_fn,
            model_name="fake-model",
            prototypes={},
        )


def test_empty_role_pool_raises_at_construction(fake_embed_fn):
    """A role with no prototypes is not embeddable; reject it loudly upfront."""
    with pytest.raises(ValueError, match="empty"):
        RoleClassifier(
            embed_fn=fake_embed_fn,
            model_name="fake-model",
            prototypes={"only_role": []},
        )


def test_custom_taxonomy_with_arbitrary_role_names(fake_embed_fn):
    """Custom prototypes can use role names outside the default ROLES set."""
    custom = {
        "alpha": [{"id": f"a{i}", "text": f"alpha example {i}", "topic": "t"} for i in range(5)],
        "beta": [{"id": f"b{i}", "text": f"beta example {i}", "topic": "t"} for i in range(5)],
    }
    clf = RoleClassifier(
        embed_fn=fake_embed_fn,
        model_name="fake-model",
        prototypes=custom,
    )
    assert clf.roles == ("alpha", "beta")
    result = clf.classify("any text")
    assert result.role in {"alpha", "beta"}
    assert set(result.role_similarities.keys()) == {"alpha", "beta"}


def test_single_role_taxonomy_confidence_is_inf(fake_embed_fn):
    """With one role, no top-2 exists; confidence is +inf by convention."""
    custom = {
        "only": [{"id": f"o{i}", "text": f"only example {i}", "topic": "t"} for i in range(5)],
    }
    clf = RoleClassifier(
        embed_fn=fake_embed_fn,
        model_name="fake-model",
        prototypes=custom,
    )
    result = clf.classify("anything")
    assert result.role == "only"
    assert result.confidence == float("inf")


def test_roles_property_matches_default_taxonomy(fake_embed_fn):
    """The ``roles`` property exposes the taxonomy order."""
    clf = RoleClassifier(embed_fn=fake_embed_fn, model_name="fake-model")
    assert clf.roles == ROLES


def test_roles_tuple_matches_default_prototypes():
    """Sanity: ROLES module constant matches the bundled prototype JSON keys."""
    protos = _load_prototypes_default()
    assert set(ROLES) == set(protos.keys())
