"""Phase tests — ``should_recall`` decider + layered recall in Memory.

Split between pure unit tests (no vstash) and integration tests
that exercise the full ``Memory.recall`` path with a real vstash
instance.
"""

from __future__ import annotations

from pathlib import Path

from merken import (
    LayeredRecaller,
    LayerRequest,
    Memory,
    RecallContext,
    RecallPlan,
    SemanticOnlyRecaller,
)

# --------------------------------------------------------------- pure policy


def _ctx(top_k: int = 5) -> RecallContext:
    return RecallContext(project="unit", top_k=top_k)


def test_layered_default_plans_semantic_then_episodic() -> None:
    decider = LayeredRecaller()
    plan = decider.decide("anything", _ctx())

    assert isinstance(plan, RecallPlan)
    assert plan.policy == "LayeredRecaller"
    assert [r.layer for r in plan.layers] == ["semantic", "episodic"]
    # Default budgets
    assert plan.layers[0].top_k == 5
    assert plan.layers[1].top_k == 3


def test_layered_respects_configured_budgets() -> None:
    decider = LayeredRecaller(top_k_semantic=2, top_k_episodic=7)
    plan = decider.decide("anything", _ctx())

    assert plan.layers[0] == LayerRequest(layer="semantic", top_k=2)
    assert plan.layers[1] == LayerRequest(layer="episodic", top_k=7)
    assert "sem=2" in plan.reason
    assert "epi=7" in plan.reason


def test_semantic_only_baseline() -> None:
    decider = SemanticOnlyRecaller()
    plan = decider.decide("query", _ctx(top_k=5))

    assert len(plan.layers) == 1
    assert plan.layers[0].layer == "semantic"
    assert plan.layers[0].top_k == 5
    assert plan.reason == "semantic_only"


# ---------------------------------------------------- integration with Memory


def test_memory_recall_routes_via_decider_by_default(tmp_path: Path) -> None:
    """Without an explicit layer, recall uses the LayeredRecaller path
    and finds hits across both semantic and episodic."""
    with Memory(project="recall_routed", db=tmp_path / "e.db") as mem:
        mem.remember("The user prefers the color teal.", layer="semantic")
        mem.remember("A raw observation about teal monitors.", layer="episodic")

        hits = mem.recall("teal preference")

    assert hits
    # Both were about teal; the semantic one should at least be present
    texts = " ".join((h.text or "") for h in hits)
    assert "teal" in texts.lower()


def test_memory_recall_explicit_layer_bypasses_decider(tmp_path: Path) -> None:
    """When a layer is explicitly passed, recall goes straight to vstash
    without writing a should_recall audit row."""
    with Memory(project="recall_bypass", db=tmp_path / "e.db") as mem:
        mem.remember("A fact about quartz crystals.", layer="semantic")
        mem.remember("An event about granite rocks.", layer="episodic")

        # Explicit layer — decider should not fire
        hits = mem.recall("quartz", top_k=5, layer="semantic")

    assert hits
    assert any("quartz" in (h.text or "").lower() for h in hits)


def test_memory_recall_dedupes_by_path(tmp_path: Path) -> None:
    """When the same doc shows up in semantic and episodic (shouldn't
    normally happen, but we're defensive), the deduped result has it
    once."""
    with Memory(project="recall_dedupe", db=tmp_path / "e.db") as mem:
        mem.remember("A cross-layer test document about fossils.", layer="semantic")

        hits = mem.recall("fossils", top_k=10)

    seen_paths = [getattr(h, "path", None) for h in hits]
    assert len(seen_paths) == len(set(seen_paths))


def test_memory_recall_writes_audit_row(tmp_path: Path) -> None:
    """A decider-routed recall writes one should_recall audit row."""
    with Memory(project="recall_audit", db=tmp_path / "e.db") as mem:
        mem.remember("An event about volcanoes erupting.")
        mem.recall("volcanoes")

        rows = mem.audit(query="should_recall", top_k=10)

    assert rows
    bodies = " ".join((r.text or "") for r in rows)
    assert "should_recall" in bodies
    assert "layered_sem" in bodies or "sem=" in bodies


def test_memory_recall_explicit_layer_does_not_write_recall_audit(
    tmp_path: Path,
) -> None:
    """Explicit-layer recall is the escape hatch — no audit row for it
    (avoids polluting audit with benchmark runs)."""
    with Memory(project="recall_no_audit", db=tmp_path / "e.db") as mem:
        mem.remember("A fact about tectonics.", layer="semantic")
        mem.recall("tectonics", layer="semantic")

        rows = mem.audit(query="should_recall", top_k=10)

    # Explicit-layer recalls don't emit should_recall audit rows
    assert not any("should_recall" in (r.text or "") for r in rows)


def test_memory_recall_with_custom_decider(tmp_path: Path) -> None:
    """A user-supplied decider replaces the default."""
    with Memory(
        project="recall_custom",
        db=tmp_path / "e.db",
        recall_decider=SemanticOnlyRecaller(),
    ) as mem:
        mem.remember("A semantic fact about auroras.", layer="semantic")
        mem.remember("An episodic note about auroras.", layer="episodic")

        hits = mem.recall("auroras")
        rows = mem.audit(query="should_recall", top_k=10)

    # The SemanticOnlyRecaller only queries semantic
    assert any("auroras" in (h.text or "").lower() for h in hits)
    assert any("semantic_only" in (r.text or "") for r in rows)
