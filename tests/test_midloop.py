"""Tests for the midloop primitive (Phase 2 sketch).

Coverage targets:
- StepObservation / MidloopDecision dataclasses behave as expected.
- NoopMidloopDecider always returns NONE / ON_TRACK.
- HeuristicMidloopDecider:
  - Skips STRUCTURED tasks unconditionally.
  - Returns insufficient_history when trajectory is too small.
  - Detects LOOPING (high jaccard) -> WHISPER.
  - Detects DRIFTING (low task_relevance) -> WHISPER.
  - Detects STUCK (length / latency / tool_diversity triggers) -> WHISPER.
  - Returns ON_TRACK with low confidence when no signal fires.
- ShadowMidloopDecider:
  - Primary's decision is authoritative.
  - Reason string carries shadow_agree / shadow_disagree tag.
  - Shadow exception is caught with shadow_error: tag.
- is_persistable_step filter rules.
- format_midloop_audit_row body shape.
"""

from __future__ import annotations

import json
import time

from merken.policies.midloop import (
    CognitiveState,
    HeuristicMidloopDecider,
    HeuristicThresholds,
    InterventionAction,
    MidloopContext,
    MidloopDecision,
    NoopMidloopDecider,
    ShadowMidloopDecider,
    StepObservation,
    TaskCategory,
    ToolCall,
    format_midloop_audit_row,
    is_persistable_step,
)

# ----------------------------------------------------------------- helpers

def _obs(
    *,
    step_index: int = 0,
    output: str = "the model said something",
    task_id: str = "t1",
    category: TaskCategory = TaskCategory.EXPLORATORY,
    latency_ms: int = 1000,
    tool_calls: list[ToolCall] | None = None,
) -> StepObservation:
    return StepObservation(
        step_id=f"s{step_index}",
        task_id=task_id,
        task_category=category,
        timestamp=time.time(),
        output_text=output,
        step_index=step_index,
        latency_ms=latency_ms,
        tool_calls=tool_calls or [],
    )


def _ctx(
    *,
    category: TaskCategory = TaskCategory.EXPLORATORY,
    task_description: str = "",
) -> MidloopContext:
    return MidloopContext(
        project="test_midloop",
        task_id="t1",
        task_category=category,
        task_description=task_description,
    )


# ----------------------------------------------------------------- noop

class TestNoopMidloopDecider:
    def test_always_returns_none(self) -> None:
        d = NoopMidloopDecider()
        result = d.decide(_obs(), _ctx(), [])
        assert result.intervene is False
        assert result.action == InterventionAction.NONE
        assert result.state == CognitiveState.ON_TRACK
        assert result.policy == "NoopMidloopDecider"

    def test_ignores_trajectory_length(self) -> None:
        d = NoopMidloopDecider()
        traj = [_obs(step_index=i, output=f"step {i}") for i in range(20)]
        result = d.decide(_obs(step_index=20), _ctx(), traj)
        assert result.intervene is False


# ----------------------------------------------------------------- heuristic

class TestHeuristicStructuredSkip:
    def test_structured_task_never_intervenes(self) -> None:
        d = HeuristicMidloopDecider()
        ctx = _ctx(category=TaskCategory.STRUCTURED)
        traj = [_obs(step_index=i, output="repeat repeat repeat") for i in range(5)]
        # Even with looping pattern, STRUCTURED skips.
        result = d.decide(
            _obs(step_index=5, output="repeat repeat repeat", category=TaskCategory.STRUCTURED),
            ctx, traj,
        )
        assert result.intervene is False
        assert result.reason == "structured_task_skip"


class TestHeuristicHistoryGate:
    def test_returns_insufficient_history_when_too_few_observations(self) -> None:
        d = HeuristicMidloopDecider()
        result = d.decide(_obs(), _ctx(), [])
        assert result.intervene is False
        assert "insufficient_history" in result.reason

    def test_below_min_steps_does_not_fire_even_on_looping(self) -> None:
        d = HeuristicMidloopDecider(HeuristicThresholds(min_steps_to_decide=5))
        traj = [_obs(step_index=i, output="same same same") for i in range(3)]
        result = d.decide(
            _obs(step_index=3, output="same same same"), _ctx(), traj,
        )
        assert result.intervene is False
        assert "insufficient_history" in result.reason


class TestHeuristicLooping:
    def test_high_jaccard_triggers_looping_whisper(self) -> None:
        d = HeuristicMidloopDecider()
        # 4 nearly identical outputs in a row -> LOOPING.
        traj = [
            _obs(step_index=i, output="let me check the cache configuration")
            for i in range(4)
        ]
        result = d.decide(
            _obs(step_index=4, output="let me check the cache configuration again"),
            _ctx(), traj,
        )
        assert result.intervene is True
        assert result.state == CognitiveState.LOOPING
        assert result.action == InterventionAction.WHISPER
        assert result.signals["jaccard_last_n"] >= 0.5
        assert "looping" in result.reason


class TestHeuristicDrifting:
    def test_low_task_relevance_triggers_drifting(self) -> None:
        # Outputs MUST have low pairwise overlap so LOOPING does not
        # fire before DRIFTING. Each step uses a disjoint vocabulary.
        d = HeuristicMidloopDecider(
            HeuristicThresholds(relevance_drifting=0.5)
        )
        traj = [
            _obs(step_index=0, output="alpha bravo charlie delta echo"),
            _obs(step_index=1, output="foxtrot golf hotel india juliet"),
            _obs(step_index=2, output="kilo lima mike november oscar"),
            _obs(step_index=3, output="papa quebec romeo sierra tango"),
        ]
        ctx = _ctx(task_description="rust async runtime tokio scheduler")
        result = d.decide(
            _obs(step_index=4, output="uniform victor whiskey xray yankee"),
            ctx, traj,
        )
        assert result.intervene is True
        assert result.state == CognitiveState.DRIFTING
        assert result.action == InterventionAction.WHISPER

    def test_empty_task_description_does_not_trigger_drifting(self) -> None:
        # task_relevance defaults to 1.0 when description is empty,
        # so the drifting check never fires regardless of output.
        d = HeuristicMidloopDecider()
        traj = [
            _obs(step_index=0, output="alpha bravo charlie"),
            _obs(step_index=1, output="delta echo foxtrot"),
            _obs(step_index=2, output="golf hotel india"),
            _obs(step_index=3, output="juliet kilo lima"),
        ]
        result = d.decide(
            _obs(step_index=4, output="mike november oscar"),
            _ctx(),  # no task_description
            traj,
        )
        # Either ON_TRACK or some other state, but NOT drifting.
        assert result.state != CognitiveState.DRIFTING


class TestHeuristicStuck:
    def test_shrinking_output_triggers_stuck(self) -> None:
        # Output length monotonically shrinks. Outputs must have
        # NEAR-ZERO pairwise jaccard so LOOPING does not fire first.
        d = HeuristicMidloopDecider(
            HeuristicThresholds(length_trend_stuck=-0.05, jaccard_looping=0.99)
        )
        traj = [
            _obs(step_index=0, output=" ".join(f"alpha{i}" for i in range(200))),
            _obs(step_index=1, output=" ".join(f"bravo{i}" for i in range(160))),
            _obs(step_index=2, output=" ".join(f"charlie{i}" for i in range(120))),
            _obs(step_index=3, output=" ".join(f"delta{i}" for i in range(80))),
        ]
        result = d.decide(
            _obs(step_index=4, output=" ".join(f"echo{i}" for i in range(40))),
            _ctx(), traj,
        )
        assert result.intervene is True
        assert result.state == CognitiveState.STUCK
        assert "length_trend" in result.reason

    def test_growing_latency_triggers_stuck(self) -> None:
        d = HeuristicMidloopDecider(
            HeuristicThresholds(latency_trend_stuck=0.05, jaccard_looping=0.99)
        )
        # Each step has unique vocabulary; latency monotonically grows.
        traj = [
            _obs(step_index=i, output=f"alpha{i} bravo{i} charlie{i}",
                 latency_ms=100 * (i + 1))
            for i in range(4)
        ]
        result = d.decide(
            _obs(step_index=4, output="delta zeta theta", latency_ms=500),
            _ctx(), traj,
        )
        assert result.intervene is True
        assert result.state == CognitiveState.STUCK
        assert "latency_trend" in result.reason

    def test_repeating_tool_calls_triggers_stuck(self) -> None:
        d = HeuristicMidloopDecider(
            HeuristicThresholds(tool_diversity_stuck=0.5, jaccard_looping=0.99)
        )
        traj = [
            _obs(
                step_index=i, output=f"alpha{i} bravo{i} charlie{i} delta{i}",
                tool_calls=[ToolCall(name="grep")],
            )
            for i in range(4)
        ]
        result = d.decide(
            _obs(
                step_index=4, output="echo foxtrot golf hotel",
                tool_calls=[ToolCall(name="grep")],
            ),
            _ctx(), traj,
        )
        assert result.intervene is True
        assert result.state == CognitiveState.STUCK
        assert "tool_diversity" in result.reason


class TestHeuristicOnTrack:
    def test_diverse_outputs_no_intervention(self) -> None:
        d = HeuristicMidloopDecider()
        traj = [
            _obs(
                step_index=0,
                output="reading the auth module to understand session handling",
            ),
            _obs(
                step_index=1,
                output="found the login flow uses jwt cookies for stateful sessions",
            ),
            _obs(
                step_index=2,
                output="now examining the refresh token rotation behavior",
            ),
            _obs(
                step_index=3,
                output="the rotation interval is configured at fifteen minutes by default",
            ),
        ]
        result = d.decide(
            _obs(
                step_index=4,
                output="checking how the middleware enforces the rotation policy",
            ),
            _ctx(),
            traj,
        )
        assert result.intervene is False
        assert result.state == CognitiveState.ON_TRACK
        assert result.action == InterventionAction.NONE


# ----------------------------------------------------------------- shadow

class TestShadowMidloopDecider:
    def test_primary_decision_wins(self) -> None:
        # primary=Noop, shadow=Heuristic that WOULD intervene on
        # looping. Primary's NONE must be the final action.
        primary = NoopMidloopDecider()
        shadow = HeuristicMidloopDecider()
        chained = ShadowMidloopDecider(primary=primary, shadow=shadow)
        traj = [_obs(step_index=i, output="repeat repeat repeat") for i in range(4)]
        result = chained.decide(
            _obs(step_index=4, output="repeat repeat repeat"),
            _ctx(), traj,
        )
        assert result.intervene is False
        assert result.action == InterventionAction.NONE
        assert "shadow_disagree" in result.reason
        # Shadow's policy + would-be action embedded in reason.
        assert "HeuristicMidloopDecider" in result.reason
        assert "whisper" in result.reason
        # Primary's signals propagate (Noop has none).
        assert result.signals == {}

    def test_agreement_tagged(self) -> None:
        # Both deciders agree (both are Noop -> never intervene).
        chained = ShadowMidloopDecider(
            primary=NoopMidloopDecider(),
            shadow=NoopMidloopDecider(),
        )
        result = chained.decide(_obs(), _ctx(), [])
        assert result.intervene is False
        assert "shadow_agree" in result.reason

    def test_shadow_exception_caught(self) -> None:
        class BoomDecider:
            name = "BoomDecider"

            def decide(self, observation, ctx, trajectory):
                raise RuntimeError("kaboom")

        chained = ShadowMidloopDecider(
            primary=NoopMidloopDecider(),
            shadow=BoomDecider(),
        )
        result = chained.decide(_obs(), _ctx(), [])
        # Primary still wins, exception annotated.
        assert result.intervene is False
        assert "shadow_error:RuntimeError" in result.reason


# ----------------------------------------------------------------- persistence

class TestIsPersistableStep:
    def _decision(
        self,
        state: CognitiveState = CognitiveState.ON_TRACK,
        confidence: float = 1.0,
        intervene: bool = False,
    ) -> MidloopDecision:
        return MidloopDecision(
            intervene=intervene,
            action=InterventionAction.NONE if not intervene else InterventionAction.WHISPER,
            confidence=confidence,
            reason="test",
            state=state,
            policy="test",
        )

    def test_on_track_with_no_change_drops(self) -> None:
        prev = self._decision()
        cur = self._decision()
        assert is_persistable_step(cur, prev, is_last_step=False) is False

    def test_first_step_drops_when_on_track(self) -> None:
        cur = self._decision()
        assert is_persistable_step(cur, None, is_last_step=False) is False

    def test_intervention_state_always_persists(self) -> None:
        prev = self._decision()
        cur = self._decision(state=CognitiveState.LOOPING, intervene=True)
        assert is_persistable_step(cur, prev, is_last_step=False) is True

    def test_state_change_persists(self) -> None:
        prev = self._decision(state=CognitiveState.LOOPING)
        cur = self._decision(state=CognitiveState.ON_TRACK)
        assert is_persistable_step(cur, prev, is_last_step=False) is True

    def test_confidence_jump_persists(self) -> None:
        prev = self._decision(confidence=0.5)
        cur = self._decision(confidence=0.9)
        assert is_persistable_step(cur, prev, is_last_step=False) is True

    def test_small_confidence_change_drops(self) -> None:
        prev = self._decision(confidence=0.5)
        cur = self._decision(confidence=0.55)
        assert is_persistable_step(cur, prev, is_last_step=False) is False

    def test_last_step_always_persists(self) -> None:
        prev = self._decision()
        cur = self._decision()
        assert is_persistable_step(cur, prev, is_last_step=True) is True


# ----------------------------------------------------------------- audit row

class TestFormatMidloopAuditRow:
    def test_title_includes_task_and_step(self) -> None:
        obs = _obs(step_index=42, task_id="task_xyz")
        decision = NoopMidloopDecider().decide(obs, _ctx(), [])
        title, body = format_midloop_audit_row(obs, decision)
        assert "audit:should_intervene" in title
        assert "task_xyz" in title
        assert "step0042" in title

    def test_body_carries_required_fields(self) -> None:
        obs = _obs(
            step_index=3,
            output="some output that is long enough to preview",
        )
        traj = [_obs(step_index=i, output="repeat repeat") for i in range(4)]
        decision = HeuristicMidloopDecider().decide(
            _obs(step_index=4, output="repeat repeat"),
            _ctx(),
            traj,
        )
        _, body = format_midloop_audit_row(obs, decision)
        # FTS-friendly fields embedded in body.
        for key in (
            "task_id:", "step_id:", "step_index:",
            "task_category:", "intervene:", "action:",
            "state:", "confidence:", "reason:", "policy:",
            "signals:", "output_preview:",
        ):
            assert key in body, f"missing {key} in audit body"

    def test_signals_serialized_as_json(self) -> None:
        obs = _obs(step_index=4, output="output")
        traj = [_obs(step_index=i, output="x" * 100) for i in range(4)]
        decision = HeuristicMidloopDecider().decide(obs, _ctx(), traj)
        _, body = format_midloop_audit_row(obs, decision)
        # Find the signals: line and parse it.
        signals_line = next(
            line for line in body.split("\n") if line.startswith("signals:")
        )
        payload = json.loads(signals_line[len("signals:"):].strip())
        assert isinstance(payload, dict)


# ----------------------------------------------------------------- protocol

def test_decider_protocol_is_satisfied_by_all_three() -> None:
    """Structural typing -- if these compile, all three deciders
    satisfy the MidloopDecider Protocol."""
    from merken.policies.midloop import MidloopDecider

    deciders: list[MidloopDecider] = [
        NoopMidloopDecider(),
        HeuristicMidloopDecider(),
        ShadowMidloopDecider(NoopMidloopDecider(), NoopMidloopDecider()),
    ]
    for d in deciders:
        assert hasattr(d, "name") and isinstance(d.name, str)
        assert callable(d.decide)
