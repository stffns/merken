"""merken — agent-loop layer for persistent memory, built on top of vstash.

See ``CONSTITUTION.md`` for what merken is, what it isn't, and the principles
that should outlive any specific implementation.
"""

# Must import _shadow FIRST: when MERKEN_SHADOW is set, this eagerly
# loads torch before vstash/fastembed gets imported transitively via
# .memory / .consolidation -- see notes/nanogpt-training-log.md
# Mistake #10 for why the load order matters.
from merken import _shadow as _shadow  # noqa: F401
from merken.consolidation import ConsolidationResult, Fact
from merken.enrichment import Enricher, LLMEnricher, jev_decision_gate
from merken.memory import ForgetResult, Memory, RememberResult
from merken.policies import (
    AlwaysWrite,
    ChainedWriteDecider,
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    ContentTypePriorDecider,
    Decision,
    Event,
    ForgetConsolidated,
    ForgetConsolidatedOrSuperseded,
    ForgetContext,
    ForgetDecider,
    ForgetDecision,
    ForgetSuperseded,
    HeuristicWriteDecider,
    LayeredRecaller,
    LayerRequest,
    NeverConsolidate,
    NeverForget,
    PeriodicConsolidator,
    RecallContext,
    RecallDecider,
    RecallPlan,
    SemanticOnlyRecaller,
    ShadowWriteDecider,
    WriteContext,
    WriteDecider,
)

__version__ = "0.1.0"
__all__ = [
    "AlwaysWrite",
    "ChainedWriteDecider",
    "ContentTypePriorDecider",
    "ConsolidateContext",
    "ConsolidateDecider",
    "ConsolidationDecision",
    "ConsolidationResult",
    "Decision",
    "Enricher",
    "Event",
    "Fact",
    "ForgetConsolidated",
    "ForgetConsolidatedOrSuperseded",
    "ForgetContext",
    "ForgetDecider",
    "ForgetDecision",
    "ForgetResult",
    "ForgetSuperseded",
    "HeuristicWriteDecider",
    "LayerRequest",
    "LLMEnricher",
    "LayeredRecaller",
    "Memory",
    "NeverConsolidate",
    "NeverForget",
    "PeriodicConsolidator",
    "RecallContext",
    "RecallDecider",
    "RecallPlan",
    "RememberResult",
    "SemanticOnlyRecaller",
    "ShadowWriteDecider",
    "WriteContext",
    "WriteDecider",
    "__version__",
    "jev_decision_gate",
]
