"""``D1WriteDecider`` -- should_remember answered by a local decision model.

Liquid AI's open d1 (~2B parameters) answers typed questions about a state with calibrated
probabilities in one forward pass, no generated tokens. Asked whether an event records a decision
or durable knowledge, it is a zero-shot write filter: no training data, no merken-specific
fine-tune, and a real probability for the audit row instead of a fixed confidence.

Not a default. Per CONSTITUTION §9 it ships as an opt-in decider with its benchmark in
``experiments/d1_write_filter/``. Needs ``pip install 'merken[d1]'`` (PyTorch, torchvision,
transformers); the model downloads from Hugging Face on first use. ``adapter=`` loads a LoRA
trained on merken's own labels by ``experiments/d1_write_filter/train.py`` (also needs ``peft``).

    from merken import Memory
    from merken.policies.d1_write import D1WriteDecider

    mem = Memory(project="agent", write_decider=D1WriteDecider())
"""

from __future__ import annotations

from typing import Any

from merken.policies.types import Decision, Event, WriteContext

# Matches merken's labels: DECISION = what was chosen / changed / learned (and why), or a
# reference doc; routine activity is noise even when it carries a number. Tuned on
# knowledge_update_20topics only.
INSTRUCTIONS = (
    "Does this record a decision or durable project knowledge: what was chosen, changed, adopted, "
    "dropped or learned, ideally with the reason, or a reference document (design, architecture, "
    "playbook, agreed plan) someone will consult again? Answer no for routine activity: status "
    "updates, progress reports, metric bulletins, meeting or review notices, tool and command "
    "output, acknowledgements, and passing remarks."
)


class D1WriteDecider:
    """Write when P(worth remembering) >= ``threshold``.

    The decision model is loaded on the first ``decide`` call and kept for the life of the
    decider. ``max_chars`` truncates very long events before they reach the model (the head of a
    document carries its type and topic); it does not skip them.
    """

    name = "D1WriteDecider"

    def __init__(
        self,
        model: str = "LiquidAI/d1-2B",
        threshold: float = 0.7,
        device: str = "auto",
        max_chars: int = 12_000,
        instructions: str = INSTRUCTIONS,
        adapter: str | None = None,
    ) -> None:
        self.model_name = model
        self.adapter = adapter
        self.threshold = threshold
        self.device = device
        self.max_chars = max_chars
        self.instructions = instructions
        self._model: Any = None
        self._torch: Any = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError("D1WriteDecider needs: pip install 'merken[d1]'") from exc
        dev = self.device
        if dev == "auto":
            dev = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        dtype = torch.float32 if dev == "cpu" else torch.bfloat16
        self._torch = torch
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, dtype=dtype)
        if self.adapter:
            self._attach_adapter(model)
        self._model = model.to(dev).eval()

    def _attach_adapter(self, model: Any) -> None:
        """Inject a LoRA from ``experiments/d1_write_filter/train.py`` (needs ``peft``)."""
        import json
        from pathlib import Path

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        root = Path(self.adapter)
        cfg = LoraConfig(**json.loads((root / "lora.json").read_text()))
        inject_adapter_in_model(cfg, model)
        set_peft_model_state_dict(model, load_file(str(root / "adapter.safetensors")))

    def probabilities(self, texts: list[str]) -> list[float]:
        """P(worth remembering) for each text, batched in one call."""
        if self._model is None:
            self._load()
        q = {"keep": {"type": "noul", "instructions": self.instructions}}
        requests = [(t[: self.max_chars], q) for t in texts]
        with self._torch.inference_mode():
            out = self._model.system_one_batch(requests)
        return [float(o["answers"]["keep"]["noul"]) for o in out]

    def decide(self, event: Event, ctx: WriteContext) -> Decision:  # noqa: ARG002
        text = event.text.strip()
        if not text:
            return Decision(write=False, reason="empty", confidence=1.0, policy=self.name)
        p = self.probabilities([text])[0]
        write = p >= self.threshold
        return Decision(
            write=write,
            reason=f"d1_keep:{p:.3f}",
            confidence=p if write else 1.0 - p,
            policy=self.name,
        )
