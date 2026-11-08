"""Frozen-encoder + MLP WriteDecider.

Architectural sibling to NanoGPTWriteDecider. Instead of an
autoregressive model that has to learn English + the classification
feature jointly, this decider uses a frozen pre-trained encoder
(default: vstash's production bge-small-en-v1.5) plus a small MLP
head trained only on the classification objective.

Trained by `experiments/nanogpt/train_v8c_frozen_encoder.py`. The
head ships as a torch state_dict (`head.pt`) plus a meta.json with
the encoder id and the MLP shape.

API matches NanoGPTWriteDecider so merken.Memory can swap via
``ChainedWriteDecider`` or direct substitution. Decision.reason
contains ``P_dec`` + ``encoder=<model>`` so audit rows still have
a diagnostic string.
"""

from __future__ import annotations

import json
from pathlib import Path

from merken.policies import Decision, Event, WriteContext


class FrozenEncoderWriteDecider:
    """WriteDecider using vstash embedder + MLP classification head.

    ``confidence_threshold`` is applied to softmax(logits)[1] = P(DECISION).
    Same semantic as NanoGPTWriteDecider: write iff P_dec > P_noi AND
    P_dec >= threshold.
    """

    name = "frozen-encoder-classifier"

    def __init__(
        self,
        head_path: str | Path,
        meta_path: str | Path,
        *,
        confidence_threshold: float = 0.5,
        encoder_override: str | None = None,
    ) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self._nn = nn
        self._threshold = confidence_threshold

        meta = json.loads(Path(meta_path).read_text())
        self._encoder_id = encoder_override or meta["encoder"]
        self._dim = meta["dim"]
        self._hidden = meta["hidden_dim"]

        class Head(nn.Module):
            def __init__(self, dim: int, hidden: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, 2),
                )

            def forward(self, x):
                return self.net(x)

        self._model = Head(self._dim, self._hidden)
        state = torch.load(str(head_path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _embed(self, text: str):
        """Call vstash's embedder. Cached per call (no batching at
        inference -- decide() is per-event). For bulk ingestion the
        caller should batch upstream.
        """
        from vstash.embed import embed_texts
        vecs = embed_texts([text], model_name=self._encoder_id, backend="auto")
        return vecs[0]

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        import torch

        vec = self._embed(event.text)
        x = self._torch.tensor(vec, dtype=self._torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[0]
        p_noise = float(probs[0])
        p_decision = float(probs[1])

        is_signal = p_decision > p_noise and p_decision >= self._threshold
        reason = (
            f"P(D)={p_decision:.3f} P(N)={p_noise:.3f} "
            f"encoder={self._encoder_id}"
        )
        return Decision(
            write=is_signal,
            reason=reason,
            confidence=max(p_decision, p_noise),
            policy=self.name,
        )
