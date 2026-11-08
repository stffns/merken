"""nanoGPT-based event classifier for merken write decisions.

Loads a trained nanoGPT checkpoint and classifies events as NOISE or
DECISION (signal). Used as a WriteDecider to filter noise at ingestion
time, keeping the memory store clean for retrieval.

Two tokenization modes are supported, auto-detected from meta.pkl:

  char-level (v1/v2/v3b): 81-token vocab, one id per character. We check
    whether the model predicts 'D' (DECISION) or 'N' (NOISE) as the first
    character after <|label|>.

  BPE (v4): 512-token custom vocab with 'DECISION' and 'NOISE' as single
    tokens. Meta carries a `tokenizer_path` pointing at a HuggingFace
    tokenizer.json. We check logits for the DECISION vs NOISE token ids.

v3b and v4 were trained with verb-marker prefixes ([VERB:replaced],
[CTX:routine], ...) so the decider prepends them at inference when
`use_verb_markers` is on. Default: on for BPE, off for char-level.

Requires: torch. BPE mode additionally requires `tokenizers`.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from merken.policies import Decision, Event, WriteContext

if TYPE_CHECKING:
    from merken.classifiers.calibration import CalibrationHead

_DECISION_VERBS = {
    "replaced", "migrated", "switched", "adopted", "deployed",
    "implemented", "chose", "built", "set", "added", "upgraded",
    "moved", "extended", "reverted", "configured", "enabled",
}
_NOISE_VERBS = {
    "discussed", "considering", "evaluated", "explored", "debated",
    "reviewed", "updated", "investigating", "investigated", "scheduled",
    "linked", "filed", "resolved", "escalated", "created", "closed",
    "rotated", "cleaned", "bumped", "requested", "noted",
}
_ROUTINE_HEADS = {
    "sprint", "ticket", "retro", "demo", "oncall", "slack", "meeting",
    "budget", "workshop", "code", "incident", "performance", "security",
    "dependency", "planning",
}


def extract_verb_marker(text: str) -> str:
    """Mirror of nanoGPT/data/merken_bpe/prepare.py::extract_verb_marker.

    Training-time and inference-time markers must match exactly -- the
    token id for [VERB:replaced] is only meaningful if the training data
    produced it the same way. Keep this function in lockstep with the
    prepare.py in the nanoGPT repo.
    """
    words = text.lower().split()
    for w in words[:6]:
        clean = re.sub(r"[^a-z]", "", w)
        if clean in _DECISION_VERBS or clean in _NOISE_VERBS:
            return f"[VERB:{clean}]"
    if words and re.sub(r"[^a-z]", "", words[0].lower()) in _ROUTINE_HEADS:
        return "[CTX:routine]"
    return "[CTX:unknown]"


@dataclass
class NanoGPTConfig:
    """Mirrors nanoGPT's GPTConfig for inference only."""
    block_size: int = 256
    vocab_size: int = 81
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


def _load_model(checkpoint_path: Path):
    """Load a nanoGPT checkpoint for inference."""
    import math

    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    # Minimal model definition (inference only, no training code)
    class LayerNorm(nn.Module):
        def __init__(self, ndim, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        def forward(self, input):
            return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.dropout = config.dropout
            self.flash = hasattr(F, "scaled_dot_product_attention")

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            if self.flash:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(
                    torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0,
                    float("-inf"),
                )
                att = F.softmax(att, dim=-1)
                y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.c_proj(y))

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight

        def forward(self, idx):
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
            x = self.transformer.drop(
                self.transformer.wte(idx) + self.transformer.wpe(pos)
            )
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            return self.lm_head(x[:, [-1], :])

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = NanoGPTConfig(**{
        k: v for k, v in ckpt["model_args"].items()
        if k in NanoGPTConfig.__dataclass_fields__
    })
    model = GPT(config)
    state_dict = ckpt["model"]
    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class NanoGPTWriteDecider:
    """WriteDecider that uses a trained nanoGPT to filter noise.

    Auto-detects tokenization from meta.pkl. In char-level mode, it reads
    the logit for the first character after <|label|> ('D' or 'N'). In
    BPE mode, it reads the logit for the 'DECISION' vs 'NOISE' tokens.
    """

    name = "nanogpt-classifier"

    def __init__(
        self,
        checkpoint_path: str | Path,
        meta_path: str | Path,
        *,
        confidence_threshold: float = 0.6,
        use_verb_markers: bool | None = None,
        calibrator: CalibrationHead | None = None,
    ) -> None:
        """Create a NanoGPT write decider.

        Args:
            calibrator: optional post-hoc calibrator. When provided,
                the `reason` string gains a ``P_cal`` field and the
                `confidence` field reports the calibrated probability.
                The pass/fail decision (``write``) still uses the RAW
                P(D) against ``confidence_threshold`` so existing
                graduation metrics remain comparable. Display-only.
        """
        import torch
        self._torch = torch

        self._model, self._config = _load_model(Path(checkpoint_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._stoi: dict[str, int] = meta["stoi"]
        self._itos: dict = meta["itos"]
        self._threshold = confidence_threshold

        # BPE mode iff meta points at a HF tokenizer.json.
        bpe_path = meta.get("tokenizer_path")
        if bpe_path:
            from tokenizers import Tokenizer
            self._bpe = Tokenizer.from_file(str(bpe_path))
            self._d_id = self._stoi.get("DECISION")
            self._n_id = self._stoi.get("NOISE")
            if self._d_id is None or self._n_id is None:
                raise ValueError(
                    "BPE vocab is missing DECISION/NOISE tokens; cannot "
                    "classify. Retrain the tokenizer."
                )
        else:
            self._bpe = None
            self._d_id = self._stoi.get("D")
            self._n_id = self._stoi.get("N")

        self._use_markers = (
            (self._bpe is not None) if use_verb_markers is None else use_verb_markers
        )
        self._calibrator = calibrator

    def _encode(self, prompt: str) -> list[int]:
        if self._bpe is not None:
            return list(self._bpe.encode(prompt).ids)
        return [self._stoi.get(c, 0) for c in prompt]

    def decide(self, event: Event, ctx: WriteContext) -> Decision:
        if self._use_markers:
            prompt = f"{extract_verb_marker(event.text)}<|event|>{event.text}<|label|>"
        else:
            prompt = f"<|event|>{event.text}<|label|>"

        ids = self._encode(prompt)
        if len(ids) > self._config.block_size:
            ids = ids[-self._config.block_size:]

        x = self._torch.tensor([ids], dtype=self._torch.long)

        with self._torch.no_grad():
            logits = self._model(x)

        probs = self._torch.nn.functional.softmax(logits[0, -1, :], dim=-1)

        p_decision = float(probs[self._d_id]) if self._d_id is not None else 0.0
        p_noise = float(probs[self._n_id]) if self._n_id is not None else 0.0

        is_signal = p_decision > p_noise and p_decision >= self._threshold

        reason = f"P(D)={p_decision:.3f} P(N)={p_noise:.3f}"
        reported_confidence = max(p_decision, p_noise)
        if self._calibrator is not None:
            p_cal = self._calibrator.calibrate(p_decision, event.text)
            reason = f"{reason} P_cal={p_cal:.3f}"
            reported_confidence = p_cal

        return Decision(
            write=is_signal,
            reason=reason,
            confidence=reported_confidence,
            policy=self.name,
        )
