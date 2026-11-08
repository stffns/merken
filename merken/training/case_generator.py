"""Generate (prompt, truth) clinical-case pairs from Type B protocol clauses.

The large model is the ground-truth oracle: it sees the authoritative
protocol clause and produces N realistic clinical scenarios whose
correct answer follows STRICTLY from the clause. The smaller model
(``response_generator.py``) will later attempt the same prompts
WITHOUT the protocol; the gap between truth and small-model response
is the midloop training signal.

Mock-friendly design: ``CaseGenerator`` takes a ``LLMClient``
callable. The default implementations wrap the anthropic SDK or the
google-genai SDK; tests pass a fake that returns canned JSON or a
canned list. This keeps the suite offline + zero-cost while preserving
the real-world path.

Two client contracts are supported:

- **Text-mode** (default): the client returns a raw JSON string and
  ``CaseGenerator`` parses it with ``_extract_json_array`` (tolerant
  of markdown fences + leading prose). Use this when the upstream
  API doesn't enforce JSON structure on the wire.
- **Structured-mode**: the client returns ``list[dict]`` directly,
  bypassing the text parsing. Use this when the API guarantees the
  shape via response_mime_type / response_schema (Gemini's
  structured output, OpenAI's response_format=json_schema, etc).

Pass ``structured=True`` to ``CaseGenerator`` when wiring a
structured-mode client so the parser step is skipped. The recovery
script in ``experiments/midloop_pilot/recover_failed.py`` (PR #23)
demonstrated structured-mode is faster + zero JSON parse failures
on the live MedLocal corpus.

Both backends support structured mode:

- ``default_gemini_client(structured=True)``: SDK enforces a
  Pydantic schema on the wire via ``response_schema``.
- ``default_anthropic_client(structured=True)``: forces the model
  to call a single ``emit_clinical_cases`` tool whose input schema
  is validated by the Anthropic runtime before the response returns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Text-mode: (system_prompt, user_prompt) -> raw model output text.
LLMClient = Callable[[str, str], str]

# Structured-mode: (system_prompt, user_prompt) -> parsed list of dicts.
# Used by clients backed by structured-output APIs that guarantee the
# response shape on the wire (e.g. Gemini response_schema).
LLMStructuredClient = Callable[[str, str], list[dict]]


@dataclass
class ProtocolClause:
    """One Type B authoritative source clause to expand into cases.

    ``protocol_id`` is the canonical identifier (e.g.
    ``who_pneumonia_amoxicillin_dose``). Used in case_id derivation
    and downstream traceability so a generated case can always be
    linked back to its source clause.
    """
    protocol_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedCase:
    """One synthetic clinical case ready for the response generator."""
    case_id: str
    prompt: str
    truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


_SYSTEM_PROMPT = (
    "You generate realistic clinical scenarios for training a memory-"
    "intervention model. Given an authoritative protocol clause, you "
    "produce N scenarios where a clinician would need exactly this "
    "knowledge.\n\n"
    "Each scenario must have:\n"
    "- prompt: a realistic clinician question or situation, "
    "1-3 sentences. Concrete details (age, weight, signs).\n"
    "- truth: the correct answer DERIVED STRICTLY from the protocol "
    "clause. Do not add information not present in the clause. If the "
    "clause is silent on a detail, do not invent it.\n\n"
    "Respond ONLY with a JSON array of objects with `prompt` and "
    "`truth` keys. No prose, no markdown fences, no commentary."
)

_USER_TEMPLATE = (
    "Protocol id: {protocol_id}\n"
    "Protocol clause:\n```\n{protocol_text}\n```\n\n"
    "Generate {n} distinct clinical scenarios. Vary patient details "
    "(age, weight, presentation) but keep the protocol-derived "
    "answer correct in every case. Output the JSON array now."
)


def _extract_json_array(raw: str) -> list[dict]:
    """Tolerant JSON extraction.

    LLMs sometimes wrap responses in ```json ... ``` fences or add
    a leading sentence. We strip those by scanning for the first
    `[` and the matching final `]`. If parsing fails we re-raise
    with the original text in the message so the caller can debug.
    """
    text = raw.strip()
    # Strip code fences.
    fence_match = re.match(r"```(?:json)?\s*", text)
    if fence_match:
        text = text[fence_match.end():]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # If the model added a sentence before the array, find the array.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"could not locate JSON array in LLM response: {raw[:200]!r}"
        )
    candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON array parse failed: {e}; candidate={candidate[:200]!r}"
        ) from e
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    return parsed


class CaseGenerator:
    """Turn one ProtocolClause into N GeneratedCase rows via an LLM.

    Construct once per pipeline run; reuse across clauses. Stateless
    aside from the LLM client reference.
    """

    def __init__(
        self,
        llm_client: LLMClient | LLMStructuredClient,
        *,
        structured: bool = False,
    ) -> None:
        """Build a CaseGenerator.

        ``structured=False`` (default): ``llm_client`` returns a raw
        JSON string; we parse it with ``_extract_json_array``
        (tolerant of markdown fences + prose). Use with
        ``default_anthropic_client`` and any API where you cannot
        enforce JSON structure on the wire.

        ``structured=True``: ``llm_client`` returns ``list[dict]``
        directly, bypassing text parsing. Use with
        ``default_gemini_client(structured=True)`` and any API that
        guarantees the response shape (Gemini response_schema,
        OpenAI json_schema, etc).
        """
        self._llm = llm_client
        self._structured = structured

    def generate(self, clause: ProtocolClause, n: int = 5) -> list[GeneratedCase]:
        if n <= 0:
            return []
        user = _USER_TEMPLATE.format(
            protocol_id=clause.protocol_id,
            protocol_text=clause.text,
            n=n,
        )
        result = self._llm(_SYSTEM_PROMPT, user)
        if self._structured:
            if not isinstance(result, list):
                raise TypeError(
                    f"structured=True client must return list[dict]; "
                    f"got {type(result).__name__}"
                )
            rows = result
        else:
            # Symmetric defensive check: a structured client passed
            # without `structured=True` would return a list and the
            # downstream _extract_json_array(...).strip() would fail
            # with an unhelpful AttributeError. Surface the misuse
            # explicitly. Per PR #25 review (Copilot, 2026-04-19).
            if not isinstance(result, str):
                raise TypeError(
                    f"text-mode client must return str (got "
                    f"{type(result).__name__}). If your client returns "
                    f"list[dict] (e.g. default_gemini_client(structured=True)), "
                    f"pass `structured=True` to CaseGenerator."
                )
            rows = _extract_json_array(result)

        cases: list[GeneratedCase] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            truth = row.get("truth")
            if not prompt or not truth:
                continue
            case_id = f"{clause.protocol_id}__case_{i:03d}"
            cases.append(GeneratedCase(
                case_id=case_id,
                prompt=str(prompt).strip(),
                truth=str(truth).strip(),
                metadata={"protocol_id": clause.protocol_id, **clause.metadata},
            ))
        return cases


_ANTHROPIC_TOOL_NAME = "emit_clinical_cases"
_ANTHROPIC_CASES_TOOL = {
    "name": _ANTHROPIC_TOOL_NAME,
    "description": (
        "Emit the generated clinical scenarios. Each item must derive "
        "STRICTLY from the supplied protocol clause."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Realistic clinician question or "
                                "situation, 1-3 sentences."
                            ),
                        },
                        "truth": {
                            "type": "string",
                            "description": (
                                "Correct answer derived strictly from "
                                "the protocol clause."
                            ),
                        },
                    },
                    "required": ["prompt", "truth"],
                },
            },
        },
        "required": ["cases"],
    },
}


def default_anthropic_client(
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    *,
    structured: bool = False,
) -> LLMClient | LLMStructuredClient:
    """Build a client backed by the anthropic SDK.

    Lazy-imports anthropic so the module remains importable without
    the SDK installed (tests use fake clients). When the SDK is
    needed it is imported once at construction time.

    The default model is Claude Sonnet 4.6 -- per project preference,
    use the latest Sonnet for case generation (capable enough to
    follow the strict-derivation instruction without leaking
    out-of-clause info).

    ``structured=False`` (default) returns an ``LLMClient`` (text
    mode) that returns the raw model text for ``_extract_json_array``
    to parse.

    ``structured=True`` returns an ``LLMStructuredClient`` that
    forces the model to call a single ``emit_clinical_cases`` tool
    with a typed ``list[{prompt, truth}]`` input. The Anthropic
    runtime validates the tool input against the JSON schema before
    returning, so the response is GUARANTEED to be valid JSON with
    the right shape -- mirrors the parse-failure-free guarantee
    that ``default_gemini_client(structured=True)`` provides via
    ``response_schema``. Pair with ``CaseGenerator(client, structured=True)``.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    if not structured:
        def _fn_text(system: str, user: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            # Iterate to find the first text-bearing block. Newer
            # models can prepend a `thought` (extended thinking) or
            # other non-text block; reading content[0] blindly drops
            # the actual answer. Per PR #26 review (Gemini).
            for block in resp.content or []:
                if getattr(block, "type", None) != "text":
                    continue
                text = getattr(block, "text", "") or ""
                if text:
                    return text
            return ""
        return _fn_text

    def _fn_struct(system: str, user: str) -> list[dict]:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[_ANTHROPIC_CASES_TOOL],
            tool_choice={"type": "tool", "name": _ANTHROPIC_TOOL_NAME},
        )
        for block in resp.content or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            # Validate the tool input shape rather than silently
            # returning []. The runtime SHOULD enforce the schema,
            # but a SDK regression or upstream change shouldn't
            # corrupt the dataset. Per PR #26 review (Copilot).
            payload = getattr(block, "input", None)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Anthropic tool_use returned non-dict input: "
                    f"{type(payload).__name__}"
                )
            cases = payload.get("cases")
            if not isinstance(cases, list):
                raise ValueError(
                    f"Anthropic tool_use payload missing/invalid "
                    f"`cases` field: got {type(cases).__name__}"
                )
            return [
                {"prompt": c["prompt"], "truth": c["truth"]}
                for c in cases
                if isinstance(c, dict) and "prompt" in c and "truth" in c
            ]
        # Tool-forcing should make this unreachable; surface the
        # failure clearly rather than returning [] silently.
        raise ValueError(
            "Anthropic structured call returned no tool_use block; "
            "likely a transient API issue or model refusal."
        )

    return _fn_struct


def default_gemini_client(
    api_key: str,
    model: str = "gemini-2.5-flash",
    *,
    structured: bool = True,
    timeout_s: float = 60.0,
) -> LLMStructuredClient | LLMClient:
    """Build a Gemini-backed client. Defaults to structured-output mode.

    Lazy-imports google-genai so the module is importable without
    the SDK installed (tests use fake clients). Per PR #25 review
    (Gemini code-assist + Copilot, 2026-04-19) we use the SDK's
    NATIVE features instead of wrapping calls in a thread pool:

    - ``system_instruction`` on ``GenerateContentConfig`` carries
      the system prompt without manual string concat.
    - ``http_options.timeout`` on the same config gives the SDK's
      HTTP layer a hard deadline. The SDK closes the connection
      cleanly on timeout, unlike a futures.Future cancel() which
      cannot stop in-flight Python work and leaves a poisoned
      worker thread behind.

    ``structured=True`` (RECOMMENDED, default) returns an
    ``LLMStructuredClient`` -- the SDK enforces a Pydantic schema
    (``list[_GeneratedCaseSchema]``) on the wire so the response
    is GUARANTEED to be valid JSON with the right shape. The
    recovery script in ``experiments/midloop_pilot/recover_failed.py``
    demonstrated this is faster + zero JSON parse failures on the
    77-protocol MedLocal corpus that text-mode left 2 protocols
    failing on. Pair with ``CaseGenerator(client, structured=True)``.

    ``structured=False`` returns an ``LLMClient`` (text mode) for
    callers who explicitly want raw text + tolerant parsing.

    ``timeout_s=0`` disables the per-call timeout (NOT recommended
    -- the SDK can hang in SSL_read for many minutes under
    transient network conditions; we observed a 23-min hang during
    the 77-protocol scale-up before adding this protection).
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        # Library code raises ImportError (not SystemExit) so the
        # caller can catch + degrade. The install hint is in the
        # message so a top-level handler can show it to the user
        # without a traceback.
        raise ImportError(
            "google-genai not installed. Run `pip install google-genai`. "
            f"[{e}]"
        ) from e
    from pydantic import BaseModel

    class _GeneratedCaseSchema(BaseModel):
        prompt: str
        truth: str

    # http_options.timeout is in milliseconds. timeout_s=0 -> omit
    # the http_options entirely (SDK default = no per-call timeout).
    http_options = (
        types.HttpOptions(timeout=int(timeout_s * 1000))
        if timeout_s and timeout_s > 0
        else None
    )

    client = genai.Client(api_key=api_key, http_options=http_options)

    def _build_config(system: str):
        kwargs: dict = {"system_instruction": system}
        if structured:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = list[_GeneratedCaseSchema]
        return types.GenerateContentConfig(**kwargs)

    def _fn(system: str, user: str):
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=_build_config(system),
        )
        if structured:
            parsed = resp.parsed
            if parsed is not None:
                return [
                    {"prompt": c.prompt, "truth": c.truth} for c in parsed
                ]
            # Fallback: schema didn't materialize; surface a clear
            # error rather than returning [] silently.
            text = (resp.text or "").strip()
            if not text:
                raise ValueError(
                    "Gemini structured call returned no parsed schema "
                    "and no text; likely a transient API issue."
                )
            return _extract_json_array(text)
        return (resp.text or "").strip()

    return _fn
