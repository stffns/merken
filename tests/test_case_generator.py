"""Tests for `merken.training.case_generator`.

Uses a fake LLMClient so the suite stays offline + deterministic.
The real anthropic-backed path is exercised manually via the CLI.
"""

from __future__ import annotations

import pytest

from merken.training.case_generator import (
    CaseGenerator,
    GeneratedCase,
    LLMClient,
    ProtocolClause,
    _extract_json_array,
)

# ----------------------------------------------------------------- json extract

def test_extract_clean_json_array() -> None:
    raw = '[{"prompt": "p1", "truth": "t1"}, {"prompt": "p2", "truth": "t2"}]'
    out = _extract_json_array(raw)
    assert len(out) == 2
    assert out[0]["prompt"] == "p1"


def test_extract_strips_markdown_fences() -> None:
    raw = '```json\n[{"prompt": "p", "truth": "t"}]\n```'
    out = _extract_json_array(raw)
    assert out == [{"prompt": "p", "truth": "t"}]


def test_extract_strips_unlabeled_fences() -> None:
    raw = '```\n[{"prompt": "p", "truth": "t"}]\n```'
    out = _extract_json_array(raw)
    assert out == [{"prompt": "p", "truth": "t"}]


def test_extract_skips_leading_prose() -> None:
    raw = 'Here are the cases:\n[{"prompt": "p", "truth": "t"}]'
    out = _extract_json_array(raw)
    assert len(out) == 1


def test_extract_raises_on_missing_brackets() -> None:
    with pytest.raises(ValueError, match="could not locate JSON array"):
        _extract_json_array("just some prose with no JSON here")


def test_extract_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError, match="JSON array parse failed"):
        _extract_json_array('[{"prompt": "p", broken}]')


def test_extract_raises_on_object_not_array() -> None:
    with pytest.raises(ValueError, match="locate JSON array"):
        _extract_json_array('{"prompt": "p", "truth": "t"}')


# ----------------------------------------------------------------- generation

def _fake_client(canned_response: str) -> LLMClient:
    """Build an LLMClient that ignores prompts and returns a canned string."""
    def _fn(system: str, user: str) -> str:
        return canned_response
    return _fn


def test_generate_emits_n_cases_with_stable_ids() -> None:
    canned = (
        '[{"prompt": "child 5y/o, fever, cough", '
        '  "truth": "amoxicillin 50 mg/kg/day for 5 days"},'
        ' {"prompt": "child 3y/o, productive cough", '
        '  "truth": "amoxicillin 50 mg/kg/day for 5 days"}]'
    )
    gen = CaseGenerator(_fake_client(canned))
    clause = ProtocolClause(
        protocol_id="who_pneumonia_amox",
        text="...",
        metadata={"source": "WHO"},
    )
    cases = gen.generate(clause, n=2)
    assert len(cases) == 2
    assert all(isinstance(c, GeneratedCase) for c in cases)
    assert cases[0].case_id == "who_pneumonia_amox__case_000"
    assert cases[1].case_id == "who_pneumonia_amox__case_001"
    # Metadata carries protocol_id + clause metadata.
    assert cases[0].metadata["protocol_id"] == "who_pneumonia_amox"
    assert cases[0].metadata["source"] == "WHO"


def test_generate_respects_n_zero() -> None:
    gen = CaseGenerator(_fake_client("[]"))
    cases = gen.generate(ProtocolClause("p", "..."), n=0)
    assert cases == []


def test_generate_skips_malformed_rows() -> None:
    canned = (
        '[{"prompt": "ok", "truth": "ok"},'
        ' {"prompt": "missing truth"},'   # incomplete
        ' "not even a dict",'             # wrong type
        ' {"truth": "missing prompt"},'   # incomplete
        ' {"prompt": "valid two", "truth": "valid"}]'
    )
    gen = CaseGenerator(_fake_client(canned))
    cases = gen.generate(ProtocolClause("p", "..."), n=5)
    assert len(cases) == 2
    assert cases[0].prompt == "ok"
    assert cases[1].prompt == "valid two"


def test_generate_strips_whitespace_in_fields() -> None:
    canned = '[{"prompt": "  query  ", "truth": "\\n\\nanswer\\n  "}]'
    gen = CaseGenerator(_fake_client(canned))
    cases = gen.generate(ProtocolClause("p", "..."), n=1)
    assert cases[0].prompt == "query"
    assert cases[0].truth == "answer"


def test_generate_propagates_extraction_errors() -> None:
    gen = CaseGenerator(_fake_client("totally not json"))
    with pytest.raises(ValueError):
        gen.generate(ProtocolClause("p", "..."), n=1)


# ----------------------------------------------------------------- structured

def _fake_structured_client(canned: list):
    """Build a structured client that returns a canned list."""
    def _fn(system: str, user: str) -> list:
        return canned
    return _fn


def test_structured_mode_skips_text_parsing() -> None:
    """structured=True path takes the list directly; never calls
    _extract_json_array. Verifiable because we pass a canned list
    with 2 valid entries."""
    canned = [
        {"prompt": "child fever 38.5", "truth": "amoxicillin 50 mg/kg"},
        {"prompt": "adult cough 5 days", "truth": "wait and observe"},
    ]
    gen = CaseGenerator(_fake_structured_client(canned), structured=True)
    cases = gen.generate(ProtocolClause("p_struct", "..."), n=2)
    assert len(cases) == 2
    assert cases[0].case_id == "p_struct__case_000"
    assert cases[0].prompt == "child fever 38.5"
    assert cases[0].truth == "amoxicillin 50 mg/kg"


def test_structured_mode_rejects_non_list_return() -> None:
    """A structured client MUST return list[dict]; raising on a
    string return surfaces a clear TypeError instead of corrupting
    the output silently."""
    def bad_client(system: str, user: str):
        return "this is text not a list"

    gen = CaseGenerator(bad_client, structured=True)
    with pytest.raises(TypeError, match="must return list"):
        gen.generate(ProtocolClause("p", "..."), n=1)


def test_structured_mode_skips_malformed_entries_same_as_text() -> None:
    """Validation on prompt/truth presence is identical between modes."""
    canned = [
        {"prompt": "ok", "truth": "ok"},
        {"prompt": "missing truth"},        # incomplete
        "not even a dict",                  # wrong type
        {"prompt": "valid two", "truth": "valid"},
    ]
    gen = CaseGenerator(_fake_structured_client(canned), structured=True)
    cases = gen.generate(ProtocolClause("p", "..."), n=4)
    assert len(cases) == 2
    assert cases[0].prompt == "ok"
    assert cases[1].prompt == "valid two"


def test_text_mode_default_unchanged() -> None:
    """structured=False (default) preserves the existing behavior:
    raw text in, _extract_json_array parses, cases come out."""
    raw = '[{"prompt": "p_text", "truth": "t_text"}]'
    gen = CaseGenerator(_fake_client(raw))  # no structured= kwarg
    cases = gen.generate(ProtocolClause("p", "..."), n=1)
    assert len(cases) == 1
    assert cases[0].prompt == "p_text"


def test_text_mode_rejects_list_return_with_actionable_error() -> None:
    """A structured client passed WITHOUT structured=True surfaces
    a clear TypeError pointing at the misuse. Symmetric to the
    structured-mode str check. Per PR #25 review (Copilot)."""
    structured_like = _fake_structured_client(
        [{"prompt": "p", "truth": "t"}]
    )
    gen = CaseGenerator(structured_like)  # MISSING structured=True
    with pytest.raises(TypeError, match="text-mode client must return str"):
        gen.generate(ProtocolClause("p", "..."), n=1)


# ---------------------------------------------------- anthropic structured

class _FakeBlock:
    """Minimal stand-in for anthropic SDK content blocks. Stores
    `input` as-passed (not coerced) so tests can inject None to
    simulate SDK regressions."""
    def __init__(self, *, type: str, text: str = "", input=None):
        self.type = type
        self.text = text
        self.input = input


class _FakeMessages:
    def __init__(self, response_content, capture):
        self._response_content = response_content
        self._capture = capture

    def create(self, **kwargs):
        self._capture.update(kwargs)
        class _Resp:
            content = self._response_content
        return _Resp()


class _FakeAnthropicClient:
    def __init__(self, response_content, capture):
        self.messages = _FakeMessages(response_content, capture)


def _install_fake_anthropic(monkeypatch, response_content, capture):
    """Inject a fake `anthropic` module so the lazy import resolves
    to our stub. Test-local: removed when monkeypatch tears down."""
    import sys
    import types as pytypes
    fake_module = pytypes.ModuleType("anthropic")
    fake_module.Anthropic = lambda api_key: _FakeAnthropicClient(
        response_content, capture
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


def test_anthropic_structured_forces_tool_use_and_parses(monkeypatch) -> None:
    """structured=True forces a single emit_clinical_cases tool call
    and pulls the cases out of the tool_use block."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    response = [
        _FakeBlock(
            type="tool_use",
            input={"cases": [
                {"prompt": "kid 5y/o cough", "truth": "amox 50mg/kg"},
                {"prompt": "kid 3y/o fever", "truth": "amox 50mg/kg"},
            ]},
        ),
    ]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test", structured=True)
    out = fn("system prompt", "user prompt")

    assert out == [
        {"prompt": "kid 5y/o cough", "truth": "amox 50mg/kg"},
        {"prompt": "kid 3y/o fever", "truth": "amox 50mg/kg"},
    ]
    # Verify the request actually forced the tool.
    assert capture["tool_choice"] == {
        "type": "tool", "name": "emit_clinical_cases",
    }
    assert capture["tools"][0]["name"] == "emit_clinical_cases"
    assert capture["system"] == "system prompt"


def test_anthropic_structured_skips_text_blocks_finds_tool_use(monkeypatch) -> None:
    """If the response interleaves text + tool_use, the parser still
    finds the tool_use (e.g. a future model emits commentary first)."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    response = [
        _FakeBlock(type="text", text="Here are the cases:"),
        _FakeBlock(
            type="tool_use",
            input={"cases": [{"prompt": "p", "truth": "t"}]},
        ),
    ]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test", structured=True)
    out = fn("s", "u")
    assert out == [{"prompt": "p", "truth": "t"}]


def test_anthropic_structured_raises_on_no_tool_use(monkeypatch) -> None:
    """No tool_use block in the response surfaces a clear ValueError
    rather than a silent empty list."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    response = [_FakeBlock(type="text", text="I refuse")]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test", structured=True)
    with pytest.raises(ValueError, match="no tool_use block"):
        fn("s", "u")


def test_anthropic_text_mode_unchanged_by_structured_flag(monkeypatch) -> None:
    """structured=False preserves the existing behavior: returns the
    first text block's text. No tools, no tool_choice in the request."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    response = [_FakeBlock(type="text", text='[{"prompt":"p","truth":"t"}]')]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test")  # structured=False default
    out = fn("s", "u")
    assert out == '[{"prompt":"p","truth":"t"}]'
    assert "tools" not in capture
    assert "tool_choice" not in capture


def test_anthropic_text_mode_skips_thought_block(monkeypatch) -> None:
    """Newer Claude models can prepend a `thought` (extended thinking)
    block before the actual text. Reading content[0] blindly would
    drop the answer; iterate to find the first text block instead.
    Per PR #26 review (Gemini)."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    response = [
        _FakeBlock(type="thinking", text="(reasoning trace)"),
        _FakeBlock(type="text", text='[{"prompt":"p","truth":"t"}]'),
    ]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test")
    out = fn("s", "u")
    assert out == '[{"prompt":"p","truth":"t"}]'


def test_anthropic_structured_raises_on_missing_cases_field(monkeypatch) -> None:
    """If the SDK ever returns a tool_use block whose input is missing
    or has the wrong shape (regression / upstream change), surface a
    clear ValueError rather than silently returning []. Per PR #26
    review (Copilot)."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    # tool_use block whose input is None (could happen on a SDK regression).
    response = [_FakeBlock(type="tool_use", input=None)]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test", structured=True)
    with pytest.raises(ValueError, match="non-dict input"):
        fn("s", "u")


def test_anthropic_structured_raises_on_non_list_cases(monkeypatch) -> None:
    """tool_use input present but `cases` field is wrong type."""
    from merken.training.case_generator import default_anthropic_client

    capture: dict = {}
    response = [_FakeBlock(type="tool_use", input={"cases": "not a list"})]
    _install_fake_anthropic(monkeypatch, response, capture)

    fn = default_anthropic_client(api_key="test", structured=True)
    with pytest.raises(ValueError, match="missing/invalid `cases`"):
        fn("s", "u")
