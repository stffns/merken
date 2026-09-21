"""Dataset loading for LongMemEval.

Two paths:

- ``load_fixture(path)`` — load a tiny synthetic JSON file. Used by tests
  and by anyone who wants to dry-run the runner without touching the real
  dataset. The fixture format is a strict subset of the LongMemEval JSON
  schema, so the same parser handles both.
- ``load_longmemeval(cache_dir, subset)`` — download the real LongMemEval
  dataset from HuggingFace. **Currently a stub** — implemented in a
  follow-up commit, once we've actually validated the parser against the
  fixture path. The stub raises ``NotImplementedError`` with the exact
  steps to fetch the file manually in the meantime.

The Conversation dataclass is the *only* thing the runner depends on. As
long as both loaders return ``list[Conversation]``, we can swap dataset
sources without touching the eval logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Turn:
    """One message inside a session."""

    role: str
    content: str


@dataclass
class Conversation:
    """One LongMemEval question with its full haystack of sessions.

    The runner ingests every turn of every haystack session into a fresh
    ``Memory`` instance, then asks the question and checks whether any of
    the top-k recall hits come from a session in ``answer_session_ids``.
    """

    question_id: str
    question: str
    answer: str
    answer_session_ids: list[str]
    haystack_sessions: dict[str, list[Turn]] = field(default_factory=dict)
    question_type: str | None = None
    # Session timestamps as the dataset gives them ("2023/05/20 (Sat) 02:21"),
    # keyed by session id, and the date the question is asked. Needed by
    # time-aware rerankers: on chat logs the supersession signal for
    # knowledge-update questions lives here, not in the text.
    session_dates: dict[str, str] = field(default_factory=dict)
    question_date: str | None = None

    @property
    def n_turns(self) -> int:
        return sum(len(s) for s in self.haystack_sessions.values())

    @property
    def n_sessions(self) -> int:
        return len(self.haystack_sessions)


def _parse_record(record: dict[str, Any]) -> Conversation:
    """Parse one JSON record into a ``Conversation``.

    Tolerates both the fixture format (``haystack_sessions`` as a dict of
    session_id → turns) and the LongMemEval format (``haystack_sessions``
    as a list of session arrays paired with ``haystack_session_ids``).
    """
    qid = str(record["question_id"])
    question = record["question"]
    answer = record.get("answer", "")
    answer_session_ids = [str(s) for s in record.get("answer_session_ids", [])]
    question_type = record.get("question_type")

    raw_sessions = record["haystack_sessions"]
    sessions: dict[str, list[Turn]] = {}

    if isinstance(raw_sessions, dict):
        for sid, turns in raw_sessions.items():
            sessions[str(sid)] = [Turn(role=t["role"], content=t["content"]) for t in turns]
    elif isinstance(raw_sessions, list):
        ids = [str(s) for s in record.get("haystack_session_ids", [])]
        if len(ids) != len(raw_sessions):
            raise ValueError(
                f"haystack_session_ids ({len(ids)}) does not match "
                f"haystack_sessions ({len(raw_sessions)}) for {qid}"
            )
        for sid, turns in zip(ids, raw_sessions, strict=True):
            sessions[sid] = [Turn(role=t["role"], content=t["content"]) for t in turns]
    else:
        raise TypeError(f"unexpected haystack_sessions type for {qid}: {type(raw_sessions)}")

    session_dates: dict[str, str] = {}
    raw_dates = record.get("haystack_dates")
    if isinstance(raw_dates, list) and len(raw_dates) == len(sessions):
        session_dates = dict(zip(sessions.keys(), (str(d) for d in raw_dates), strict=True))

    return Conversation(
        question_id=qid,
        question=question,
        answer=answer,
        answer_session_ids=answer_session_ids,
        haystack_sessions=sessions,
        question_type=question_type,
        session_dates=session_dates,
        question_date=record.get("question_date"),
    )


def load_fixture(path: str | Path) -> list[Conversation]:
    """Load a tiny synthetic LongMemEval-shaped fixture from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"fixture root must be a list, got {type(data).__name__}")
    return [_parse_record(rec) for rec in data]


# --------------------------------------------------------------- HF download

# We deliberately use the *cleaned* dataset (xiaowu0162/longmemeval-cleaned).
# The original xiaowu0162/longmemeval is marked deprecated by its author —
# noisy haystack sessions interfered with answer correctness.
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

# Subset → (filename, approx size in MB) for README/error messages.
_SUBSETS: dict[str, tuple[str, int]] = {
    "longmemeval_oracle": ("longmemeval_oracle.json", 15),
    "longmemeval_s": ("longmemeval_s_cleaned.json", 277),
    "longmemeval_m": ("longmemeval_m_cleaned.json", 2737),
}

DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


def _download_to(url: str, dest: Path) -> None:
    """Stream a URL to a file. No extra deps — stdlib only."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def load_longmemeval(
    subset: str = "longmemeval_s",
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
) -> list[Conversation]:
    """Load (and optionally download) a LongMemEval split.

    Parameters
    ----------
    subset:
        One of ``longmemeval_oracle`` (15 MB, 500 questions, oracle context
        only — useful for parser sanity, not for retrieval evaluation),
        ``longmemeval_s`` (~277 MB, 500 questions with full distractor
        haystacks — the real R@k benchmark), or ``longmemeval_m`` (~2.7 GB,
        much larger haystacks).
    cache_dir:
        Where to cache downloaded files. Defaults to
        ``experiments/longmemeval/.cache/`` (gitignored).
    download:
        If False and the file is missing, raises ``FileNotFoundError``
        instead of fetching. Useful in tests.
    """
    if subset not in _SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; choose from {sorted(_SUBSETS)}")

    filename, _ = _SUBSETS[subset]
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    path = cache / filename

    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} not found and download=False; fetch it manually "
                f"from {_HF_BASE}/{filename}"
            )
        _download_to(f"{_HF_BASE}/{filename}", path)

    return load_fixture(path)
