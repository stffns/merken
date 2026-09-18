"""The Memory class — merken's only public surface for now.

Phase 1 wiring (CONSTITUTION §11 + ultra-plan):

- ``remember`` runs the configured ``should_remember`` policy first, writes an
  audit row for every decision (write *or* skip), and only then calls
  ``vstash.Memory.remember`` if the decision says so.
- ``recall`` is still a thin pass-through over ``vstash.Memory.search``. It
  is naturally isolated from the audit log because audit lives in its own
  vstash collection (``merken_audit``).
- ``audit`` lets you query the decision log directly.

The boundary with vstash is sacred (CONSTITUTION §6): every storage call
goes through vstash's public API. Engram never reads the SQLite tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import vstash

from merken.audit import (
    AUDIT_COLLECTION,
    AUDIT_LAYER,
    TOMBSTONE_COLLECTION,
    TOMBSTONE_LAYER,
    format_audit_row,
    format_consolidate_audit_row,
    format_forget_audit_row,
    format_recall_audit_row,
    format_tombstone_row,
)
from merken.consolidation import (
    ConsolidationResult,
    Fact,
    cluster_by_embedding,
    cluster_by_jaccard,
    cluster_by_recall,
    fact_fingerprint,
    materialize_fact,
)
from merken.policies.should_consolidate import (
    ConsolidateContext,
    ConsolidateDecider,
    ConsolidationDecision,
    PeriodicConsolidator,
)
from merken.policies.should_forget import (
    ForgetContext,
    ForgetDecider,
    ForgetDecision,
    NeverForget,
)
from merken.policies.should_recall import (
    LayeredRecaller,
    RecallContext,
    RecallDecider,
    RecallPlan,
)
from merken.policies.should_remember import (
    ChainedWriteDecider,
    HeuristicWriteDecider,
    ShadowWriteDecider,
)
from merken.policies.types import Decision, Event, WriteContext, WriteDecider
from merken.reranking import Reranker
from merken.sourcing import is_safely_mutable

if TYPE_CHECKING:
    from vstash import IngestResult, SearchResult

    from merken.policies.midloop import (
        MidloopDecider,
        MidloopDecision,
        StepObservation,
    )

DEFAULT_LAYER = "episodic"
DEFAULT_COLLECTION = "default"


def _default_write_decider() -> WriteDecider:
    """Build the write decider with env-based shadow / primary wiring.

    Precedence (MERKEN_PRIMARY wins if both are set -- typing PRIMARY
    is the more deliberate act):

    1. ``MERKEN_PRIMARY`` set -> ``ChainedWriteDecider(Heuristic, classifier)``.
       Heuristic keeps hygiene gates; classifier decides write/skip for
       anything that passes the gates.
    2. ``MERKEN_SHADOW`` set -> ``ShadowWriteDecider(Heuristic, classifier)``.
       Heuristic stays authoritative; classifier only annotates the
       audit reason.
    3. Neither set (or classifier construction fails) -> plain
       ``HeuristicWriteDecider()``. An opt-in classifier never blocks
       writes, even when misconfigured.
    """
    from merken._shadow import load_primary_from_env, load_shadow_from_env

    heuristic = HeuristicWriteDecider()

    try:
        primary_classifier = load_primary_from_env()
    except Exception:
        primary_classifier = None
    if primary_classifier is not None:
        return ChainedWriteDecider(heuristic, primary_classifier)

    try:
        shadow = load_shadow_from_env()
    except Exception:
        return heuristic
    if shadow is None:
        return heuristic
    return ShadowWriteDecider(heuristic, shadow)


def _resolve_vstash_embed_model(vstash_memory: vstash.Memory) -> str:
    """Return the embedding model this vstash Memory is actually using.

    Priority:

    1. ``store_meta.embedding_model`` from the vstash SQLite DB. This is
       authoritative for an existing store because it records the model
       vstash used to *ingest* the chunks that are now sitting in the
       vector index. Reading any other model at clustering time creates
       a silent vector-space mismatch between "how merken groups" and
       "how vstash retrieves."
    2. ``vstash.config.EmbeddingsConfig().model`` — vstash's current
       factory default. Used when the store is fresh (no ingests yet,
       so no ``store_meta`` row) and when reading the DB fails for any
       reason.

    The earlier implementation hardcoded ``BAAI/bge-small-en-v1.5`` as
    ``DEFAULT_EMBED_MODEL``, which was wrong for any user who had
    configured vstash with a different model. On Jay's live vstash
    (which uses ``paraphrase-multilingual-MiniLM-L12-v2``), merken
    consolidation was re-embedding chunks with bge-small and then
    writing facts that vstash indexed back with multilingual. Clusters
    were internally coherent but misaligned with the actual retrieval
    vector space. Caught 2026-04-09 while verifying the store wasn't
    mixed (it wasn't — but merken was pretending it was bge-small).
    """
    import sqlite3

    from vstash.config import EmbeddingsConfig

    fallback = EmbeddingsConfig().model

    db_path = getattr(getattr(vstash_memory, "_store", None), "db_path", None)
    if db_path is None:
        return fallback

    try:
        con = sqlite3.connect(str(db_path))
        try:
            row = con.execute(
                "SELECT value FROM store_meta WHERE key = ?",
                ("embedding_model",),
            ).fetchone()
        finally:
            con.close()
    except Exception:
        return fallback

    if row and row[0]:
        return row[0]
    return fallback


@dataclass(frozen=True)
class ForgetResult:
    """The outcome of a ``Memory.forget`` call.

    ``tombstoned`` is the list of event paths that were tombstoned
    in this call. ``skipped`` pairs each surviving event path with
    the reason the decider gave for keeping it.
    """

    tombstoned: list[str]
    skipped: list[tuple[str, str]]
    events_examined: int
    decider: str


@dataclass(frozen=True)
class RememberResult:
    """The outcome of a ``Memory.remember`` call.

    ``written`` reflects whether the event actually landed in vstash.
    ``decision`` is the policy verdict (always populated, even on skips).
    ``ingest`` is the underlying ``vstash.IngestResult`` when written, else
    ``None``.
    """

    written: bool
    decision: Decision
    ingest: IngestResult | None


class Memory:
    """Agent-loop memory, backed by a single ``vstash.Memory`` instance.

    Parameters
    ----------
    project:
        Logical project name. Becomes the vstash ``project`` tag on every
        write and the default filter on every read.
    db:
        Optional path to the vstash SQLite file. When omitted, vstash uses
        its default location.
    config:
        Optional path to a vstash config file/profile.
    write_decider:
        Optional custom ``should_remember`` policy. Defaults to
        ``HeuristicWriteDecider()``. Pass ``AlwaysWrite()`` for the
        store-everything baseline used in benchmarks.
    """

    def __init__(
        self,
        project: str,
        *,
        db: str | Path | None = None,
        config: str | Path | None = None,
        collection: str = DEFAULT_COLLECTION,
        write_decider: WriteDecider | None = None,
        consolidate_decider: ConsolidateDecider | None = None,
        recall_decider: RecallDecider | None = None,
        forget_decider: ForgetDecider | None = None,
        midloop_decider: MidloopDecider | None = None,
        temporal_weight: float = 0.0,
        reranker: Reranker | None = None,
        recall_overfetch: int = 4,
        trajectory_window: int = 20,
    ) -> None:
        self.project = project
        self.collection = collection
        self._temporal_weight = temporal_weight
        # Optional post-retrieval reranker (``merken.reranking.Reranker``).
        # When set, ``recall`` over-fetches ``top_k * recall_overfetch``
        # candidates from every layer before reranking. None = off.
        self._reranker = reranker
        self._recall_overfetch = max(1, recall_overfetch)
        self._vstash = vstash.Memory(
            config=config,
            project=project,
            db=db,
            collection=collection,
        )
        self._write_decider: WriteDecider = write_decider or _default_write_decider()
        self._consolidate_decider: ConsolidateDecider = (
            consolidate_decider or PeriodicConsolidator()
        )
        self._recall_decider: RecallDecider = recall_decider or LayeredRecaller()
        self._forget_decider: ForgetDecider = forget_decider or NeverForget()

        # Midloop default is Noop (never intervenes). The trajectory
        # is per-Memory instance state, intentionally NOT persisted
        # step-a-step (per midloop spec section "Trajectory, no
        # snapshot"). Callers reuse the same Memory instance across a
        # session to maintain trajectory coherence.
        from collections import deque

        from merken.policies.midloop import NoopMidloopDecider
        self._midloop_decider = midloop_decider or NoopMidloopDecider()
        self._trajectory: deque = deque(maxlen=trajectory_window)
        self._prev_midloop_decision = None

        # Wire the write decider's hydration to vstash so cross-invocation
        # dedup works without the caller knowing about _seen sets. If the
        # decider doesn't support hydration (e.g. AlwaysWrite, or a
        # user's custom decider), this is a silent no-op.
        if hasattr(self._write_decider, "set_hydrate_fn"):
            self._write_decider.set_hydrate_fn(self._hydrate_write_seen)

    def _hydrate_write_seen(self) -> list[str]:
        """Pull existing episodic texts for the write decider's dedup set.

        Called lazily on the decider's first ``decide()`` after init.
        One ``list()`` + N ``get_document_chunks()`` calls. Expensive
        per init but O(1) per subsequent write.
        """
        docs = self._vstash.list(
            collection=self.collection,
            layer=DEFAULT_LAYER,
        )
        out: list[str] = []
        for doc in docs:
            try:
                chunks = self._vstash.get_document_chunks(
                    doc.path,
                    collection=self.collection,
                )
                text = " ".join(chunks).strip()
                if text:
                    out.append(text)
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------ write

    def remember(
        self,
        text: str,
        *,
        layer: str = DEFAULT_LAYER,
        title: str | None = None,
        tags: str | None = None,
    ) -> RememberResult:
        """Submit an event to memory.

        The configured ``should_remember`` policy decides whether the event
        is written. Both outcomes (write and skip) produce an audit row.
        """
        event = Event(text=text, layer=layer, title=title, tags=tags)
        ctx = WriteContext(project=self.project)

        decision = self._write_decider.decide(event, ctx)
        self._write_audit(event, decision)

        if not decision.write:
            return RememberResult(written=False, decision=decision, ingest=None)

        ingest = self._vstash.remember(
            text,
            title=title,
            collection=self.collection,
            layer=layer,
            tags=tags,
        )

        # vstash has its own guardrails (e.g. it rejects text shorter
        # than ~20 chars with status="empty", no error, no chunks).
        # Surface that as a failed write instead of claiming success —
        # a downstream caller who reads ``written=True`` on a
        # vstash-rejected ingest would be lied to.
        status = getattr(ingest, "status", "ok")
        if status != "ok":
            override = Decision(
                write=False,
                reason=f"vstash_rejected:{status}",
                confidence=1.0,
                policy=decision.policy,
            )
            self._write_audit(event, override)
            return RememberResult(written=False, decision=override, ingest=ingest)

        return RememberResult(written=True, decision=decision, ingest=ingest)

    # ------------------------------------------------------------------- read

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        layer: str | None = None,
        temporal_weight: float | None = None,
    ) -> list[SearchResult]:
        """Read from memory.

        When ``layer`` is not specified, routes through the configured
        ``should_recall`` policy: the decider returns a ``RecallPlan``
        with a priority-ordered list of per-layer requests, we query
        each in turn, dedupe by vstash path, and truncate to the
        caller's ``top_k``.

        When ``layer`` IS specified, the decider is bypassed and the
        call is a pass-through to ``vstash.Memory.search`` with that
        layer. This is the escape hatch for benchmarks and for callers
        that already know exactly which layer they want.

        Either way, recall is scoped to the merken collection and
        never returns rows from the ``merken_audit`` collection.
        """
        if layer is not None:
            return self._vstash.search(
                query,
                top_k=top_k,
                collection=self.collection,
                layer=layer,
            )

        ctx = RecallContext(project=self.project, top_k=top_k)
        plan = self._recall_decider.decide(query, ctx)

        # Over-fetch budget. Any reranker (recency or a ``Reranker``)
        # needs more candidates than the caller's top_k to reorder
        # meaningfully. Before 2026-09 this only widened the *merge*
        # limit while each layer kept fetching its fixed plan budget
        # (5 + 3 by default), so rerankers only ever saw 8 candidates
        # regardless of top_k. The per-layer fetch below now honours
        # the over-fetch limit too.
        tw = temporal_weight if temporal_weight is not None else self._temporal_weight
        if self._reranker is not None:
            fetch_limit = top_k * self._recall_overfetch
        elif tw > 0.0:
            fetch_limit = top_k * 2
        else:
            fetch_limit = top_k
        reranker_name = (
            self._reranker.name if self._reranker is not None
            else ("rerank_by_recency" if tw > 0.0 else None)
        )
        self._write_recall_audit(query, plan, reranker=reranker_name)

        # Fetch from every layer first, then interleave round-robin.
        # The earlier implementation drained each layer sequentially,
        # which meant that when the first layer returned ``top_k``
        # hits, later layers were never consulted. On a smoke test
        # against real vstash content (2026-04-09), that made the
        # "episodic fallback" a fallback only in name: a Kafka
        # meeting note (singleton in episodic) never surfaced for
        # the "Kafka merchant pipeline" query because four semantic
        # facts filled the top_k=3 budget first.
        #
        # Round-robin interleave guarantees every requested layer
        # gets at least one slot in the final list (until the
        # user's top_k is reached), which is what "layered recall"
        # was supposed to mean all along.
        per_layer_hits: list[list[SearchResult]] = []
        empty_budget = 0
        for req in plan.layers:
            # When a prior layer returned nothing, redistribute its
            # budget to subsequent layers so the caller's top_k is
            # still satisfiable. Without this, an empty semantic
            # layer (no consolidation) wastes its budget and the
            # episodic layer only fetches its own smaller quota.
            effective_top_k = req.top_k + empty_budget
            if fetch_limit > top_k:
                effective_top_k = max(effective_top_k, fetch_limit)
            layer_hits = self._vstash.search(
                query,
                top_k=effective_top_k,
                collection=self.collection,
                layer=req.layer,
            )
            hits_list = list(layer_hits)
            if hits_list:
                per_layer_hits.append(hits_list)
                empty_budget = 0
            else:
                empty_budget += req.top_k

        seen_paths: set[str] = set()
        merged: list[SearchResult] = []
        max_len = max((len(hs) for hs in per_layer_hits), default=0)
        for i in range(max_len):
            for layer_hits in per_layer_hits:
                if i >= len(layer_hits):
                    continue
                h = layer_hits[i]
                path = getattr(h, "path", None)
                if path is not None and path in seen_paths:
                    continue
                if path is not None:
                    seen_paths.add(path)
                merged.append(h)
                if len(merged) >= fetch_limit:
                    break
            if len(merged) >= fetch_limit:
                break

        if tw > 0.0:
            from merken.reranking import rerank_by_recency

            merged = rerank_by_recency(merged, temporal_weight=tw)

        if self._reranker is not None and merged:
            merged = list(self._reranker.rerank(query, merged))

        return merged[:top_k]

    def recall_with_briefs(
        self,
        query: str,
        *,
        top_k: int = 5,
        brief_k: int = 3,
        max_brief_tokens: int = 8000,
    ) -> tuple[list[Any], list[str]]:
        """Recall with separate brief-layer search.

        Returns ``(episodic_hits, brief_texts)`` where:
        - ``episodic_hits`` are the normal recall results (top_k)
        - ``brief_texts`` are the top brief_k briefs from the semantic
          layer filtered to method:brief_v1, ordered by relevance

        The caller should prepend brief_texts to the LLM context before
        the episodic hits. This avoids briefs competing with episodic
        docs in the same retrieval pool.
        """
        # 1. Search briefs in semantic layer
        brief_hits = self._vstash.search(
            query,
            top_k=brief_k * 2,  # over-fetch, then filter
            collection=self.collection,
            layer="semantic",
        )
        brief_texts = []
        token_budget = max_brief_tokens
        for h in brief_hits:
            tags = getattr(h, "tags", "") or ""
            if "method:brief_v1" not in tags:
                continue
            text = h.text
            # Rough token estimate: chars / 4
            est_tokens = len(text) // 4
            if token_budget - est_tokens < 0:
                break
            brief_texts.append(text)
            token_budget -= est_tokens
            if len(brief_texts) >= brief_k:
                break

        # 2. Normal episodic recall
        episodic_hits = self.recall(query, top_k=top_k)

        return episodic_hits, brief_texts

    # ----------------------------------------------------------- consolidate

    def consolidate(
        self,
        *,
        method: str = "embedding_v1",
        min_cluster: int = 2,
        embedding_threshold: float = 0.70,
        embedding_linkage: str = "complete",
        recall_top_k: int = 5,
        jaccard_threshold: float = 0.5,
        force: bool = False,
        synthesize_fn: Any | None = None,
        materialize_fn: Any | None = None,
    ) -> ConsolidationResult:
        """Cluster episodic events into semantic facts. Phase 2, no LLM.

        Pulls every ``layer="episodic"`` document in this merken
        collection, reassembles the text from its chunks, clusters
        them, and writes one ``layer="semantic"`` fact per cluster of
        size ≥ ``min_cluster``. Singletons are skipped (they're
        already findable as episodic).

        Every run, write or skip, logs a ``should_consolidate`` audit
        row. Fact titles are a stable hash of ``derived_from`` so
        re-running is idempotent — no duplicate semantic rows.

        Parameters
        ----------
        method:
            Clustering strategy. ``"embedding_v1"`` (default) embeds
            each event via the vstash embedder and clusters by raw
            cosine similarity above ``embedding_threshold``. This is
            the only method that reliably handles paraphrased natural
            language. ``"recall_v1"`` delegates to vstash's hybrid
            search but is brittle on small corpora — see
            ``cluster_by_recall`` for the limitation. ``"jaccard_v1"``
            is near-duplicate only; see ``cluster_by_jaccard``.
        min_cluster:
            Minimum cluster size for a fact to be written. ``2`` means
            "require corroboration across at least two episodic
            events." Singletons always stay in the episodic layer.
        embedding_threshold:
            Cosine similarity cutoff when ``method="embedding_v1"``.
            ``0.70`` was picked by a grid search across the three
            loop_quality scenarios (2026-04-09). It achieves 100%
            query pass rate and 100% cluster purity on all three.
            See ``experiments/loop_quality/RESULTS.md`` for the
            full grid and trade-offs.
        embedding_linkage:
            ``"complete"`` (default) or ``"single"``. Complete-link
            merges only when every cross-cluster pair is above
            ``embedding_threshold``; single-link merges on any one
            above-threshold edge. Complete avoids the cascade where
            one weak-but-genuine edge contaminates a transitive
            cluster. See ``cluster_by_embedding`` for the trade-off.
        recall_top_k:
            Neighbors per event when ``method="recall_v1"``.
        jaccard_threshold:
            Token-set Jaccard similarity when ``method="jaccard_v1"``.
        force:
            Bypass the ``should_consolidate`` decider and always run.
            Useful in tests and when the caller has already decided.
        materialize_fn:
            Optional ``cluster -> Fact`` callable that takes full control
            of materialization: the fact text *and* its ``derived_from``.
            Use it when the materializer must expel cluster members
            (e.g. ``merken.classifiers.jev.JevMaterializer``). Takes
            precedence over ``synthesize_fn``.
        """
        docs = self._vstash.list(
            collection=self.collection,
            layer="episodic",
        )

        # Type-B (authoritative) events are excluded from consolidation
        # via the fail-closed `is_safely_mutable` predicate -- see
        # `merken/sourcing.py`. An event with an unrecognized
        # `source:<value>` tag is conservatively treated as immutable
        # so that protocol-class content cannot be silently summarized
        # away.
        events: list[tuple[str, str]] = []
        for doc in docs:
            if not is_safely_mutable(getattr(doc, "tags", None)):
                continue
            chunks = self._vstash.get_document_chunks(
                doc.path,
                collection=self.collection,
            )
            full_text = " ".join(chunks).strip()
            if full_text:
                events.append((doc.path, full_text))

        ctx = ConsolidateContext(project=self.project)
        decision = self._consolidate_decider.decide(len(events), ctx)
        self._write_consolidate_audit(decision, len(events))

        if not decision.proceed and not force:
            return ConsolidationResult(
                events_examined=len(events),
                facts_written=0,
                facts=[],
                skipped=True,
                reason=decision.reason,
                decider=decision.policy,
                method=method,
            )

        if method == "embedding_v1":
            model_name = _resolve_vstash_embed_model(self._vstash)

            def _embed(texts: list[str]) -> list[Any]:
                from vstash.embed import embed_texts
                return embed_texts(
                    texts,
                    model_name=model_name,
                    backend="auto",
                )
            clusters = cluster_by_embedding(
                events,
                embed_fn=_embed,
                threshold=embedding_threshold,
                linkage=embedding_linkage,
            )
            reason = (
                f"clustered_embedding>={embedding_threshold}"
                f"_linkage={embedding_linkage}"
                f"_mincluster={min_cluster}"
            )
        elif method == "jaccard_v1":
            clusters = cluster_by_jaccard(events, threshold=jaccard_threshold)
            reason = f"clustered_jaccard>={jaccard_threshold}_mincluster={min_cluster}"
        elif method == "recall_v1":
            def _cluster_recall(query: str, top_k: int) -> list[Any]:
                return self._vstash.search(
                    query,
                    top_k=top_k,
                    collection=self.collection,
                    layer="episodic",
                )
            clusters = cluster_by_recall(
                events,
                recall_fn=_cluster_recall,
                top_k=recall_top_k,
            )
            reason = f"clustered_recall_topk={recall_top_k}_mincluster={min_cluster}"
        elif method == "brief_v1":
            if synthesize_fn is None:
                raise ValueError(
                    "brief_v1 requires synthesize_fn -- pass an LLM callable"
                )
            # Fingerprint check: skip LLM if episodic set unchanged
            import hashlib as _hl
            all_paths = sorted(path for path, _ in events)
            events_fingerprint = _hl.sha1(
                ",".join(all_paths).encode("utf-8")
            ).hexdigest()[:16]

            existing_briefs = self._vstash.list(
                collection=self.collection,
                layer="semantic",
            )
            existing_brief_titles = {
                getattr(d, "title", "") for d in existing_briefs
            }
            fp_tag = f"events_fp:{events_fingerprint}"

            # Check if any existing brief has this fingerprint
            already_generated = any(
                fp_tag in (getattr(d, "tags", "") or "")
                for d in existing_briefs
            )
            if already_generated:
                return ConsolidationResult(
                    events_examined=len(events),
                    facts_written=0,
                    facts=[],
                    skipped=True,
                    reason=f"brief_v1_skipped_fingerprint={events_fingerprint}",
                    decider=decision.policy,
                    method=method,
                )

            # Tombstone any prior briefs that don't share the new
            # fingerprint. Probe B (2026-04-19) showed that without
            # supersession, recall-briefs surfaces stale snapshots
            # alongside fresh ones and downstream LLMs treat both
            # as current state. The full text + tags are preserved
            # in merken_tombstones so a caller can `merken tombstones`
            # to inspect the historical snapshot.
            stale_briefs = [
                d for d in existing_briefs
                if "method:brief_v1" in (getattr(d, "tags", "") or "")
                and fp_tag not in (getattr(d, "tags", "") or "")
            ]
            for stale in stale_briefs:
                self._supersede_brief(stale, reason=f"superseded_by_fp={events_fingerprint}")

            from merken.consolidation import generate_briefs
            briefs = generate_briefs(events, synthesize_fn)
            facts_written = []
            for i, brief in enumerate(briefs):
                # Each brief needs a unique fingerprint. Include brief
                # content hash since all briefs share the same derived_from.
                brief_hash = _hl.sha1(brief.encode("utf-8")).hexdigest()[:12]
                fact = Fact(
                    text=brief,
                    derived_from=all_paths,
                    cluster_size=len(events),
                    method="brief_v1",
                )
                self._vstash.remember(
                    fact.text,
                    title=f"brief_{brief_hash}",
                    collection=self.collection,
                    layer="semantic",
                    tags=f"method:brief_v1,{fp_tag}",
                )
                facts_written.append(fact)

            return ConsolidationResult(
                events_examined=len(events),
                facts_written=len(facts_written),
                facts=facts_written,
                skipped=False,
                reason=f"brief_v1_generated_fp={events_fingerprint}",
                decider=decision.policy,
                method=method,
            )
        else:
            raise ValueError(
                f"unknown consolidation method {method!r}; "
                f"expected 'embedding_v1', 'recall_v1', 'jaccard_v1', or 'brief_v1'"
            )

        facts_written = []
        for cluster in clusters:
            if len(cluster) < min_cluster:
                continue
            if materialize_fn is not None:
                # Full control over the fact: text AND provenance. A
                # materializer may expel cluster members it decides do
                # not belong (see ``merken.classifiers.jev.JevMaterializer``).
                fact = materialize_fn(cluster)
            elif synthesize_fn is not None:
                from merken.consolidation import materialize_fact_llm
                fact = materialize_fact_llm(cluster, synthesize_fn)
            else:
                fact = materialize_fact(cluster)
            fp = fact_fingerprint(fact)
            self._vstash.remember(
                fact.text,
                title=f"fact_{fp}",
                collection=self.collection,
                layer="semantic",
                tags=f"derived_from:{','.join(fact.derived_from)}",
            )
            facts_written.append(fact)

        return ConsolidationResult(
            events_examined=len(events),
            facts_written=len(facts_written),
            facts=facts_written,
            skipped=False,
            reason=reason,
            decider=decision.policy,
            method=method,
        )

    # --------------------------------------------------------------- forget

    def forget(self, *, force: bool = False) -> ForgetResult:
        """Tombstone episodic events whose content is preserved in facts.

        Walks the semantic layer to build a reverse map
        ``event_path → [fact_paths that cite it in derived_from]``,
        then asks the configured ``should_forget`` decider about
        each episodic event. When the decider says yes (or ``force``
        is set), the event is:

        1. Copied to ``merken_tombstones`` with full text + metadata
           + provenance to the facts that preserve it. This is the
           authoritative forgetting record — reversible via
           ``unforget`` (not implemented yet; future slice).
        2. Removed from the merken collection via ``vstash.remove``
           so it no longer surfaces in recall.

        Every decision (tombstone or skip) writes a ``should_forget``
        audit row. Forget operations are intentionally slow and
        loud on purpose — losing a memory is a big deal, even with
        the tombstone safety net.

        Parameters
        ----------
        force:
            Tombstone *every* episodic event regardless of the
            decider. Useful for ``mem.forget(force=True)`` as a
            "wipe the episodic layer" operation after a known-good
            consolidation pass. Still writes tombstones and audit
            rows for each — nothing is destroyed, only moved.
        """
        facts = self._vstash.list(
            collection=self.collection,
            layer="semantic",
        )
        derived_in: dict[str, list[str]] = {}
        for fact in facts:
            tags = fact.tags or ""
            if tags.startswith("derived_from:"):
                paths = tags[len("derived_from:"):].split(",")
                for p in paths:
                    p = p.strip()
                    if p:
                        derived_in.setdefault(p, []).append(fact.path)

        episodic = self._vstash.list(
            collection=self.collection,
            layer="episodic",
        )

        # Type-B (authoritative) events are excluded from forget via
        # the fail-closed `is_safely_mutable` predicate -- see
        # `merken/sourcing.py`. An event with an unrecognized
        # `source:<value>` tag is conservatively treated as immutable
        # so a clinical protocol cannot be tombstoned by accident.
        episodic = [
            e for e in episodic
            if is_safely_mutable(getattr(e, "tags", None))
        ]

        # Build supersession map: for each event, which newer events
        # claim to supersede it? An event B with tag
        # "supersedes:event_A_title" means B replaces A.
        superseded_by: dict[str, list[str]] = {}
        title_to_path: dict[str, str] = {}
        for event in episodic:
            if event.title:
                title_to_path[event.title] = event.path

        for event in episodic:
            if not event.tags:
                continue
            for tag in event.tags.split(","):
                tag = tag.strip()
                if tag.startswith("supersedes:"):
                    target_title = tag[len("supersedes:"):]
                    target_path = title_to_path.get(target_title)
                    if target_path and target_path != event.path:
                        superseded_by.setdefault(target_path, []).append(
                            event.title or event.path
                        )

        tombstoned: list[str] = []
        skipped: list[tuple[str, str]] = []

        for event in episodic:
            chunks = self._vstash.get_document_chunks(
                event.path,
                collection=self.collection,
            )
            full_text = " ".join(chunks).strip()

            event_derived = derived_in.get(event.path, [])
            ctx = ForgetContext(
                project=self.project,
                derived_in_facts=event_derived,
                superseded_by=superseded_by.get(event.path, []),
            )
            decision = self._forget_decider.decide(
                event.path,
                full_text,
                ctx,
            )

            self._write_forget_audit(event.path, decision, event_derived)

            if not (decision.tombstone or force):
                skipped.append((event.path, decision.reason))
                continue

            self._write_tombstone(
                event_path=event.path,
                event_text=full_text,
                event_title=event.title,
                event_layer=event.layer,
                event_tags=event.tags,
                derived_in_facts=event_derived,
                reason=decision.reason if decision.tombstone else "forced",
                policy=decision.policy if decision.tombstone else "force",
            )
            try:
                self._vstash.remove(event.path)
                tombstoned.append(event.path)
            except Exception:
                # If remove fails, the tombstone still exists — the
                # event is in both places until the next forget run.
                # Better than silently losing the tombstone.
                skipped.append((event.path, "vstash_remove_failed"))

        return ForgetResult(
            tombstoned=tombstoned,
            skipped=skipped,
            events_examined=len(episodic),
            decider=self._forget_decider.name,
        )

    def tombstones(
        self,
        query: str = "tombstone",
        *,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """Query the tombstone collection.

        Use this to find "what did I forget?" The tombstone
        collection is searched in isolation from normal memory and
        from the audit log.
        """
        return self._vstash.search(
            query,
            top_k=top_k,
            collection=TOMBSTONE_COLLECTION,
            layer=TOMBSTONE_LAYER,
        )

    # --------------------------------------------------------------- audit

    def audit(
        self,
        query: str = "should_remember",
        *,
        top_k: int = 20,
        fts_only: bool = False,
    ) -> list[SearchResult]:
        """Query the audit log.

        Use this to answer "why was X kept / dropped?" The audit collection
        is searched in isolation from normal memory. Pass ``fts_only=True``
        when the query is a literal token (e.g. ``shadow_disagree``) so
        vstash uses its full-text index directly instead of hybrid vector
        + FTS -- embedding tiny tag tokens degrades recall on small
        collections.
        """
        return self._vstash.search(
            query,
            top_k=top_k,
            collection=AUDIT_COLLECTION,
            layer=AUDIT_LAYER,
            retrieval_mode="fts_only" if fts_only else None,
        )

    # --------------------------------------------------------------- midloop

    def observe_step(
        self,
        observation: StepObservation,
        *,
        is_last_step: bool = False,
    ) -> MidloopDecision:
        """Submit one step observation to the midloop and (maybe) audit it.

        Per midloop spec section "Trayectoria, no snapshot":
          - The trajectory window is per-Memory-instance state, NOT
            persisted step-by-step. Reuse the same Memory instance
            across a session for trajectory coherence.
          - The full trajectory (last N observations) is passed to the
            decider so it can compute trends.
          - Only "interesting" steps land in the audit log. The
            ``is_persistable_step`` filter drops on_track-with-no-change
            rows so the audit collection does not flood. Last step
            before task outcome always persists -- pass
            ``is_last_step=True`` on the final observation of a task.

        Returns the ``MidloopDecision`` so the caller can act on
        ``decision.intervene`` / ``decision.action`` if it wants
        runtime behavior to follow the midloop's recommendation.
        """
        from merken.policies.midloop import (
            MidloopContext,
            is_persistable_step,
        )

        ctx = MidloopContext(
            project=self.project,
            task_id=observation.task_id,
            task_category=observation.task_category,
            trajectory_size=len(self._trajectory),
            task_description=observation.metadata.get("task_description", ""),
        )
        decision = self._midloop_decider.decide(
            observation=observation,
            ctx=ctx,
            trajectory=list(self._trajectory),
        )

        if is_persistable_step(decision, self._prev_midloop_decision, is_last_step):
            self._write_midloop_audit(observation, decision)

        self._trajectory.append(observation)
        self._prev_midloop_decision = decision
        return decision

    def reset_trajectory(self) -> None:
        """Clear the in-process trajectory window (typically on task switch).

        Call between tasks so signals from the previous task do not
        leak into the next task's decisions. The audit log still has
        the prior task's persisted rows.
        """
        self._trajectory.clear()
        self._prev_midloop_decision = None

    # --------------------------------------------------------------- labels

    def remember_label(self, *, event_title: str, body: str) -> None:
        """Write one oracular label into the ``merken_labels`` collection.

        Used by ``merken.labeling`` when an oracle classifies a
        shadow-mode disagreement. Stored in the project's own vstash so
        labels are always co-located with the audit rows they annotate.
        Failure is silent for the same reason audit writes are silent:
        label failures must not break normal memory ops.
        """
        from merken.labeling import LABEL_COLLECTION, LABEL_LAYER, _label_title

        try:
            self._vstash.remember(
                body,
                title=_label_title(event_title),
                collection=LABEL_COLLECTION,
                layer=LABEL_LAYER,
            )
        except Exception:
            pass

    def search_labels(self, query: str = "label:", *, top_k: int = 200):
        """Query the label store for previously-oracled disagreements."""
        from merken.labeling import LABEL_COLLECTION, LABEL_LAYER

        return self._vstash.search(
            query,
            top_k=top_k,
            collection=LABEL_COLLECTION,
            layer=LABEL_LAYER,
        )

    # --------------------------------------------------------------- internals

    def _write_audit(self, event: Event, decision: Decision) -> None:
        """Write one audit row, bypassing the loop.

        Audit failures must never break the user's call. We swallow the
        exception so a flaky audit write can't take down a good ``remember``.
        The cost is that audit gaps are silent — acceptable for Phase 1, to
        be revisited if it ever bites in practice.
        """
        title, body = format_audit_row(event, decision)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_consolidate_audit(
        self,
        decision: ConsolidationDecision,
        n_events: int,
    ) -> None:
        """Audit one ``should_consolidate`` decision. Same fail-open policy."""
        title, body = format_consolidate_audit_row(decision, n_events)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_midloop_audit(
        self,
        observation: StepObservation,
        decision: MidloopDecision,
    ) -> None:
        """Audit one ``should_intervene`` decision.

        Same fail-open policy as the other audit writers: a flaky
        write must never break the user's call. Uses the canonical
        format from ``merken.audit.format_midloop_audit_row``.
        """
        from merken.audit import format_midloop_audit_row
        title, body = format_midloop_audit_row(observation, decision)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _supersede_brief(self, doc, reason: str) -> None:
        """Tombstone a stale brief during a re-consolidation cycle.

        Called by ``consolidate(method="brief_v1")`` when a fresh
        events_fingerprint produces a new brief set; older briefs in
        the same semantic layer get superseded so ``recall-briefs``
        only surfaces the current snapshot. Full text and original
        tags are preserved in ``merken_tombstones`` for inspection
        via ``Memory.tombstones`` / ``merken tombstones`` CLI.

        Failure is silent: a brief that fails to supersede stays
        live in semantic, which is the same as the prior behavior.
        Worst case: stale briefs accumulate; recall surfaces them.
        Best case: clean snapshot.
        """
        try:
            chunks = self._vstash.get_document_chunks(
                doc.path, collection=self.collection,
            )
            text = " ".join(chunks).strip()
            if not text:
                return
            t_title, t_body = format_tombstone_row(
                event_path=doc.path,
                event_text=text,
                event_title=getattr(doc, "title", None),
                event_layer="semantic",
                event_tags=getattr(doc, "tags", None),
                derived_in_facts=[],
                reason=reason,
                policy="brief_v1_supersede",
            )
            self._vstash.remember(
                t_body, title=t_title,
                collection=TOMBSTONE_COLLECTION, layer=TOMBSTONE_LAYER,
            )
            self._vstash.remove(doc.path)
        except Exception:
            pass

    def _write_recall_audit(
        self, query: str, plan: RecallPlan, reranker: str | None = None
    ) -> None:
        """Audit one ``should_recall`` decision. Fail-open."""
        title, body = format_recall_audit_row(query, plan, reranker=reranker)
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_forget_audit(
        self,
        event_path: str,
        decision: ForgetDecision,
        derived_in_facts: list[str],
    ) -> None:
        """Audit one ``should_forget`` decision. Fail-open."""
        title, body = format_forget_audit_row(
            event_path, decision, derived_in_facts
        )
        try:
            self._vstash.remember(
                body,
                title=title,
                collection=AUDIT_COLLECTION,
                layer=AUDIT_LAYER,
            )
        except Exception:
            pass

    def _write_tombstone(
        self,
        *,
        event_path: str,
        event_text: str,
        event_title: str | None,
        event_layer: str | None,
        event_tags: str | None,
        derived_in_facts: list[str],
        reason: str,
        policy: str,
    ) -> None:
        """Persist the full-text tombstone so the event can be unforgotten.

        Unlike the audit row (which is a decision log), this stores
        everything needed to reconstruct the event: full text, title,
        layer, tags. Lives in ``merken_tombstones`` collection so a
        user can query "what did I forget?" without touching audit.

        NOT fail-open. If the tombstone write fails, we raise — we
        must not remove the event from the main collection without a
        tombstone to restore from.
        """
        title, body = format_tombstone_row(
            event_path=event_path,
            event_text=event_text,
            event_title=event_title,
            event_layer=event_layer,
            event_tags=event_tags,
            derived_in_facts=derived_in_facts,
            reason=reason,
            policy=policy,
        )
        self._vstash.remember(
            body,
            title=title,
            collection=TOMBSTONE_COLLECTION,
            layer=TOMBSTONE_LAYER,
        )

    # ------------------------------------------------------------------- misc

    def close(self) -> None:
        """Release the underlying vstash handle."""
        self._vstash.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
