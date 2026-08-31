"""First-stage hybrid search: BM25 + dense, fused by RRF.

- Lexical: FTS5 over chunks and notes (BM25 ranking built in).
- Dense: VectorIndex search over the same content, one embedding space.
- Fusion: Reciprocal Rank Fusion — rank-based, so no score-scale
  calibration between channels (v1's min-max normalizations and
  round-robin interleaving disappear). Multi-anchor queries (R2
  sub-questions) fuse the same way: RRF across all anchor rankings.

Returns candidates as (owner_kind, owner_id, fused_rank/score) with
enough provenance for assembly to fetch text and cite sources.

Scales (docs/retrieval.md: one scoring scale per stage)
------------------------------------------------------
``rrf_score`` is this stage's only scale: a sum of ``1/(RRF_K + rank)``
terms, comparable within one result list and meaningless outside it.
``Candidate.score`` stays None here; rerank.py fills it with absolute
cross-encoder scores, the scale inclusion and the no-answer gate use.

Robustness
----------
A channel that has nothing to say contributes nothing rather than
failing: a KB with no vector index yet (or an empty one) is searched
lexically, and an FTS query with no hits leaves the dense ranking alone.
Owners in the vector index that no longer exist in the store (a stale
index) are dropped instead of raising.

API
---
- ``RRF_K``: the RRF constant (60).
- ``class Candidate``: owner_kind, owner_id, text, doc_id, section_id,
  rrf_score, score=None.
- ``hybrid_search(kb, queries, k=50, *, embedder=None) -> [Candidate]``,
  sorted by rrf_score descending, at most k, deduplicated by
  (owner_kind, owner_id).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from wikirag.config import KnowledgeBase
from wikirag.ingestion import EMBEDDING_MODEL_KEY
from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL, EmbedderLike, get_embedder
from wikirag.storage.vectors import INDEX_FILENAME, VectorIndex

#: RRF constant. 60 is the value from the original TREC formulation and the
#: de-facto default; it flattens the head of each ranking so no single channel
#: can dominate the fusion.
RRF_K = 60

#: Each channel is asked for more than k so that fusion has something to fuse:
#: a candidate ranked mid-pool by both channels can overtake a channel's top
#: hit, which is the whole point of RRF.
_POOL_FACTOR = 2

_OwnerKey = tuple[str, str]


@dataclass
class Candidate:
    """One retrieval hit, before reranking."""

    owner_kind: str  # "chunk" | "note"
    owner_id: str
    text: str
    doc_id: str | None
    section_id: str | None
    rrf_score: float
    score: float | None = None  # absolute reranker score; set by rerank.py


def hybrid_search(
    kb: KnowledgeBase,
    queries: Sequence[str],
    k: int = 50,
    *,
    embedder: EmbedderLike | None = None,
) -> list[Candidate]:
    """Fuse BM25 and dense rankings of every query into one candidate list.

    Each query contributes two rankings (lexical, dense); a candidate's
    rrf_score is the sum of ``1/(RRF_K + rank)`` over every ranking it appears
    in, rank starting at 1. The result is sorted by rrf_score descending, cut
    to ``k``, and holds one entry per (owner_kind, owner_id).

    ``embedder`` overrides the KB's configured encoder (tests, alternative
    spaces). When it is omitted the shared encoder is resolved lazily — a KB
    without a vector index never loads a model.
    """
    wanted = [query for query in queries if query and query.strip()]
    if k <= 0 or not wanted:
        return []
    pool = k * _POOL_FACTOR

    fused: dict[_OwnerKey, float] = {}
    provenance: dict[_OwnerKey, dict[str, Any]] = {}
    index, resolved = _open_index(kb, embedder)

    for query in wanted:
        hits = kb.store.fts_search(query, limit=pool)
        for hit in hits:
            provenance.setdefault((hit["owner_kind"], hit["owner_id"]), hit)
        _fuse(fused, [(hit["owner_kind"], hit["owner_id"]) for hit in hits])
        if index is not None and resolved is not None:
            neighbors = index.search(resolved.encode_query(query), k=pool)
            _fuse(fused, [(owner.kind, owner.id) for owner, _ in neighbors])

    _hydrate(kb, fused, provenance)
    candidates = [
        Candidate(
            owner_kind=kind,
            owner_id=owner_id,
            text=str(provenance[(kind, owner_id)]["text"]),
            doc_id=provenance[(kind, owner_id)]["doc_id"],
            section_id=provenance[(kind, owner_id)]["section_id"],
            rrf_score=score,
        )
        for (kind, owner_id), score in fused.items()
        if (kind, owner_id) in provenance
    ]
    candidates.sort(key=lambda c: (-c.rrf_score, c.owner_kind, c.owner_id))
    return candidates[:k]


def _fuse(fused: dict[_OwnerKey, float], ranking: Sequence[_OwnerKey]) -> None:
    """Add one ranking's RRF contribution; ranks start at 1."""
    for rank, key in enumerate(ranking, start=1):
        fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)


def _open_index(
    kb: KnowledgeBase, embedder: EmbedderLike | None
) -> tuple[VectorIndex | None, EmbedderLike | None]:
    """The KB's vector index and the encoder for it, or (None, None).

    The index file is checked before the encoder is resolved: reading a real
    Embedder's ``dim`` loads the model, and a KB that has not been indexed yet
    must not pay for it. The index is opened under the encoder's own model
    name, so vectors from another embedding space are discarded by
    VectorIndex.open rather than compared against incomparable queries.
    """
    if not (kb.vectors_dir / INDEX_FILENAME).is_file():
        return None, None
    if embedder is None:
        model = kb.store.config_get(EMBEDDING_MODEL_KEY, DEFAULT_EMBEDDING_MODEL)
        embedder = get_embedder(model)
    index = VectorIndex.open(kb.vectors_dir, embedder.model_name, embedder.dim)
    return (index, embedder) if len(index) else (None, None)


def _hydrate(
    kb: KnowledgeBase,
    fused: dict[_OwnerKey, float],
    provenance: dict[_OwnerKey, dict[str, Any]],
) -> None:
    """Fetch text and provenance for dense-only hits; FTS rows already carry it.

    Owners the store no longer knows keep no provenance entry and are dropped
    by the caller: a vector index can outlive the rows it points at.
    """
    chunk_ids = [
        owner_id
        for (kind, owner_id) in fused
        if kind == "chunk" and (kind, owner_id) not in provenance
    ]
    for row in kb.store.get_chunks(chunk_ids):
        provenance[("chunk", str(row["id"]))] = {
            "text": row["text"],
            "doc_id": row["doc_id"],
            "section_id": row["section_id"],
        }
    for kind, owner_id in fused:
        if kind != "note" or (kind, owner_id) in provenance:
            continue
        note = kb.store.get_note(int(owner_id))
        if note is not None:
            provenance[(kind, owner_id)] = {
                "text": note["text"],
                "doc_id": note["doc_id"],
                "section_id": None,
            }
