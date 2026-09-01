"""Orientation (R1+): identify active regions from the gist layer. No LLM.

Before fine retrieval, a cheap pass over L1/L3 content (summary and
overview embeddings + FTS, keyword names) identifies the *active regions*
for the query: which documents, sections, and concepts are in play.

Contract (D12, confidence prior 55 — the weakest bet, designed to cap its
own downside):
- Regions BOOST fine-stage candidates (a bounded score bonus applied
  before/within reranking) and select which L2 notes deserve context.
- Regions NEVER filter the global channels: an orientation error must not
  blind the system. Ablation flag to disable entirely (evaluation.md Q3).

What exists, not what is configured
-----------------------------------
Orientation branches on the content actually in the store, never on the
KB's S level (docs/architecture.md). On an S0 KB there are no notes and no
keywords: the gist probe finds nothing, :func:`orient` returns empty
Regions *before* an embedder is resolved or the vector index is opened,
and :meth:`Regions.boost` degrades to the identity. Orientation costs a
handful of indexed SELECTs there, and the pipeline behaves exactly as R0.

Scales (docs/retrieval.md: one scoring scale per stage)
------------------------------------------------------
Region scores are internal, normalized to ``[0, 1]`` with the strongest
region at 1.0 — only their ratios matter. The boost is applied to
``Candidate.rrf_score``, the FIRST-stage scale, before reranking: the
reranker's absolute scores stay untouched because they drive the
no-answer gate.

Boost formula
-------------
``rrf_score * (1 + MAX_BOOST * region_score)``, with ``region_score`` the
clamped sum of the candidate's document and section region scores.
Multiplicative, because RRF scores have no meaningful unit — a fraction is
the only pool-independent bonus — and because they are strictly positive
sums of ``1/(RRF_K + rank)`` terms, so a bonus can never zero, negate or
drop anything. A candidate in no active region keeps its score exactly.
``MAX_BOOST`` is the cap that prices a wrong orientation: at 0.25 a
boosted candidate needs to be within 20% of a rival's score to overtake
it, so orientation can re-order near-ties (where it is most likely to
help) and cannot rewrite the ranking (where it would do the damage D12
worries about).

Signals (all cheap, no LLM, no network beyond the local encoder)
---------------------------------------------------------------
- Lexical: one FTS query over notes; gist notes (kind ``summary`` /
  ``overview``) contribute ``1/rank`` to the document they belong to.
  A steep weight, not RRF's flattened one: orientation needs contrast
  between regions, not a fusion-safe tie.
- Dense: one vector search restricted to note owners, scored by cosine.
  Skipped entirely when the KB has no gist notes or no vector index.
- Sections: for documents that scored, section titles overlapping the
  query tokens (``get_sections``) — the only section-level gist there is,
  since notes are document- or keyword-scoped.
- Keywords: L2/L3 concept names occurring verbatim in the query.

The overview note has no document, so it raises no region; it competes for
rank in the two note channels, which is exactly right — a query the atlas
answers is a query about the KB at large, and orientation should not
commit to a document for it.

API
---
- ``MAX_BOOST``: the cap, as a fraction of a candidate's rrf_score.
- ``class Regions``: ``doc_ids``, ``section_ids``, ``keyword_ids`` (all
  ``-> score in [0, 1]``), ``is_empty()``, ``boost(candidates)``.
- ``orient(kb, query, *, embedder=None, enabled=True) -> Regions``;
  ``enabled=False`` is the ablation (evaluation.md Q3): empty Regions, no
  store access, no embedder resolution.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from wikirag.config import KnowledgeBase
from wikirag.ingestion import EMBEDDING_MODEL_KEY
from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL, EmbedderLike, get_embedder
from wikirag.retrieval.search import Candidate
from wikirag.storage.vectors import INDEX_FILENAME, VectorIndex

#: Largest fraction by which orientation may raise a first-stage score. The
#: single number that bounds the damage of a wrong orientation (D12).
MAX_BOOST = 0.25

#: How many gist hits each channel contributes. Small on purpose: orientation
#: is a map, and a map naming every document names nothing.
_CHANNEL_LIMIT = 10

#: Section titles are short; single characters and one-letter numerals carry no
#: topical signal and would make every section of an active document match.
_MIN_TOKEN_LEN = 2

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LEN}


@dataclass
class Regions:
    """The active regions for one query: documents, sections, concepts.

    Scores are normalized per family to ``[0, 1]``, strongest region at 1.0.
    They are an internal scale — only ratios are meaningful — and they never
    express exclusion: a region absent from these maps is not suppressed, it
    is merely not promoted.
    """

    doc_ids: dict[str, float] = field(default_factory=dict)
    section_ids: dict[str, float] = field(default_factory=dict)
    keyword_ids: dict[int, float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """True when orientation found nothing to point at (the S0 case)."""
        return not (self.doc_ids or self.section_ids or self.keyword_ids)

    def region_score(self, candidate: Candidate) -> float:
        """How active this candidate's location is, in ``[0, 1]``.

        Document and section evidence add up (a hit in an active section of an
        active document is the strongest possible location) and saturate at 1.0,
        so the cap below stays a cap.
        """
        score = self.doc_ids.get(candidate.doc_id or "", 0.0)
        score += self.section_ids.get(candidate.section_id or "", 0.0)
        return min(1.0, score)

    def boost(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Re-score ``candidates`` by region activity; never filter, never drop.

        Returns fresh Candidate objects (the inputs keep their original
        scores) with ``rrf_score * (1 + MAX_BOOST * region_score)``, sorted by
        the adjusted score descending, ties broken by owner for determinism.
        Every input candidate appears in the output exactly once. Empty
        Regions are the identity: the same objects, in the same order.

        Keyword regions do not take part — a candidate carries no keyword, and
        resolving one would mean a store round-trip per candidate. They exist
        for assembly's "which L2 notes deserve context" decision.
        """
        if self.is_empty():
            return list(candidates)
        boosted = [
            replace(c, rrf_score=c.rrf_score * (1.0 + MAX_BOOST * self.region_score(c)))
            for c in candidates
        ]
        boosted.sort(key=lambda c: (-c.rrf_score, c.owner_kind, c.owner_id))
        return boosted


def orient(
    kb: KnowledgeBase,
    query: str,
    *,
    embedder: EmbedderLike | None = None,
    enabled: bool = True,
) -> Regions:
    """Locate the query's active regions in the gist layer. No LLM call.

    ``enabled=False`` (the evaluation.md Q3 ablation) and an empty query
    return empty Regions without touching the store. A KB whose gist layer is
    empty — every S0 KB — returns empty Regions after a few indexed SELECTs,
    without resolving an embedder or opening the vector index.

    ``embedder`` overrides the KB's configured encoder (tests, alternative
    spaces); it is used only if the KB has gist notes and a vector index.
    """
    if not enabled or not query.strip():
        return Regions()

    gist_notes = _gist_notes(kb)
    keyword_ids = _keyword_regions(kb, query)
    if not gist_notes:
        return Regions(keyword_ids=keyword_ids)

    evidence: dict[str, float] = {}  # doc_id -> raw, un-normalized evidence
    for note_id, weight in _lexical_gist_hits(kb, query, gist_notes).items():
        _record(evidence, gist_notes[note_id], weight)
    for note_id, weight in _dense_gist_hits(kb, query, gist_notes, embedder).items():
        _record(evidence, gist_notes[note_id], weight)

    doc_ids = _normalized(evidence)
    return Regions(
        doc_ids=doc_ids,
        section_ids=_section_regions(kb, query, doc_ids),
        keyword_ids=keyword_ids,
    )


# --------------------------------------------------------------- gist probe


def _gist_notes(kb: KnowledgeBase) -> dict[int, str | None]:
    """Every gist note in the store: note_id -> its doc_id (None = overview).

    The gist layer is the ``overview`` note (L3) plus the per-document
    ``summary`` notes (L1). Keyword notes (L2) are deliberately excluded: they
    are content to retrieve, not a map to orient by, and their concepts reach
    orientation through keyword names instead.

    The store has no "list notes by kind" API, so the gist is enumerated by
    subject: the single overview, plus one summary per document. That is one
    indexed lookup per document — negligible at personal-KB scale, and it is
    what lets an S0 KB (no notes at all) skip both note channels, the embedder
    and the vector index entirely.
    """
    notes: dict[int, str | None] = {}
    overview = kb.store.get_note(kind="overview")
    if overview is not None:
        notes[int(overview["id"])] = overview["doc_id"]
    for document in kb.store.list_documents():
        summary = kb.store.get_note(kind="summary", doc_id=document["id"])
        if summary is not None:
            notes[int(summary["id"])] = summary["doc_id"]
    return notes


# ------------------------------------------------------------- gist channels


def _lexical_gist_hits(
    kb: KnowledgeBase, query: str, gist_notes: dict[int, str | None]
) -> dict[int, float]:
    """BM25 over notes, weighted ``1/rank``; non-gist notes are skipped."""
    hits = kb.store.fts_search(query, limit=_CHANNEL_LIMIT, kinds=("note",))
    weights: dict[int, float] = {}
    for rank, hit in enumerate(hits, start=1):
        note_id = int(hit["owner_id"])
        if note_id in gist_notes:
            weights[note_id] = 1.0 / rank
    return weights


def _dense_gist_hits(
    kb: KnowledgeBase,
    query: str,
    gist_notes: dict[int, str | None],
    embedder: EmbedderLike | None,
) -> dict[int, float]:
    """Cosine similarity of the query to gist-note embeddings, or nothing.

    Negative cosines are evidence of the opposite and are dropped rather than
    subtracted: regions only ever promote.
    """
    if not (kb.vectors_dir / INDEX_FILENAME).is_file():
        return {}
    if embedder is None:
        # Reading a real Embedder's dim loads the model, so this happens only
        # once a gist and an index are both known to exist (as in search.py).
        model = kb.store.config_get(EMBEDDING_MODEL_KEY, DEFAULT_EMBEDDING_MODEL)
        embedder = get_embedder(model)
    index = VectorIndex.open(kb.vectors_dir, embedder.model_name, embedder.dim)
    if not len(index):
        return {}
    neighbors = index.search(
        embedder.encode_query(query), k=_CHANNEL_LIMIT, filter_kind="note"
    )
    weights: dict[int, float] = {}
    for owner, score in neighbors:
        note_id = int(owner.id)
        if note_id in gist_notes and score > 0.0:
            weights[note_id] = float(score)
    return weights


# -------------------------------------------------------------- region maps


def _record(evidence: dict[str, float], doc_id: str | None, weight: float) -> None:
    """Accumulate a note's evidence on its document; the overview has none.

    A document keeps its best note's evidence rather than the sum, so a
    document that happens to own more gist notes cannot out-shout a better
    match.
    """
    if doc_id is None:
        return
    evidence[doc_id] = max(evidence.get(doc_id, 0.0), weight)


def _normalized(evidence: dict[str, float]) -> dict[str, float]:
    """Scale raw evidence into [0, 1] with the strongest region at 1.0."""
    top = max(evidence.values(), default=0.0)
    if top <= 0.0:
        return {}
    return {key: value / top for key, value in evidence.items()}


def _section_regions(
    kb: KnowledgeBase, query: str, doc_ids: dict[str, float]
) -> dict[str, float]:
    """Sections of active documents whose titles overlap the query.

    Scored as the document's own score times the fraction of the title's
    tokens the query carries, so a section can only ever be as active as the
    document containing it.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return {}
    sections: dict[str, float] = {}
    for doc_id, doc_score in doc_ids.items():
        for section in kb.store.get_sections(doc_id):
            title_tokens = _tokens(section["title"] or "")
            if not title_tokens:
                continue
            overlap = len(title_tokens & query_tokens) / len(title_tokens)
            if overlap > 0.0:
                sections[str(section["id"])] = doc_score * overlap
    return sections


def _keyword_regions(kb: KnowledgeBase, query: str) -> dict[int, float]:
    """Concepts whose name occurs verbatim in the query.

    Whole-token matching on the canonical name: substring matching would make
    every short keyword active on every query, which is how orientation turns
    from a map into noise. Multi-word names are matched as a phrase, so they
    are strictly more specific than the single words they contain — all
    matches are equally certain, hence all score 1.0.
    """
    haystack = " ".join(_TOKEN_RE.findall(query.lower()))
    if not haystack:
        return {}
    matched: dict[int, float] = {}
    for keyword in kb.store.list_keywords():
        name = " ".join(_TOKEN_RE.findall(str(keyword["canonical_name"]).lower()))
        if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", haystack):
            matched[int(keyword["id"])] = 1.0
    return matched
