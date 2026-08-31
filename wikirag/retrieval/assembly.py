"""Context assembly: one merged candidate list -> context under budget.

- Single pool (chunks, notes, summaries — whatever scored), one scale.
- Retrieve small, read big (D12): included chunks are expanded to their
  section window (structural neighbors from the store), deduplicated by
  chunk ID so expansion never double-counts budget.
- Greedy fill under token budget with per-document and per-keyword
  diversity caps (v1's MAX_CHUNKS_PER_KEYWORD insight, generalized).
- Dedup is structural: a chunk exists once, referenced by ID
  (v1's text-fingerprint dedup has nothing left to do).
- Every included entry keeps provenance (doc, section, chunk IDs) so the
  answer can cite.
- ONE token estimator shared package-wide (``ingestion.chunking.estimate_tokens``)
  — v1 had three that disagreed.
- Overview primer (if S3) is prepended outside the budget, its cost
  reserved like the answer allowance. **Not built here**: the primer slot
  arrives with Wave 4 (``synthesis/atlas.py``), when there is an overview
  note to prepend. No dead code for it exists yet.

Input
-----
``scored_candidates`` is rerank's output: candidates sorted by ``.score``
descending, with ``.score`` set. Only attributes are used (duck-typed over
``retrieval.search.Candidate``): ``owner_kind``, ``owner_id``, ``text``,
``doc_id``, ``section_id``, ``rrf_score``, ``score``.

Entry format (frozen — ``answer.py``'s prompts rely on it)
----------------------------------------------------------
A header line followed by the entry text; entries joined by a blank line::

    [1] Doc title — Section title
    ...text...

    [2] Note: keyword
    ...text...

The ``— Section title`` half is omitted when the section is unknown or
untitled; a document falls back to its filename, then to its ID. A note's
name is its keyword, else its document's title, else its owner ID.

Fill policy
-----------
Greedy in the given (score-descending) order, and **skip, never stop**: a
candidate that does not fit is dropped and the scan continues, so smaller
later entries still get in. Budget is checked against the *assembled* text
(header lines and blank-line separators included), so ``token_count`` is
always ``<= budget_tokens``.

Fallback ladder for a chunk candidate (window expansion respects the budget):

1. the full section window, minus chunks already included in an earlier
   entry (an entry whose every chunk is already included is skipped). What
   survives is one *contiguous* run around the hit: a used middle splits a
   window, and gluing its prefix to its suffix would present non-adjacent
   text as continuous;
2. a partial window centred on the hit chunk — the widest symmetric radius
   that fits;
3. the hit chunk alone (radius 0);
4. nothing: the candidate is skipped and the scan continues.

Diversity: at most ``max_entries_per_doc`` entries per ``doc_id`` (default
4), counting notes that belong to a document. Candidates without a
``doc_id`` are uncapped. The per-keyword cap of v1 is subsumed: entries are
capped by their document, which is the unit a reader can over-weight here.

Public API
----------
- ``ContextEntry`` — one numbered, cited entry with full provenance
- ``Context`` — ``.text``, ``.entries``, ``.token_count``
- ``DEFAULT_MAX_ENTRIES_PER_DOC``
- ``assemble(kb, scored_candidates, budget_tokens, *, max_entries_per_doc=4)
  -> Context``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from wikirag.config import KnowledgeBase
from wikirag.ingestion.chunking import estimate_tokens

if TYPE_CHECKING:  # the contract is duck-typed; search.py is a parallel module
    from wikirag.retrieval.search import Candidate

__all__ = [
    "DEFAULT_MAX_ENTRIES_PER_DOC",
    "Context",
    "ContextEntry",
    "assemble",
]

DEFAULT_MAX_ENTRIES_PER_DOC = 4

ENTRY_SEPARATOR = "\n\n"


@dataclass
class ContextEntry:
    """One numbered context entry, with everything an answer needs to cite."""

    ref: int
    owner_kind: str
    owner_id: str
    doc_id: str | None
    section_id: str | None
    section_title: str | None
    doc_title: str | None
    chunk_ids: list[str]
    score: float
    text: str


@dataclass
class Context:
    """The assembled context: formatted text plus per-entry provenance."""

    text: str
    entries: list[ContextEntry] = field(default_factory=list)
    token_count: int = 0


class _Titles:
    """Per-call cache of document, section and note titles (one KB read each)."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._store = kb.store
        self._docs: dict[str, dict[str, Any] | None] = {}
        self._sections: dict[str, dict[str, str | None]] = {}

    def doc_title(self, doc_id: str | None) -> str | None:
        if doc_id is None:
            return None
        if doc_id not in self._docs:
            self._docs[doc_id] = self._store.get_document(doc_id)
        row = self._docs[doc_id]
        if row is None:
            return doc_id
        return row.get("title") or row.get("filename") or doc_id

    def section_title(self, doc_id: str | None, section_id: str | None) -> str | None:
        if doc_id is None or section_id is None:
            return None
        if doc_id not in self._sections:
            self._sections[doc_id] = {
                s["id"]: s.get("title") for s in self._store.get_sections(doc_id)
            }
        return self._sections[doc_id].get(section_id)

    def note_name(self, owner_id: str) -> str:
        """A note's display name: keyword, else document title, else its ID."""
        try:
            note = self._store.get_note(int(owner_id))
        except (TypeError, ValueError):
            return owner_id
        if note is None:
            return owner_id
        if note.get("keyword_id") is not None:
            keyword = self._store.get_keyword(keyword_id=int(note["keyword_id"]))
            if keyword is not None:
                return str(keyword["name"])
        if note.get("doc_id") is not None:
            return self.doc_title(str(note["doc_id"])) or owner_id
        return owner_id


def _header(ref: int, owner_kind: str, doc_title: str | None, section_title: str | None) -> str:
    if owner_kind == "note":
        return f"[{ref}] Note: {doc_title}"
    if section_title:
        return f"[{ref}] {doc_title} — {section_title}"
    return f"[{ref}] {doc_title}"


def _format(header: str, text: str) -> str:
    return f"{header}\n{text}"


def _fits(parts: Sequence[str], candidate_part: str, budget_tokens: int) -> bool:
    """Budget is checked on the assembled text, separators and headers included."""
    return (
        estimate_tokens(ENTRY_SEPARATOR.join([*parts, candidate_part])) <= budget_tokens
    )


def _score_of(candidate: Candidate) -> float:
    """Reranker score; RRF is the fallback for a pool that skipped rerank."""
    score = getattr(candidate, "score", None)
    return float(score) if score is not None else float(candidate.rrf_score)


def assemble(
    kb: KnowledgeBase,
    scored_candidates: Sequence[Candidate],
    budget_tokens: int,
    *,
    max_entries_per_doc: int = DEFAULT_MAX_ENTRIES_PER_DOC,
) -> Context:
    """Turn a scored candidate pool into cited context under a token budget.

    Candidates are consumed in the order given (rerank's score order). See the
    module docstring for the fill policy, the window fallback ladder and the
    frozen entry format.
    """
    if not scored_candidates or budget_tokens <= 0:
        return Context(text="", entries=[], token_count=0)

    store = kb.store
    titles = _Titles(kb)

    parts: list[str] = []
    entries: list[ContextEntry] = []
    used_chunks: set[str] = set()
    per_doc: dict[str, int] = {}

    for candidate in scored_candidates:
        doc_id = candidate.doc_id
        if doc_id is not None and per_doc.get(doc_id, 0) >= max_entries_per_doc:
            continue

        ref = len(entries) + 1
        if candidate.owner_kind == "note":
            built = _build_note_entry(candidate, ref, titles, parts, budget_tokens)
        else:
            built = _build_chunk_entry(
                candidate, ref, titles, store, used_chunks, parts, budget_tokens
            )
        if built is None:
            continue

        entry, part = built
        entries.append(entry)
        parts.append(part)
        used_chunks.update(entry.chunk_ids)
        if doc_id is not None:
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1

    text = ENTRY_SEPARATOR.join(parts)
    return Context(text=text, entries=entries, token_count=estimate_tokens(text))


def _build_note_entry(
    candidate: Candidate,
    ref: int,
    titles: _Titles,
    parts: Sequence[str],
    budget_tokens: int,
) -> tuple[ContextEntry, str] | None:
    """Notes go in as-is: they are already the consolidated unit, no expansion."""
    name = titles.note_name(candidate.owner_id)
    part = _format(_header(ref, "note", name, None), candidate.text)
    if not _fits(parts, part, budget_tokens):
        return None
    entry = ContextEntry(
        ref=ref,
        owner_kind="note",
        owner_id=candidate.owner_id,
        doc_id=candidate.doc_id,
        section_id=None,
        section_title=None,
        doc_title=titles.doc_title(candidate.doc_id),
        chunk_ids=[],
        score=_score_of(candidate),
        text=candidate.text,
    )
    return entry, part


def _build_chunk_entry(
    candidate: Candidate,
    ref: int,
    titles: _Titles,
    store: Any,
    used_chunks: set[str],
    parts: Sequence[str],
    budget_tokens: int,
) -> tuple[ContextEntry, str] | None:
    window = store.section_window(candidate.owner_id)
    if not window:
        window = store.get_chunks([candidate.owner_id])
    available = _available_run(window, used_chunks, candidate.owner_id)
    if not available:
        return None  # every chunk of this window is already in an earlier entry

    centre = _centre_index(
        available, candidate.owner_id, _hit_ordinal(window, candidate.owner_id)
    )
    doc_id = candidate.doc_id or available[centre]["doc_id"]
    section_id = candidate.section_id or available[centre]["section_id"]
    doc_title = titles.doc_title(doc_id)
    section_title = titles.section_title(doc_id, section_id)
    header = _header(ref, "chunk", doc_title, section_title)

    # Fallback ladder: widest symmetric radius around the hit that fits, down
    # to the hit chunk alone; nothing fits -> skip this candidate.
    max_radius = max(centre, len(available) - 1 - centre)
    for radius in range(max_radius, -1, -1):
        slice_ = available[max(0, centre - radius) : centre + radius + 1]
        text = "\n\n".join(c["text"] for c in slice_)
        part = _format(header, text)
        if _fits(parts, part, budget_tokens):
            entry = ContextEntry(
                ref=ref,
                owner_kind="chunk",
                owner_id=candidate.owner_id,
                doc_id=doc_id,
                section_id=section_id,
                section_title=section_title,
                doc_title=doc_title,
                chunk_ids=[c["id"] for c in slice_],
                score=_score_of(candidate),
                text=text,
            )
            return entry, part
    return None


def _available_run(
    window: Sequence[dict[str, Any]],
    used_chunks: set[str],
    hit_id: str,
) -> list[dict[str, Any]]:
    """The contiguous stretch of not-yet-included chunks around the hit.

    Earlier entries can carve a used middle out of a window, leaving a prefix
    and a suffix. Joining those into one entry would present non-adjacent text
    as continuous, so only one contiguous run is kept: the one holding the hit,
    else the run nearest it by ordinal (the hit itself may already be included).
    """
    runs: list[list[dict[str, Any]]] = []
    for chunk in window:
        if chunk["id"] in used_chunks:
            runs.append([])
        elif runs and runs[-1]:
            runs[-1].append(chunk)
        else:
            runs.append([chunk])
    runs = [run for run in runs if run]
    if not runs:
        return []
    for run in runs:
        if any(c["id"] == hit_id for c in run):
            return run
    hit_ordinal = _hit_ordinal(window, hit_id)
    return min(runs, key=lambda run: min(abs(c["ordinal"] - hit_ordinal) for c in run))


def _hit_ordinal(window: Sequence[dict[str, Any]], hit_id: str) -> int:
    return next(
        (c["ordinal"] for c in window if c["id"] == hit_id), window[0]["ordinal"]
    )


def _centre_index(
    available: Sequence[dict[str, Any]], hit_id: str, hit_ordinal: int
) -> int:
    """Where the hit sits in its run; its nearest surviving chunk otherwise."""
    for i, chunk in enumerate(available):
        if chunk["id"] == hit_id:
            return i
    return min(
        range(len(available)),
        key=lambda i: (abs(available[i]["ordinal"] - hit_ordinal), i),
    )
