"""Deterministic, structure-aware chunker. Local, no LLM.

Contract (docs/ingestion.md):
- Input: an ExtractedDoc's sections. Output: chunks with stable IDs
  derived from (doc_id, structural path, ordinal).
- Target size ~400-600 tokens (embedder.max_seq_tokens minus margin is
  the binding constraint — v1 silently truncated embeddings by ignoring
  it). Split on section, then paragraph boundaries; small tail-merge to
  avoid fragment chunks; modest overlap only where a split severs a
  paragraph mid-argument.
- Re-running on the same file yields byte-identical chunks and IDs.

Chunk *enrichment* (contextual headers etc.) is NOT here — that's the
enrich stage, which may rewrite what gets embedded but never what the
canonical chunk text is.

Identity
--------
- ``section_id = f"{doc_id}#{section.path}"`` (path is dotted 1-based:
  "3", "3.2").
- ``chunk_id = f"{doc_id}#{section.path}#{n}"``, n 0-based within the
  section. The ``ordinal`` column is instead the chunk's *global*
  position in the document, because the store orders by it.

Sizing rules
------------
- ``max_tokens`` is a hard bound: no returned chunk exceeds it under
  ``token_counter``. The margin against the embedder's real limit is the
  caller's business; passing ~600 gives the ~400-600 target band.
- ``min_tokens = max_tokens // MIN_CHUNK_DIVISOR`` (8, i.e. 75 tokens at
  600) is the fragment threshold. After greedy packing, any chunk below
  it is merged with a neighbour. A chunk may stay below it only when the
  whole section is that small, or when merging with either neighbour
  would break the hard bound (a lone near-max paragraph next to a short
  one) — the bound wins over the threshold.
- A paragraph is split on sentences only when it alone exceeds
  ``max_tokens``; a sentence that alone exceeds it is split on words, and
  a single word that still exceeds it on characters. The last two are
  degenerate-input guards, not normal behaviour.
- Overlap exists only across a severed paragraph: the piece after a
  sentence-level split repeats the previous piece's last sentence, and
  only when that sentence fits ``max_tokens // OVERLAP_DIVISOR``. Overlap
  text is part of the chunk text (and of its token_count) but is never
  part of the canonical content: dropping every repeated leading sentence
  reconstructs the section.

Whitespace is normalised per paragraph (runs collapse to one space), so
PDF line-wrapping artifacts do not leak into chunk text or token counts.

Token estimator
---------------
``estimate_tokens`` is THE package-wide estimator (docs/retrieval.md:
one estimator, shared). ``max(ceil(words * 4/3), ceil(chars / 4))``:
both terms are standard English-prose approximations of BPE behaviour,
and the max makes it conservative for the two failure directions (long
compound words, and short-token-heavy text like tables or code).

API
---
- ``estimate_tokens(text) -> int``
- ``section_id(doc_id, path) -> str`` / ``chunk_id(doc_id, path, n) -> str``
- ``section_rows(doc_id, doc) -> list[dict]`` — rows for Store.add_sections
- ``chunk_document(doc_id, doc, *, max_tokens, token_counter=estimate_tokens)
  -> list[dict]`` — rows for Store.add_chunks
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MIN_CHUNK_DIVISOR",
    "OVERLAP_DIVISOR",
    "ExtractedDocLike",
    "SectionLike",
    "chunk_document",
    "chunk_id",
    "estimate_tokens",
    "section_id",
    "section_rows",
]

DEFAULT_MAX_TOKENS = 600
MIN_CHUNK_DIVISOR = 8
OVERLAP_DIVISOR = 10

TokenCounter = Callable[[str], int]


class SectionLike(Protocol):
    """Structural shape of extraction's Section (duck-typed on purpose)."""

    path: str
    title: str
    level: int
    text: str
    page_start: int
    page_end: int


class ExtractedDocLike(Protocol):
    title: str | None
    page_count: int
    sections: Sequence[SectionLike]


_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n\s*")
_WHITESPACE = re.compile(r"\s+")
# Sentence end: terminator, optional closing quote/bracket, then whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"'’”)\]]*\s+")


def estimate_tokens(text: str) -> int:
    """Package-wide token estimate. Deterministic, stdlib, conservative."""
    if not text:
        return 0
    words = len(text.split())
    return max((words * 4 + 2) // 3, (len(text) + 3) // 4)


def section_id(doc_id: str, path: str) -> str:
    return f"{doc_id}#{path}"


def chunk_id(doc_id: str, path: str, n: int) -> str:
    return f"{doc_id}#{path}#{n}"


# --------------------------------------------------------------------- units


@dataclass(frozen=True)
class _Unit:
    """One indivisible piece of a section: a paragraph or a piece of one."""

    text: str  # canonical content, appears in exactly one chunk
    overlap: str  # repeated context, "" unless a paragraph was severed
    src: int  # index of the unit `overlap` was taken from, -1 if none
    joins_prev: bool  # True when this unit continues the previous paragraph


def _paragraphs(text: str) -> list[str]:
    out: list[str] = []
    for raw in _PARAGRAPH_SPLIT.split(text):
        para = _WHITESPACE.sub(" ", raw).strip()
        if para:
            out.append(para)
    return out


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text)) if s]


def _split_by_chars(text: str, limit: int, counter: TokenCounter) -> list[str]:
    """Last-resort split of a single token-monster word."""
    step = max(1, limit * 4)
    pieces = [text[i : i + step] for i in range(0, len(text), step)]
    while any(counter(p) > limit for p in pieces) and step > 1:
        step = max(1, step // 2)
        pieces = [text[i : i + step] for i in range(0, len(text), step)]
    return pieces


def _pack_strings(
    parts: Sequence[str], limit: int, counter: TokenCounter, sep: str = " "
) -> list[str]:
    """Greedily join parts under `limit`; oversized parts stay alone."""
    out: list[str] = []
    cur = ""
    for part in parts:
        if not cur:
            cur = part
            continue
        candidate = f"{cur}{sep}{part}"
        if counter(candidate) <= limit:
            cur = candidate
        else:
            out.append(cur)
            cur = part
    if cur:
        out.append(cur)
    return out


def _split_paragraph(text: str, limit: int, counter: TokenCounter) -> list[str]:
    """Split an oversized paragraph into pieces that fit `limit`."""
    atoms: list[str] = []
    for sentence in _sentences(text) or [text]:
        if counter(sentence) <= limit:
            atoms.append(sentence)
            continue
        for word_piece in _pack_strings(sentence.split(), limit, counter):
            if counter(word_piece) <= limit:
                atoms.append(word_piece)
            else:
                atoms.extend(_split_by_chars(word_piece, limit, counter))
    return _pack_strings(atoms, limit, counter)


def _overlap_for(previous: str, budget: int, counter: TokenCounter) -> str:
    sentences = _sentences(previous)
    if not sentences:
        return ""
    tail = sentences[-1]
    return tail if counter(tail) <= budget else ""


def _units(text: str, max_tokens: int, counter: TokenCounter) -> list[_Unit]:
    overlap_budget = max(1, max_tokens // OVERLAP_DIVISOR)
    # Leave room for the repeated sentence so a piece + its overlap still fits.
    piece_limit = max(1, max_tokens - overlap_budget - 1)
    units: list[_Unit] = []
    for para in _paragraphs(text):
        if counter(para) <= max_tokens:
            units.append(_Unit(para, "", -1, False))
            continue
        pieces = _split_paragraph(para, piece_limit, counter)
        for i, piece in enumerate(pieces):
            overlap = ""
            src = -1
            if i:
                overlap = _overlap_for(pieces[i - 1], overlap_budget, counter)
                if overlap and counter(f"{overlap} {piece}") > max_tokens:
                    overlap = ""
                src = len(units) - 1 if overlap else -1
            units.append(_Unit(piece, overlap, src, bool(i)))
    return units


def _render(units: Sequence[_Unit], idxs: Sequence[int]) -> str:
    out: list[str] = []
    for pos, i in enumerate(idxs):
        unit = units[i]
        piece = unit.text
        # Drop the overlap when its source sits in this very chunk.
        if unit.overlap and not (pos > 0 and idxs[pos - 1] == unit.src):
            piece = f"{unit.overlap} {piece}"
        if pos:
            out.append(" " if unit.joins_prev and idxs[pos - 1] == i - 1 else "\n\n")
        out.append(piece)
    return "".join(out)


# -------------------------------------------------------------------- packing


def _pack_units(
    units: Sequence[_Unit], max_tokens: int, counter: TokenCounter
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for idx in range(len(units)):
        if not current:
            current = [idx]
            continue
        candidate = current + [idx]
        if counter(_render(units, candidate)) <= max_tokens:
            current = candidate
        else:
            groups.append(current)
            current = [idx]
    if current:
        groups.append(current)
    return groups


def _merge_fragments(
    units: Sequence[_Unit],
    groups: list[list[int]],
    max_tokens: int,
    counter: TokenCounter,
) -> list[list[int]]:
    """Absorb sub-threshold chunks into a neighbour, hard bound permitting."""
    min_tokens = max(1, max_tokens // MIN_CHUNK_DIVISOR)
    i = 0
    while i < len(groups) and len(groups) > 1:
        if counter(_render(units, groups[i])) >= min_tokens:
            i += 1
            continue
        best: tuple[int, int] | None = None  # (size, index of left partner)
        for left in (i - 1, i):
            right = left + 1
            if left < 0 or right >= len(groups):
                continue
            merged = groups[left] + groups[right]
            size = counter(_render(units, merged))
            if size <= max_tokens and (best is None or size < best[0]):
                best = (size, left)
        if best is None:
            i += 1
            continue
        left = best[1]
        groups[left : left + 2] = [groups[left] + groups[left + 1]]
        i = max(0, left - 1)
    return groups


# ----------------------------------------------------------------- public API


def _ordered_sections(doc: ExtractedDocLike) -> list[SectionLike]:
    sections = list(doc.sections)
    seen: set[str] = set()
    for section in sections:
        if section.path in seen:
            # IDs are (doc_id, path, n): duplicate paths would collide silently.
            raise ValueError(f"duplicate section path: {section.path!r}")
        seen.add(section.path)
    return sections


def section_rows(doc_id: str, doc: ExtractedDocLike) -> list[dict[str, Any]]:
    """Rows for ``Store.add_sections``, in document order (parents first)."""
    sections = _ordered_sections(doc)
    known = {section.path for section in sections}
    rows: list[dict[str, Any]] = []
    for ordinal, section in enumerate(sections):
        parent_path = section.path.rpartition(".")[0]
        # A parent the extractor never emitted would break the FK: keep it NULL.
        parent = section_id(doc_id, parent_path) if parent_path in known else None
        rows.append(
            {
                "id": section_id(doc_id, section.path),
                "doc_id": doc_id,
                "parent_id": parent,
                "ordinal": ordinal,
                "title": section.title,
                "level": section.level,
                # Optional on the input (SectionLike does not require them);
                # the store keeps them nullable. Pages make chunk->page
                # provenance (citations, page-level eval) a join.
                "page_start": getattr(section, "page_start", None),
                "page_end": getattr(section, "page_end", None),
            }
        )
    return rows


def chunk_document(
    doc_id: str,
    doc: ExtractedDocLike,
    *,
    max_tokens: int,
    token_counter: TokenCounter = estimate_tokens,
) -> list[dict[str, Any]]:
    """Chunk an extracted document. Rows for ``Store.add_chunks``.

    Deterministic: same input, same chunk texts, IDs and ordinals.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for section in _ordered_sections(doc):
        units = _units(section.text, max_tokens, token_counter)
        if not units:
            continue
        packed = _pack_units(units, max_tokens, token_counter)
        groups = _merge_fragments(units, packed, max_tokens, token_counter)
        for n, group in enumerate(groups):
            text = _render(units, group)
            rows.append(
                {
                    "id": chunk_id(doc_id, section.path, n),
                    "doc_id": doc_id,
                    "section_id": section_id(doc_id, section.path),
                    "ordinal": ordinal,
                    "text": text,
                    "token_count": token_counter(text),
                }
            )
            ordinal += 1
    return rows
