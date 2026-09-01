"""S1: document summaries — the L1 gist layer, one cheap pass per document.

One ``summary`` note per document, synthesized *by reference*: the model
receives the document title and a NUMBERED outline of its sections (page
ranges, plus short opening digests where the token budget allows) and
returns new prose describing what the document covers and how it is
organised, together with the entry numbers it actually used. It never
receives whole chapters and is never asked to reproduce source text
(docs/ingestion.md rule 1, no verbatim echo).

Summaries are indexed content (notes_fts, via the store's triggers) and
they are what makes orientation possible: ``retrieval/orient.py`` builds
its document regions from ``get_note(kind="summary", doc_id=...)``, so the
kind and the doc_id below are a contract, not a naming choice. They are
also the raw material for cheap atlas building (v1's Phase F insight).

Document only, not sections
---------------------------
docs/synthesis.md leaves the choice open ("per-document/section
summaries"); this module implements the document level only. A note's
subject is ``(kind, keyword_id, doc_id)`` (docs/storage.md) — the schema
has no section pointer, so a section note could not be addressed, found or
replaced; orientation already derives section regions deterministically
from section titles, with no LLM; and a per-section pass multiplies ingest
cost by the section count for a layer nothing reads. If a section layer
ever earns its cost it needs a schema change first, and that is an
orchestrator decision.

Prompt budget (how a large document is cut down)
------------------------------------------------
``estimate_tokens`` (the one package-wide estimator) prices everything
against ``budget_tokens``. Selection is deterministic, in three steps:

1. **Spine first.** With more sections than ``max_sections``, entries are
   ranked by ``(level, ordinal)`` — the shallowest sections, i.e. the
   chapters, before their subsections — and the surplus is dropped. The
   survivors are then renumbered in document order, so the outline the
   model sees still reads front to back.
2. **Titles fit before digests.** Title lines are added in spine order
   while they fit; a document whose bare outline overflows the budget is
   summarized from the part of the spine that fits, never from a
   truncated line.
3. **Digests share what is left.** The remaining budget is split evenly
   over the entries that own a chunk, clamped to
   ``[MIN_DIGEST_TOKENS, MAX_DIGEST_TOKENS]``. If the share falls below
   the minimum, only the first ``remaining // MIN_DIGEST_TOKENS`` entries
   in spine order get a digest — thin evidence about the chapters beats
   an unreadable sliver of every subsection.

A document with no sections (or none that own chunks) degrades to one
entry per chunk, subject to the same three steps.

Grounding and staleness
-----------------------
``note_sources`` records the chunk IDs whose text the model was actually
shown *and* referenced through ``covers``; if it references none (or the
field is unusable), every shown chunk is recorded instead. Nothing the
model never saw is claimed as a source — that is what makes the store's
staleness rule ("stale iff its source set changed") mean something.

Failure discipline
------------------
A summary is written only after a valid response is parsed, in one
``upsert_note`` call, so a provider failure leaves no half-written note.
Malformed JSON is retried up to ``MAX_ATTEMPTS`` and then raises: unlike
the R2 coverage judge this is not an optimization, and a document without
a gist is a job the runner must retry, not a silent gap in the gist
layer. ``AllKeysExhausted`` propagates untouched so the runner stops the
batch cleanly.

The client seam
---------------
``summarize_stage`` is the runner-compatible stage
(``Callable[[KnowledgeBase, str], None]``) registered under "summarize"
(S1+). Stages take no client argument, so the provider is resolved through
``_client_factory``, the module-level seam this module introduces for
LLM-calling stages — the exact shape of ``_embedder_factory`` in
``wikirag.ingestion``. It returns an already-metered client (stage
"summarize", role "synth"); tests and offline smoke runs swap in a factory
returning a MeteredClient around FakeLLMClient. Nothing here constructs a
provider outside that function.

API
---
- ``SUMMARY_KIND``, ``SUMMARIZE_STAGE``, ``SYNTH_ROLE``,
  ``SYNTH_MODEL_KEY``, ``DEFAULT_SYNTH_MODEL``
- ``DEFAULT_MAX_SECTIONS``, ``DEFAULT_BUDGET_TOKENS``, ``MAX_ATTEMPTS``,
  ``MIN_DIGEST_TOKENS``, ``MAX_DIGEST_TOKENS``, ``MAX_SUMMARY_TOKENS``
- ``SUMMARY_SYSTEM_PROMPT``, ``SUMMARY_RESPONSE_SCHEMA``
- ``summarize_document(kb, doc_id, *, client, max_sections=...,
  budget_tokens=...) -> int`` (the note ID)
- ``summarize_stage(kb, doc_id) -> None``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from wikirag.config import KnowledgeBase
from wikirag.ingestion.chunking import estimate_tokens
from wikirag.llm.client import AllKeysExhausted, LLMClient
from wikirag.llm.meter import MeteredClient

__all__ = [
    "DEFAULT_BUDGET_TOKENS",
    "DEFAULT_MAX_SECTIONS",
    "DEFAULT_SYNTH_MODEL",
    "MAX_ATTEMPTS",
    "MAX_DIGEST_TOKENS",
    "MAX_SUMMARY_TOKENS",
    "MIN_DIGEST_TOKENS",
    "SUMMARIZE_STAGE",
    "SUMMARY_KIND",
    "SUMMARY_RESPONSE_SCHEMA",
    "SUMMARY_SYSTEM_PROMPT",
    "SYNTH_MODEL_KEY",
    "SYNTH_ROLE",
    "summarize_document",
    "summarize_stage",
]

log = logging.getLogger(__name__)

#: Note kind read by retrieval/orient.py to raise a document region.
SUMMARY_KIND = "summary"
SUMMARIZE_STAGE = "summarize"
SYNTH_ROLE = "synth"

#: kb_config key holding the model for the `synth` role (docs/llm.md:
#: stages reference roles, config maps roles to models).
SYNTH_MODEL_KEY = "synth_model"
DEFAULT_SYNTH_MODEL = "gemini-3.5-flash-lite"

#: Outline entries shown at most; a book's chapter list, not its index.
DEFAULT_MAX_SECTIONS = 60
#: Prompt envelope for one document, in estimated tokens.
DEFAULT_BUDGET_TOKENS = 4000
#: Below this a digest is noise; above it, one section starts crowding out
#: the outline that gives the summary its shape.
MIN_DIGEST_TOKENS = 20
MAX_DIGEST_TOKENS = 60
MAX_SUMMARY_TOKENS = 800
MAX_ATTEMPTS = 2

SUMMARY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "covers": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["summary", "covers"],
}

SUMMARY_SYSTEM_PROMPT = """You write the entry summary of one document in a knowledge base. It is read by people and by a retrieval system deciding whether this document is worth opening.

You receive the document TITLE and a numbered OUTLINE of its parts ([1], [2], ...), each with its title, its pages, and sometimes a short digest of how that part opens. You never receive the full text, and you must not ask for it.

Return a JSON object:

- "summary": one to three paragraphs of NEW prose, in your own words, saying what subject matter the document covers, what kind of document it is, and how it is organised (the shape of the outline: which parts carry the core material, which are reference or appendix). Name the parts by their titles when that helps a reader. Write plainly; no headings, no bullet lists, no markdown.
- "covers": the numbers of the outline entries your summary is actually based on. Numbers only, from the outline you were given.

Hard rules:

- Never copy, quote or reproduce sentences from the digests. Every sentence you write must be your own; the digests are evidence, not material to echo.
- Never invent parts, topics or claims that the outline does not support. An outline with little detail deserves a short, cautious summary.
- Refer to parts by their number or their title, never by pasting their content.

Output the JSON object only, no markdown fences, no commentary."""


@dataclass
class _Entry:
    """One numbered outline entry: a section, or a chunk in a flat document."""

    title: str | None
    level: int
    ordinal: int
    page_start: int | None
    page_end: int | None
    chunk_id: str | None
    chunk_text: str
    number: int = 0
    digest: str = ""


# ------------------------------------------------------------------ outline


def _outline(kb: KnowledgeBase, doc_id: str) -> list[_Entry]:
    """The document's outline entries, in document order.

    Sections are the natural entries; a section's digest comes from its
    first chunk. A document whose sections own no chunks at all (or has no
    sections) falls back to one entry per chunk, so a flat extraction still
    produces a gist instead of an error.
    """
    chunks = kb.store.get_document_chunks(doc_id)
    if not chunks:
        raise RuntimeError(f"document {doc_id!r} has no chunks to summarize")

    first_chunk: dict[str, Mapping[str, Any]] = {}
    for chunk in chunks:
        section_id = chunk["section_id"]
        if section_id is not None and section_id not in first_chunk:
            first_chunk[str(section_id)] = chunk

    entries: list[_Entry] = []
    for index, section in enumerate(kb.store.get_sections(doc_id)):
        chunk = first_chunk.get(str(section["id"]))
        entries.append(
            _Entry(
                title=section["title"],
                level=int(section["level"] or 0),
                ordinal=index,
                page_start=section["page_start"],
                page_end=section["page_end"],
                chunk_id=str(chunk["id"]) if chunk is not None else None,
                chunk_text=str(chunk["text"]) if chunk is not None else "",
            )
        )
    if any(entry.chunk_id for entry in entries):
        return entries

    return [
        _Entry(
            title=None,
            level=0,
            ordinal=index,
            page_start=None,
            page_end=None,
            chunk_id=str(chunk["id"]),
            chunk_text=str(chunk["text"]),
        )
        for index, chunk in enumerate(chunks)
    ]


def _title_line(entry: _Entry) -> str:
    title = (entry.title or "").strip() or "(untitled part)"
    pages = _pages(entry)
    return f"[{entry.number}] {title}{pages}"


def _pages(entry: _Entry) -> str:
    start, end = entry.page_start, entry.page_end
    if start is None:
        return ""
    if end is None or end == start:
        return f" (p. {start})"
    return f" (pp. {start}-{end})"


def _digest(text: str, budget: int) -> str:
    """The opening of a chunk, cut to ``budget`` estimated tokens.

    Cut on word boundaries and marked with an ellipsis, so the model can see
    it is reading an opening and not a complete passage.
    """
    words = text.split()
    if not words:
        return ""
    kept: list[str] = []
    # A token is never more than a word, so scanning past `budget` words is
    # wasted work on a long chunk.
    for word in words[: max(1, budget)]:
        candidate = " ".join([*kept, word])
        if estimate_tokens(candidate) > budget:
            break
        kept.append(word)
    if not kept:
        return ""
    opening = " ".join(kept)
    return opening if len(kept) == len(words) else f"{opening} ..."


# ---------------------------------------------------------------- selection


def _spine_order(entries: Sequence[_Entry]) -> list[_Entry]:
    """Shallowest sections first, document order within a level."""
    return sorted(entries, key=lambda e: (e.level, e.ordinal))


def _select(
    entries: Sequence[_Entry], *, max_sections: int, budget: int
) -> list[_Entry]:
    """Pick and number the entries that fit the budget (docstring, steps 1-3)."""
    spine = _spine_order(entries)[: max(1, max_sections)]

    # Numbering must be known before the title lines are priced, and it is
    # assigned in document order, so the outline reads front to back.
    ordered = sorted(spine, key=lambda e: e.ordinal)
    for number, entry in enumerate(ordered, start=1):
        entry.number = number

    kept: list[_Entry] = []
    spent = 0
    for entry in _spine_order(ordered):
        cost = estimate_tokens(_title_line(entry)) + 1  # + the newline
        if spent + cost > budget:
            break
        kept.append(entry)
        spent += cost

    _add_digests(kept, budget - spent)
    return sorted(kept, key=lambda e: e.ordinal)


def _add_digests(kept: Sequence[_Entry], remaining: int) -> None:
    """Spend what the titles left on digests, spine first."""
    grounded = [entry for entry in _spine_order(kept) if entry.chunk_id]
    if not grounded or remaining < MIN_DIGEST_TOKENS:
        return
    slots = min(len(grounded), remaining // MIN_DIGEST_TOKENS)
    share = min(MAX_DIGEST_TOKENS, remaining // slots)
    for entry in grounded[:slots]:
        entry.digest = _digest(entry.chunk_text, share)


def _build_message(title: str, doc_id: str, entries: Sequence[_Entry]) -> str:
    lines = [f"TITLE: {title or doc_id}", "", "OUTLINE:"]
    for entry in entries:
        lines.append(_title_line(entry))
        if entry.digest:
            lines.append(f"    opens: {entry.digest}")
    return "\n".join(lines)


# ----------------------------------------------------------------- response


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse(text: str) -> Mapping[str, Any]:
    """Parse the response, tolerating the markdown fences models add anyway."""
    data = json.loads(_JSON_FENCE.sub("", text or "").strip())
    if not isinstance(data, dict):
        raise json.JSONDecodeError("expected a JSON object", text or "", 0)
    return data


def _sources(data: Mapping[str, Any], entries: Sequence[_Entry]) -> list[str]:
    """Chunk IDs to ground the note in: those shown AND referenced.

    "Shown" means a chunk whose text reached the prompt as a digest; when
    the budget allowed no digest at all, the outline's own chunks stand in,
    since they are what the section titles were read from. A response that
    references nothing usable is grounded in everything it was shown — the
    summary was built from that evidence either way, and an ungrounded note
    could never be told stale.
    """
    shown: dict[int, str] = {
        entry.number: entry.chunk_id for entry in entries if entry.chunk_id and entry.digest
    }
    if not shown:
        shown = {entry.number: entry.chunk_id for entry in entries if entry.chunk_id}
    covers = data.get("covers")
    cited: set[str] = set()
    if isinstance(covers, (list, tuple)):
        for raw in covers:
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                continue
            try:
                chunk_id = shown.get(int(raw))
            except (TypeError, ValueError):
                continue
            if chunk_id is not None:
                cited.add(chunk_id)
    return sorted(cited) if cited else sorted(shown.values())


# ------------------------------------------------------------ public writers


def summarize_document(
    kb: KnowledgeBase,
    doc_id: str,
    *,
    client: LLMClient,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> int:
    """Build (or rebuild) the L1 gist note for one document; returns its ID.

    ``client`` must already be metered (docs/llm.md). The note is stored as
    kind ``summary`` with this ``doc_id``, which is what
    ``retrieval/orient.py`` looks up, and it replaces any existing summary
    for the document rather than accumulating duplicates.

    Raises ``RuntimeError`` when the document has no chunks or the model
    returns nothing usable after ``MAX_ATTEMPTS``; ``AllKeysExhausted``
    propagates. Neither leaves a partial note behind.
    """
    document = kb.store.get_document(doc_id)
    if document is None:
        raise RuntimeError(f"unknown document {doc_id!r}")
    title = str(document["title"] or "").strip()

    entries = _select(
        _outline(kb, doc_id),
        max_sections=max_sections,
        # The system prompt is part of the envelope: it is sent on every call.
        budget=max(MIN_DIGEST_TOKENS, budget_tokens - estimate_tokens(SUMMARY_SYSTEM_PROMPT)),
    )
    user_message = _build_message(title, doc_id, entries)

    data = _generate(client, user_message, doc_id=doc_id)
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise RuntimeError(f"empty summary for document {doc_id!r}")

    # The document's own name belongs in the gist: orientation and FTS match
    # the note's text, and a query naming the book must reach it.
    text = f"{title}\n\n{summary}" if title else summary
    return kb.store.upsert_note(
        kind=SUMMARY_KIND,
        text=text,
        doc_id=doc_id,
        sources=_sources(data, entries),
    )


def _generate(client: LLMClient, user_message: str, *, doc_id: str) -> Mapping[str, Any]:
    """One summary response, retried on malformed JSON, never on a provider error."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = client.generate(
            system=SUMMARY_SYSTEM_PROMPT,
            user=user_message,
            json_schema=SUMMARY_RESPONSE_SCHEMA,
            max_output_tokens=MAX_SUMMARY_TOKENS,
        )
        try:
            return _parse(result.text)
        except json.JSONDecodeError:
            # docs/llm.md: the parse fallback logs every time it fires, so its
            # weight against constrained decoding is measurable.
            log.warning(
                "summarize: unparsable response for %s (attempt %d/%d)",
                doc_id,
                attempt,
                MAX_ATTEMPTS,
            )
    raise RuntimeError(f"no parsable summary for document {doc_id!r}")


# --------------------------------------------------------------- stage seam


def _default_client(kb: KnowledgeBase) -> LLMClient:
    """The metered client for the `synth` role of this KB."""
    # Imported here so that no S0 run has to load google-genai.
    from wikirag.llm.gemini import GeminiClient

    model = kb.store.config_get(SYNTH_MODEL_KEY, DEFAULT_SYNTH_MODEL)
    assert model is not None
    return MeteredClient(
        GeminiClient(model), store=kb.store, stage=SUMMARIZE_STAGE, role=SYNTH_ROLE
    )


#: Seam for tests and offline smoke runs (mirrors `_embedder_factory` in
#: wikirag.ingestion): swap in a factory returning a metered FakeLLMClient.
#: It must return an ALREADY-METERED client — the stage never wraps.
_client_factory: Callable[[KnowledgeBase], LLMClient] = _default_client


def summarize_stage(kb: KnowledgeBase, doc_id: str) -> None:
    """Runner stage "summarize" (S1+): one gist note per document.

    Idempotent the way an LLM stage has to be: a fresh (non-stale) summary
    for the document is proof of completion, so a re-run after an
    interruption costs nothing. Call ``summarize_document`` directly to
    force a rebuild (`regenerate-notes`, docs/synthesis.md).
    """
    existing = kb.store.get_note(kind=SUMMARY_KIND, doc_id=doc_id)
    if existing is not None and not existing["stale"]:
        return
    summarize_document(kb, doc_id, client=_client_factory(kb))
