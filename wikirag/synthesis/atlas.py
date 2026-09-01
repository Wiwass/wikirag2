"""S3: the atlas — overview map, hub pages, link curation.

Purely for the human product and for global/sensemaking queries; expected
to show little R0-R1 benchmark gain, and that is fine and stated up front
(docs/synthesis.md).

- Overview: map-reduce over the L1 summaries. One reduce call when they
  fit the budget, a map pass per batch plus one reduce when they do not.
  The single ``overview`` note is what ``retrieval/orient.py`` looks up as
  the KB-wide half of the gist layer and what ``router.classify`` takes as
  its primer, so the kind is a contract, not a label.
- Hub pages: curated entry points for high-centrality concepts, stored as
  ``hub`` notes (see "Note kinds" below).
- Link curation: report near-duplicate keywords and weak links.

Everything here is re-runnable. Notes are addressed by subject
(kind, keyword_id, doc_id), so a rebuild replaces the previous overview
and each previous hub instead of accumulating.

What exists, not what is configured
-----------------------------------
Like retrieval, this module branches on the content in the store and never
on the KB's S level. With no L1 summaries — every S0 KB, and any S1 KB
whose summaries have not run yet — the atlas has no material:
:func:`build_overview` returns ``None`` and writes nothing, and
:func:`build_hubs` returns ``[]`` on a KB with no keywords. Degrading
quietly is the specified behaviour, not an error path.

Note kinds
----------
``overview`` (keyword_id NULL, doc_id NULL) — the single L3 map, the kind
``orient`` reads. ``hub`` (keyword_id set) — one entry-point page per
central concept, a new kind on purpose: it must not collide with the L2
``keyword`` note (a hub is navigation, the note is content, and both exist
for the same keyword), nor with ``summary``/``overview``, whose subjects
are documents and the KB itself.

Grounding
---------
``note_sources`` takes chunk IDs only (its column is an FK to ``chunks``),
so a note cannot record that it was built from other notes. The atlas
therefore grounds *transitively*: the overview's source set is the union
of the source chunks of the summaries the model reported using, and a
hub's is its concept's tagged chunks — the material the page is an entry
point to. Both keep the staleness rule honest (``forget`` cascades into
those rows, so a shrunken source set marks the atlas stale) and both are
approximations of a provenance the schema cannot express. See "Open
issues" below.

Centrality
----------
A concept's centrality mixes three cheap signals, each normalized to
``[0, 1]`` against the strongest concept and weighted: how much material
carries it (``chunk_keywords`` tags), how much of the vault points at it
(incoming ``note_links``), and how often queries hit it (``query_log``,
the D11 usage signal, at half weight because usage is noisier and much
sparser than structure).

Link curation
-------------
Reporting is the product; mutation is opt-in and never destructive.
Merging two keywords would delete a row other tables point at, and the
store exposes no way to delete a ``note_links`` row at all, so
:func:`curate_links` never merges a keyword and never prunes a link. What
it does with ``dry_run=False`` is the safe half: it marks the notes of
near-duplicate concepts stale and queues them for consolidation, so the
next synthesis pass reconciles them under review. ``dry_run=True`` reports
and touches nothing.

No verbatim echo (docs/ingestion.md rule 1)
-------------------------------------------
Every prompt here receives numbered items — summaries, digests, concept
names — and asks for new prose plus the numbers used. Chunk text never
enters an atlas prompt; the inputs are prose the system wrote itself, and
the prompts forbid copying it forward.

Open issues
-----------
- ``note_sources`` cannot express note-to-note provenance (chunk FK only);
  the overview and hub grounding above is the transitive stand-in.
- ``add_note_links`` appends and nothing deletes a link, so a hub rebuild
  can add links but not retract them. New links are deduplicated against
  the ones already stored; a link the model stops choosing survives, and
  :func:`curate_links` is what surfaces it afterwards.

Public API
----------
- ``OVERVIEW_KIND``, ``HUB_KIND``, ``HUB_LINK_TYPE``
- ``DEFAULT_BUDGET_TOKENS``, ``DEFAULT_HUB_LIMIT``, ``DUPLICATE_SIMILARITY``
- ``ATLAS_RESPONSE_SCHEMA``, ``OVERVIEW_MAP_SYSTEM_PROMPT``,
  ``OVERVIEW_REDUCE_SYSTEM_PROMPT``, ``HUB_SYSTEM_PROMPT``
- ``class Centrality``, ``class DuplicateCandidate``, ``class WeakLink``,
  ``class CurationReport``
- ``build_overview(kb, *, client, budget_tokens=DEFAULT_BUDGET_TOKENS) -> int | None``
- ``build_hubs(kb, *, client, limit=DEFAULT_HUB_LIMIT) -> list[int]``
- ``central_keywords(kb, *, limit=None) -> list[Centrality]``
- ``curate_links(kb, *, embedder=None, dry_run=False) -> CurationReport``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from wikirag.config import KnowledgeBase
from wikirag.ingestion.chunking import estimate_tokens
from wikirag.llm.client import AllKeysExhausted, LLMClient
from wikirag.llm.embeddings import EmbedderLike

__all__ = [
    "ATLAS_RESPONSE_SCHEMA",
    "DEFAULT_BUDGET_TOKENS",
    "DEFAULT_HUB_LIMIT",
    "DUPLICATE_SIMILARITY",
    "HUB_KIND",
    "HUB_LINK_TYPE",
    "HUB_SYSTEM_PROMPT",
    "OVERVIEW_KIND",
    "OVERVIEW_MAP_SYSTEM_PROMPT",
    "OVERVIEW_REDUCE_SYSTEM_PROMPT",
    "Centrality",
    "CurationReport",
    "DuplicateCandidate",
    "WeakLink",
    "build_hubs",
    "build_overview",
    "central_keywords",
    "curate_links",
]

#: The kind orient.py and router.py depend on. Subject: (overview, NULL, NULL).
OVERVIEW_KIND = "overview"

#: Hub pages: one per central concept, keyed by keyword_id.
HUB_KIND = "hub"

#: The single type used by links a hub page owns. Hubs do not invent a link
#: vocabulary: they say "this concept is listed on that hub", and the typed
#: semantic vocabulary of D10 belongs to L2 note synthesis.
HUB_LINK_TYPE = "hub"

#: Token envelope for one map batch of summaries. The whole point of the
#: map-reduce is that no single prompt grows with the corpus.
DEFAULT_BUDGET_TOKENS = 6_000

DEFAULT_HUB_LIMIT = 8

#: Cosine above which two keyword names are reported as near-duplicates.
#: High on purpose: a false merge proposal costs a human review, and this
#: report is read by someone who trusts it.
DUPLICATE_SIMILARITY = 0.93

#: Output reservations. The overview is a page, a digest and a hub are notes.
OVERVIEW_OUTPUT_TOKENS = 1_200
MAP_OUTPUT_TOKENS = 600
HUB_OUTPUT_TOKENS = 500

#: Concept names offered to the reduce call as the KB's spine.
MAX_OVERVIEW_CONCEPTS = 20

#: Neighbours shown on one hub page, and the chunks scanned to find them.
#: The neighbourhood is a shortlist, not a census: a concept tagged on
#: hundreds of chunks would otherwise cost hundreds of lookups per hub.
MAX_HUB_NEIGHBOURS = 12
MAX_NEIGHBOUR_CHUNKS = 60

#: Queries read for the usage signal (D11).
USAGE_WINDOW = 200

_TAG_WEIGHT = 1.0
_LINK_WEIGHT = 1.0
_USAGE_WEIGHT = 0.5

_MAX_ATTEMPTS = 2

ATLAS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "used": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["text", "used"],
}

OVERVIEW_MAP_SYSTEM_PROMPT = """You are compressing part of a personal knowledge base into a digest, for a later pass that will write the base's overview.

You receive numbered document summaries ([1], [2], ...). Each is a short description of one document, already written by this system.

Produce:
- "text": one dense paragraph (at most 120 words) in your own words, saying what subjects this group of documents covers and what kind of material it is (reference, narrative, rules, notes, ...). Group documents that belong together instead of listing them one by one.
- "used": the numbers of the summaries your paragraph actually reflects.

Rules:
- Write new prose. Never copy a sentence or a distinctive phrase from a summary.
- Refer to a document by its title, never by quoting its summary.
- Use only what the numbered summaries state. Do not guess at content you were not shown, and never ask for the underlying documents.

Output the JSON object only, no markdown fences, no commentary."""

OVERVIEW_REDUCE_SYSTEM_PROMPT = """You are writing the overview of a personal knowledge base: the page someone reads first to learn what is in it and how it is organised.

You receive numbered items ([1], [2], ...) — either document summaries or digests of groups of documents, both already written by this system — and possibly a list of the base's most central concepts.

Produce:
- "text": the overview, 150 to 300 words of continuous prose. Say what the knowledge base is about, what its main areas are, how the material is organised across them, and which concepts run through it. Write it as a map for a reader who has never seen the base, not as a list of documents.
- "used": the numbers of the items your overview actually reflects.

Rules:
- Write new prose. Never copy a sentence or a distinctive phrase from an item.
- Describe only what the items state. Do not invent documents, topics, counts or dates, and never ask for the underlying source text.
- No preamble, no headings, no bullet lists — one continuous piece of prose.

Output the JSON object only, no markdown fences, no commentary."""

HUB_SYSTEM_PROMPT = """You are writing a hub page for one concept in a personal knowledge base: a short entry point that tells a reader what the concept covers here and where to go next.

You receive the concept's name, sometimes the note this system already wrote about it, and a numbered list of neighbouring concepts ([1], [2], ...) that appear alongside it in the material.

Produce:
- "text": at most 120 words of prose. Say what this concept covers in this knowledge base and how the neighbours listed relate to it. Name a neighbour in the prose only if it is genuinely related.
- "used": the numbers of the neighbours worth linking from this page, most relevant first. Leave it empty rather than linking a neighbour that only happens to co-occur.

Rules:
- Write new prose. Never copy a sentence from the note you were given.
- Use only the concept name, the note and the neighbour names. Invent no facts, no sources and no concepts that are not in the list, and never ask for the underlying documents.

Output the JSON object only, no markdown fences, no commentary."""

_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


# ------------------------------------------------------------------- results


@dataclass(frozen=True)
class Centrality:
    """How central one concept is, and on what evidence."""

    keyword_id: int
    name: str
    tag_count: int
    inbound_links: int
    query_hits: int
    score: float


@dataclass(frozen=True)
class DuplicateCandidate:
    """Two keywords that look like the same concept. A proposal, never applied."""

    keyword_ids: tuple[int, int]
    names: tuple[str, str]
    similarity: float
    reason: str  # "normalized-name" or "embedding"


@dataclass(frozen=True)
class WeakLink:
    """A stored link that carries no navigational value."""

    link_id: int
    from_note_id: int
    to_keyword_id: int
    type: str
    reason: str


@dataclass
class CurationReport:
    """What curation found, and what (if anything) it acted on.

    ``marked_stale`` and ``queued`` are empty under ``dry_run``, and are the
    only mutations curation ever performs: no keyword is merged, no link is
    pruned, nothing is deleted.
    """

    duplicates: list[DuplicateCandidate] = field(default_factory=list)
    weak_links: list[WeakLink] = field(default_factory=list)
    dry_run: bool = False
    marked_stale: list[int] = field(default_factory=list)  # note ids
    queued: list[int] = field(default_factory=list)  # keyword ids

    def is_empty(self) -> bool:
        return not (self.duplicates or self.weak_links)


# ------------------------------------------------------------ LLM plumbing


def _parse_used(raw: Any) -> list[int]:
    """The numbers a model claims to have used; anything unparsable drops."""
    if not isinstance(raw, (list, tuple)):
        return []
    numbers: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            numbers.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            numbers.append(int(item.strip()))
    return numbers


def _generate(
    client: LLMClient, system: str, user: str, max_output_tokens: int
) -> tuple[str, list[int]] | None:
    """One atlas call: prose plus the item numbers it used, or None on failure.

    Malformed or empty output is retried once and then given up on — an
    unbuildable atlas page is not an error, it is a page that does not exist
    yet. ``AllKeysExhausted`` propagates so the caller can stop the run.
    """
    for _ in range(_MAX_ATTEMPTS):
        try:
            result = client.generate(
                system=system,
                user=user,
                json_schema=ATLAS_RESPONSE_SCHEMA,
                max_output_tokens=max_output_tokens,
            )
            payload = json.loads(_JSON_FENCE.sub("", result.text or "").strip())
        except AllKeysExhausted:
            raise
        except json.JSONDecodeError:
            continue
        except Exception:
            return None
        if not isinstance(payload, dict):
            continue
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        return text.strip(), _parse_used(payload.get("used"))
    return None


def _numbered(items: Sequence[str]) -> str:
    return "\n\n".join(f"[{n}] {item}" for n, item in enumerate(items, start=1))


# ---------------------------------------------------------------- overview


@dataclass(frozen=True)
class _Summary:
    note_id: int
    title: str
    text: str

    @property
    def block(self) -> str:
        return f"{self.title}\n{self.text}"


def _document_summaries(kb: KnowledgeBase) -> list[_Summary]:
    """The L1 layer: one summary note per document, in document order.

    Read from the store rather than from ``synthesis.summaries``: the atlas
    consumes what exists, and works the same whether the summaries were built
    this run, last week, or not at all.
    """
    summaries: list[_Summary] = []
    for document in kb.store.list_documents():
        note = kb.store.get_note(kind="summary", doc_id=document["id"])
        if note is None or not str(note["text"]).strip():
            continue
        title = str(document["title"] or document["id"])
        summaries.append(
            _Summary(note_id=int(note["id"]), title=title, text=str(note["text"]).strip())
        )
    return summaries


def _batches(summaries: Sequence[_Summary], budget_tokens: int) -> list[list[_Summary]]:
    """Greedy packing into prompt-sized groups; one summary always fits alone."""
    budget = max(1, budget_tokens)
    batches: list[list[_Summary]] = []
    current: list[_Summary] = []
    spent = 0
    for summary in summaries:
        cost = estimate_tokens(summary.block)
        if current and spent + cost > budget:
            batches.append(current)
            current, spent = [], 0
        current.append(summary)
        spent += cost
    if current:
        batches.append(current)
    return batches


def _summary_message(batch: Sequence[_Summary]) -> str:
    return "DOCUMENT SUMMARIES:\n\n" + _numbered([s.block for s in batch])


def _reduce_message(items: str, concepts: Sequence[str]) -> str:
    parts = [items]
    if concepts:
        parts.append("MOST CENTRAL CONCEPTS:\n" + ", ".join(concepts))
    return "\n\n".join(parts)


def _resolve(batch: Sequence[_Summary], used: Sequence[int]) -> set[int]:
    """Item numbers (1-based, per call) back to the note IDs they stand for."""
    return {batch[n - 1].note_id for n in used if 1 <= n <= len(batch)}


def build_overview(
    kb: KnowledgeBase,
    *,
    client: LLMClient,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> int | None:
    """Build (or rebuild) the single ``overview`` note. Returns its ID.

    Map-reduce over the L1 summaries: one reduce call when they fit
    ``budget_tokens``, otherwise one map call per batch producing a digest
    and a single reduce over the digests. The most central concept names go
    to the reduce call as the KB's spine.

    Returns ``None`` and writes nothing when the KB has no L1 summaries (an
    S0 KB has no atlas material), when every call fails, or when the model
    returns nothing usable. ``AllKeysExhausted`` stops the run cleanly and
    also returns ``None``: half an overview is not a partial result worth
    storing, and the previous overview stays untouched.

    Grounding is transitive — the source chunks of the summaries the model
    reported using (see "Grounding" in the module docstring).
    """
    summaries = _document_summaries(kb)
    if not summaries:
        return None

    concepts = [c.name for c in central_keywords(kb, limit=MAX_OVERVIEW_CONCEPTS)]
    batches = _batches(summaries, budget_tokens)
    used_notes: set[int] = set()
    fed_notes: set[int] = set()

    try:
        if len(batches) == 1:
            reduced = _generate(
                client,
                OVERVIEW_REDUCE_SYSTEM_PROMPT,
                _reduce_message(_summary_message(batches[0]), concepts),
                OVERVIEW_OUTPUT_TOKENS,
            )
            if reduced is None:
                return None
            text, used = reduced
            fed_notes = {s.note_id for s in batches[0]}
            used_notes = _resolve(batches[0], used)
        else:
            digests: list[str] = []
            for batch in batches:
                mapped = _generate(
                    client,
                    OVERVIEW_MAP_SYSTEM_PROMPT,
                    _summary_message(batch),
                    MAP_OUTPUT_TOKENS,
                )
                if mapped is None:
                    continue
                digests.append(mapped[0])
                fed_notes |= {s.note_id for s in batch}
                used_notes |= _resolve(batch, mapped[1])
            if not digests:
                return None
            reduced = _generate(
                client,
                OVERVIEW_REDUCE_SYSTEM_PROMPT,
                _reduce_message("DIGESTS:\n\n" + _numbered(digests), concepts),
                OVERVIEW_OUTPUT_TOKENS,
            )
            if reduced is None:
                return None
            text = reduced[0]
    except AllKeysExhausted:
        return None

    # A model that reports no usable numbers still wrote from everything it
    # was shown, so the grounding falls back to the whole fed set rather than
    # storing an ungrounded page.
    grounding = used_notes or fed_notes
    sources = sorted(
        {chunk_id for nid in grounding for chunk_id in kb.store.note_sources(nid)}
    )
    return kb.store.upsert_note(kind=OVERVIEW_KIND, text=text, sources=sources)


# -------------------------------------------------------------- centrality


def _note_subjects(kb: KnowledgeBase) -> list[dict[str, Any]]:
    """Every note in the store, enumerated by subject.

    The store has no "list notes" API (the constraint orient.py records too),
    so the notes are reached through their subjects: the overview, one summary
    per document, and the keyword note and hub page of each keyword.
    """
    notes: list[dict[str, Any]] = []
    overview = kb.store.get_note(kind=OVERVIEW_KIND)
    if overview is not None:
        notes.append(overview)
    for document in kb.store.list_documents():
        summary = kb.store.get_note(kind="summary", doc_id=document["id"])
        if summary is not None:
            notes.append(summary)
    for keyword in kb.store.list_keywords():
        for kind in ("keyword", HUB_KIND):
            note = kb.store.get_note(kind=kind, keyword_id=int(keyword["id"]))
            if note is not None:
                notes.append(note)
    return notes


def _inbound_links(kb: KnowledgeBase) -> dict[int, int]:
    """keyword_id -> how many stored links point at it."""
    inbound: dict[int, int] = {}
    for note in _note_subjects(kb):
        for link in kb.store.note_links_from(int(note["id"])):
            target = int(link["to_keyword_id"])
            inbound[target] = inbound.get(target, 0) + 1
    return inbound


def _query_hits(kb: KnowledgeBase, keywords: Sequence[dict[str, Any]]) -> dict[int, int]:
    """keyword_id -> how often recent queries touched it (D11 usage signal).

    ``query_log.hit_keywords`` is a list of strings whose entries may be
    keyword IDs or names depending on the caller, so both are accepted. The
    query text itself is also scanned for whole-word occurrences of a concept
    name: nothing populates ``hit_keywords`` yet, and a usage signal that is
    always zero would silently drop a third of centrality.
    """
    by_id = {str(int(k["id"])): int(k["id"]) for k in keywords}
    by_name = {str(k["canonical_name"]): int(k["id"]) for k in keywords}
    patterns = {
        int(k["id"]): re.compile(
            rf"(?<!\w){re.escape(' '.join(_WORD_RE.findall(str(k['canonical_name']))))}(?!\w)"
        )
        for k in keywords
        if _WORD_RE.findall(str(k["canonical_name"]))
    }

    hits: dict[int, int] = {}
    for row in kb.store.get_queries(limit=USAGE_WINDOW):
        touched: set[int] = set()
        for raw in row.get("hit_keywords", []):
            token = str(raw).strip()
            keyword_id = by_id.get(token) or by_name.get(token.lower())
            if keyword_id is not None:
                touched.add(keyword_id)
        haystack = " ".join(_WORD_RE.findall(str(row["query"]).lower()))
        if haystack:
            touched |= {kid for kid, pattern in patterns.items() if pattern.search(haystack)}
        for keyword_id in touched:
            hits[keyword_id] = hits.get(keyword_id, 0) + 1
    return hits


def _normalize(value: int, top: int) -> float:
    return value / top if top > 0 else 0.0


def central_keywords(
    kb: KnowledgeBase, *, limit: int | None = None
) -> list[Centrality]:
    """Concepts ranked by centrality, strongest first.

    Concepts with no evidence at all (no tags, no incoming links, no query
    hits) are dropped: they are names someone proposed, not centres of
    anything. Ties break on the canonical name so the ranking is stable.
    """
    keywords = kb.store.list_keywords()
    if not keywords:
        return []

    inbound = _inbound_links(kb)
    hits = _query_hits(kb, keywords)
    tags = {
        int(k["id"]): len(kb.store.chunks_for_keyword(int(k["id"]))) for k in keywords
    }

    top_tags = max(tags.values(), default=0)
    top_links = max(inbound.values(), default=0)
    top_hits = max(hits.values(), default=0)

    ranked: list[Centrality] = []
    for keyword in keywords:
        keyword_id = int(keyword["id"])
        tag_count = tags[keyword_id]
        inbound_links = inbound.get(keyword_id, 0)
        query_hits = hits.get(keyword_id, 0)
        if not (tag_count or inbound_links or query_hits):
            continue
        score = (
            _TAG_WEIGHT * _normalize(tag_count, top_tags)
            + _LINK_WEIGHT * _normalize(inbound_links, top_links)
            + _USAGE_WEIGHT * _normalize(query_hits, top_hits)
        )
        ranked.append(
            Centrality(
                keyword_id=keyword_id,
                name=str(keyword["name"]),
                tag_count=tag_count,
                inbound_links=inbound_links,
                query_hits=query_hits,
                score=score,
            )
        )
    ranked.sort(key=lambda c: (-c.score, c.name.lower()))
    return ranked[:limit] if limit is not None else ranked


# --------------------------------------------------------------- hub pages


@dataclass(frozen=True)
class _Neighbour:
    keyword_id: int
    name: str
    shared: int
    grounding_chunk_id: str | None


def _neighbours(kb: KnowledgeBase, keyword_id: int) -> list[_Neighbour]:
    """Concepts co-occurring with this one, with a chunk that witnesses it."""
    shared: dict[int, int] = {}
    names: dict[int, str] = {}
    witness: dict[int, str] = {}
    for chunk in kb.store.chunks_for_keyword(keyword_id)[:MAX_NEIGHBOUR_CHUNKS]:
        for other in kb.store.keywords_for_chunk(str(chunk["id"])):
            other_id = int(other["id"])
            if other_id == keyword_id:
                continue
            shared[other_id] = shared.get(other_id, 0) + 1
            names[other_id] = str(other["name"])
            witness.setdefault(other_id, str(chunk["id"]))
    ranked = sorted(shared.items(), key=lambda item: (-item[1], names[item[0]].lower()))
    return [
        _Neighbour(
            keyword_id=other_id,
            name=names[other_id],
            shared=count,
            grounding_chunk_id=witness.get(other_id),
        )
        for other_id, count in ranked[:MAX_HUB_NEIGHBOURS]
    ]


def _hub_message(
    concept: Centrality, note: dict[str, Any] | None, neighbours: Sequence[_Neighbour]
) -> str:
    parts = [f"CONCEPT: {concept.name}"]
    if note is not None and str(note["text"]).strip():
        parts.append("EXISTING NOTE ON THIS CONCEPT:\n" + str(note["text"]).strip())
    if neighbours:
        parts.append(
            "NEIGHBOURING CONCEPTS:\n"
            + _numbered([f"{n.name} (shares {n.shared} passages)" for n in neighbours])
        )
    else:
        parts.append("NEIGHBOURING CONCEPTS:\n(none)")
    return "\n\n".join(parts)


def _store_hub_links(
    kb: KnowledgeBase,
    note_id: int,
    neighbours: Sequence[_Neighbour],
    used: Sequence[int],
) -> None:
    """Add the chosen neighbour links, skipping ones already stored.

    Nothing can delete a link, so a rebuild is additive and must at least not
    duplicate what it wrote last time (module docstring, "Open issues").
    """
    existing = {
        (int(link["to_keyword_id"]), str(link["type"]))
        for link in kb.store.note_links_from(note_id)
    }
    links: list[dict[str, Any]] = []
    for number in used:
        if not 1 <= number <= len(neighbours):
            continue
        neighbour = neighbours[number - 1]
        if (neighbour.keyword_id, HUB_LINK_TYPE) in existing:
            continue
        existing.add((neighbour.keyword_id, HUB_LINK_TYPE))
        links.append(
            {
                "to_keyword_id": neighbour.keyword_id,
                "type": HUB_LINK_TYPE,
                "grounding_chunk_id": neighbour.grounding_chunk_id,
            }
        )
    if links:
        kb.store.add_note_links(note_id, links)


def build_hubs(
    kb: KnowledgeBase, *, client: LLMClient, limit: int = DEFAULT_HUB_LIMIT
) -> list[int]:
    """Build (or rebuild) hub pages for the ``limit`` most central concepts.

    Returns the note IDs written, most central first. One LLM call per hub;
    a call that fails skips that concept and the run continues.
    ``AllKeysExhausted`` stops the run and returns the hubs written so far —
    the partial progress is real, each hub being a complete page.

    A KB with no keywords (S0/S1) yields ``[]`` without an LLM call.
    """
    written: list[int] = []
    for concept in central_keywords(kb, limit=max(0, limit)):
        neighbours = _neighbours(kb, concept.keyword_id)
        note = kb.store.get_note(kind="keyword", keyword_id=concept.keyword_id)
        try:
            generated = _generate(
                client,
                HUB_SYSTEM_PROMPT,
                _hub_message(concept, note, neighbours),
                HUB_OUTPUT_TOKENS,
            )
        except AllKeysExhausted:
            break
        if generated is None:
            continue
        text, used = generated
        sources = [
            str(chunk["id"]) for chunk in kb.store.chunks_for_keyword(concept.keyword_id)
        ]
        note_id = kb.store.upsert_note(
            kind=HUB_KIND,
            keyword_id=concept.keyword_id,
            text=text,
            sources=sources,
        )
        _store_hub_links(kb, note_id, neighbours, used)
        written.append(note_id)
    return written


# ----------------------------------------------------------- link curation


def _duplicate_key(canonical_name: str) -> str:
    """Fold a canonical name to its rough concept identity.

    ``canonical_name`` is already lowercased and UNIQUE, so exact equality can
    never fire; what this collapses is the spelling difference that survived
    it — punctuation, spacing, and a trailing plural.
    """
    folded = _NON_ALNUM.sub(" ", canonical_name.lower()).strip()
    words = folded.split()
    singular = []
    for word in words:
        if len(word) > 4 and word.endswith("ies"):
            singular.append(word[:-3] + "y")
        elif len(word) > 4 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
            singular.append(word[:-2])
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            singular.append(word[:-1])
        else:
            singular.append(word)
    return " ".join(singular)


def _name_duplicates(keywords: Sequence[dict[str, Any]]) -> list[DuplicateCandidate]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for keyword in keywords:
        key = _duplicate_key(str(keyword["canonical_name"]))
        if key:
            groups.setdefault(key, []).append(keyword)
    candidates: list[DuplicateCandidate] = []
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                candidates.append(
                    DuplicateCandidate(
                        keyword_ids=(int(group[i]["id"]), int(group[j]["id"])),
                        names=(str(group[i]["name"]), str(group[j]["name"])),
                        similarity=1.0,
                        reason="normalized-name",
                    )
                )
    return candidates


def _embedding_duplicates(
    keywords: Sequence[dict[str, Any]],
    embedder: EmbedderLike,
    seen: set[tuple[int, int]],
) -> list[DuplicateCandidate]:
    """Near-duplicates by cosine over the canonical names. One encode call."""
    if len(keywords) < 2:
        return []
    vectors = np.asarray(
        embedder.encode_passages([str(k["canonical_name"]) for k in keywords]),
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarity = vectors @ vectors.T

    candidates: list[DuplicateCandidate] = []
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            pair = (int(keywords[i]["id"]), int(keywords[j]["id"]))
            if pair in seen or float(similarity[i, j]) < DUPLICATE_SIMILARITY:
                continue
            candidates.append(
                DuplicateCandidate(
                    keyword_ids=pair,
                    names=(str(keywords[i]["name"]), str(keywords[j]["name"])),
                    similarity=float(similarity[i, j]),
                    reason="embedding",
                )
            )
    return candidates


def _weak_links(kb: KnowledgeBase) -> list[WeakLink]:
    """Links worth a human's attention: they point at nothing, or at nothing said.

    Two rules, both about the target rather than the wording: a target with no
    tagged material is a link into the void, and an ungrounded link to a
    concept that has no note yet is a claim nothing backs.
    """
    has_material: dict[int, bool] = {}
    has_note: dict[int, bool] = {}
    weak: list[WeakLink] = []
    for note in _note_subjects(kb):
        for link in kb.store.note_links_from(int(note["id"])):
            target = int(link["to_keyword_id"])
            if target not in has_material:
                has_material[target] = bool(kb.store.chunks_for_keyword(target))
                has_note[target] = (
                    kb.store.get_note(kind="keyword", keyword_id=target) is not None
                )
            if has_material[target] and (
                link["grounding_chunk_id"] or has_note[target]
            ):
                continue
            reason = (
                "target has no tagged material"
                if not has_material[target]
                else "ungrounded link to a concept with no note"
            )
            weak.append(
                WeakLink(
                    link_id=int(link["id"]),
                    from_note_id=int(note["id"]),
                    to_keyword_id=target,
                    type=str(link["type"]),
                    reason=reason,
                )
            )
    return weak


def curate_links(
    kb: KnowledgeBase,
    *,
    embedder: EmbedderLike | None = None,
    dry_run: bool = False,
) -> CurationReport:
    """Report near-duplicate concepts and weak links; act only on the safe half.

    Duplicate candidates come from folded canonical names (punctuation,
    spacing, plurals) and, when an ``embedder`` is given, from cosine over the
    names above ``DUPLICATE_SIMILARITY``. Without an embedder no model is
    loaded and only the name rule runs.

    Nothing is merged, pruned or deleted, ever (module docstring, "Link
    curation"). With ``dry_run=False`` the safe half is applied: the notes of
    duplicate concepts are marked stale and both concepts are queued for
    consolidation, so the next synthesis pass reconciles them. With
    ``dry_run=True`` the store is not written at all.
    """
    keywords = kb.store.list_keywords()
    duplicates = _name_duplicates(keywords)
    if embedder is not None:
        duplicates += _embedding_duplicates(
            keywords, embedder, {d.keyword_ids for d in duplicates}
        )
    duplicates.sort(key=lambda d: (-d.similarity, d.keyword_ids))

    report = CurationReport(
        duplicates=duplicates, weak_links=_weak_links(kb), dry_run=dry_run
    )
    if dry_run:
        return report

    for candidate in duplicates:
        for keyword_id in candidate.keyword_ids:
            note = kb.store.get_note(kind="keyword", keyword_id=keyword_id)
            if note is not None and not note["stale"]:
                kb.store.set_note_stale(int(note["id"]))
                report.marked_stale.append(int(note["id"]))
            if keyword_id not in report.queued:
                kb.store.queue_consolidation(keyword_id, "duplicate")
                report.queued.append(keyword_id)
    return report
