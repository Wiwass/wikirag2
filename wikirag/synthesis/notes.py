"""S2: keyword notes — the wiki layer, rebuilt on chunk references.

Per keyword: gather its tagged chunks (chunk_keywords), synthesize a
wiki-style note, choose wikilinks from an embedding-shortlisted candidate
list of *existing* keywords. The model links only names it was given, so
orphan links cannot be created (v1's whole orphan-recovery subsystem
reduces to an invariant check).

Links are written to note_links as the note is stored; provenance to
note_sources. Notes are indexed like any content (the store's FTS triggers
do that on write). Regeneration is per-keyword.

Lazy by design (D11): this module only knows HOW to synthesize one note;
WHEN is decided by consolidation.py (on-demand triggers, background jobs,
staleness).

Note kind
---------
Keyword notes are stored with ``kind="keyword"`` and their ``keyword_id``.
That is a cross-module contract: ``retrieval.verify.trigger_consolidation``
looks a keyword up by exactly that subject to decide whether it still needs
consolidating, so any other kind would leave the trigger permanently armed.

The prompt (docs/ingestion.md rule 1 — no verbatim echo)
--------------------------------------------------------
The model receives the concept name, its sources NUMBERED ``[1] [2] ...``,
and a numbered candidate list of existing concepts. It returns new prose
plus the *numbers* it used — never copies of source text, never chunk IDs
(long, easy to corrupt, and the caller already knows them). Numbers are
mapped back to chunk IDs here, so anything the model invents resolves to
nothing instead of to a wrong row.

Source selection and the grounding rule
---------------------------------------
Sources are the keyword's chunks in store order (doc_id, ordinal), capped
by count *and* an estimated token budget — a note is a summary of a
concept, not a concatenation of a corpus. The selection is deterministic,
which is what makes staleness cheap: ``note_sources`` records the whole
selection the note was built from (not only the subset the model cited),
so a changed source set is detectable by set comparison alone, and a note
built from a capped selection does not report itself stale forever. The
model must still cite at least one source it was given: a reply grounded
in nothing is a failed synthesis, not a note.

Links
-----
Candidates are the KB's other keywords, shortlisted by embedding
similarity to the note's own material (name + selected sources) — the same
"never put the whole index in a prompt" rule the enrich stage follows. The
model picks names from that shortlist only; a name outside it (invented,
or an existing keyword that did not make the shortlist) is dropped. Links
carry a type from :data:`LINK_TYPES` and, where the model names a
supporting source, its chunk ID.

Links are additive across regenerations: a target already linked from the
note is not linked twice, and links that a later synthesis no longer
chooses are left in place. Pruning weak links is S3 curation
(docs/synthesis.md), and the store deliberately exposes no link deletion.

Public API
----------
- ``NOTE_KIND``, ``LINK_TYPES``, ``DEFAULT_LINK_TYPE``, ``MAX_LINKS``
- ``DEFAULT_MAX_SOURCES``, ``DEFAULT_SOURCE_TOKENS``, ``LINK_CANDIDATES``
- ``NOTE_SYSTEM_PROMPT``, ``NOTE_RESPONSE_SCHEMA``, ``MAX_NOTE_ATTEMPTS``
- ``class NoteError(RuntimeError)``: this keyword could not be synthesized
- ``synthesize_note(kb, keyword_id, *, client, embedder=None,
  max_sources=DEFAULT_MAX_SOURCES, max_source_tokens=DEFAULT_SOURCE_TOKENS)
  -> int`` (note id)
- ``note_is_stale(kb, note_id, *, max_sources=..., max_source_tokens=...)
  -> bool``
- ``select_sources(kb, keyword_id, *, max_sources=..., max_source_tokens=...)
  -> list[dict]``: the deterministic selection, shared with staleness and
  with consolidation's demand ranking.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from wikirag.config import KnowledgeBase
from wikirag.ingestion.chunking import estimate_tokens
from wikirag.llm.client import AllKeysExhausted, LLMClient
from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL, EmbedderLike, get_embedder

__all__ = [
    "DEFAULT_LINK_TYPE",
    "DEFAULT_MAX_SOURCES",
    "DEFAULT_SOURCE_TOKENS",
    "LINK_CANDIDATES",
    "LINK_TYPES",
    "MAX_LINKS",
    "MAX_NOTE_ATTEMPTS",
    "NOTE_KIND",
    "NOTE_RESPONSE_SCHEMA",
    "NOTE_SYSTEM_PROMPT",
    "NoteError",
    "note_is_stale",
    "select_sources",
    "synthesize_note",
]

#: The note subject retrieval.verify.trigger_consolidation looks up. Frozen.
NOTE_KIND = "keyword"

#: Chunks fed to one synthesis, and the estimated token ceiling for them.
DEFAULT_MAX_SOURCES = 12
DEFAULT_SOURCE_TOKENS = 6000

#: Existing keywords shown as link candidates, and links kept from a reply.
LINK_CANDIDATES = 24
MAX_LINKS = 8

#: Link vocabulary (note_links.type, D10). Anything else is normalized to
#: DEFAULT_LINK_TYPE: a made-up relation is still a real, grounded link.
LINK_TYPES: tuple[str, ...] = (
    "related",
    "broader",
    "narrower",
    "part_of",
    "requires",
    "contrast",
)
DEFAULT_LINK_TYPE = "related"

MAX_NOTE_ATTEMPTS = 2

# The material the shortlist is computed from is truncated: a keyword with a
# hundred chunks would otherwise be encoded in full to rank two dozen names.
_SHORTLIST_MATERIAL_CHARS = 4000

NOTE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "integer"}},
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "type": {"type": "string", "enum": list(LINK_TYPES)},
                    "source": {"type": "integer"},
                },
                "required": ["concept", "type"],
            },
        },
    },
    "required": ["text", "sources", "links"],
}

NOTE_SYSTEM_PROMPT = f"""You write encyclopedia-style notes for a knowledge base wiki, one note per concept.

You receive the CONCEPT, its SOURCES as numbered entries ([1], [2], ...) drawn from the knowledge base, and CANDIDATE CONCEPTS: concepts that already exist in the wiki.

Return a JSON object with three fields:

- "text": the note. Explain what the concept is, how it works and how it relates to the rest of the material, in your own words, as continuous prose. Aim for 120-250 words, in the language of the sources. Say only what the sources support; do not copy or quote text from them, and do not repeat their wording sentence by sentence. Refer to the sources by their number only, and only where a reader needs the pointer.
- "sources": the numbers of the entries the note is actually based on. At least one; a note supported by nothing is not wanted.
- "links": at most {MAX_LINKS} connections to other concepts, each an object with "concept" (a name copied exactly from CANDIDATE CONCEPTS), "type" (one of: {", ".join(LINK_TYPES)}) and "source" (the number of the entry that supports the connection). Choose only names present in CANDIDATE CONCEPTS — never invent a concept, never link the note to itself, and link only where the sources show a real relation. An empty array is a valid answer.

Output the JSON object only, no markdown fences, no commentary."""


class NoteError(RuntimeError):
    """A keyword could not be turned into a note (no material, bad reply)."""


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


# ------------------------------------------------------------------- sources


def select_sources(
    kb: KnowledgeBase,
    keyword_id: int,
    *,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_source_tokens: int = DEFAULT_SOURCE_TOKENS,
) -> list[dict[str, Any]]:
    """The keyword's chunks a note is built from, deterministically.

    Store order (doc_id, ordinal), capped at ``max_sources`` chunks and at
    ``max_source_tokens`` estimated tokens. The first chunk is always kept,
    so an oversized chunk yields a one-source note instead of none.
    """
    return _cap(kb.store.chunks_for_keyword(keyword_id), max_sources, max_source_tokens)


def _cap(
    chunks: Sequence[Mapping[str, Any]], max_sources: int, max_source_tokens: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    budget = 0
    for chunk in chunks[: max(1, max_sources)]:
        text = str(chunk["text"])
        tokens = int(chunk["token_count"] or 0) or estimate_tokens(text)
        if selected and budget + tokens > max_source_tokens:
            break
        selected.append(dict(chunk))
        budget += tokens
    return selected


def note_is_stale(
    kb: KnowledgeBase,
    note_id: int,
    *,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_source_tokens: int = DEFAULT_SOURCE_TOKENS,
) -> bool:
    """Whether the note's source set no longer matches its keyword's chunks.

    The honest staleness test of docs/synthesis.md: recompute the selection
    :func:`synthesize_note` would use today and compare it with what the
    note recorded. Tagging a new chunk, re-tagging, and forgetting a
    document all change that set; nothing else does. A note whose subject
    is not a keyword (an L1 summary, the overview) is never stale by this
    rule, since it is not built from a keyword's chunks.
    """
    note = kb.store.get_note(note_id)
    if note is None:
        raise ValueError(f"no note {note_id}")
    keyword_id = note["keyword_id"]
    if note["kind"] != NOTE_KIND or keyword_id is None:
        return False
    current = {
        str(chunk["id"])
        for chunk in select_sources(
            kb,
            int(keyword_id),
            max_sources=max_sources,
            max_source_tokens=max_source_tokens,
        )
    }
    return current != set(kb.store.note_sources(note_id))


# -------------------------------------------------------------------- prompt


def _document_titles(kb: KnowledgeBase, chunks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    titles: dict[str, str] = {}
    for doc_id in {str(chunk["doc_id"]) for chunk in chunks}:
        document = kb.store.get_document(doc_id)
        titles[doc_id] = str((document or {}).get("title") or doc_id)
    return titles


def _build_user_message(
    kb: KnowledgeBase,
    name: str,
    sources: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    titles = _document_titles(kb, sources)
    blocks = [
        f"[{number}] {titles[str(chunk['doc_id'])]}\n{str(chunk['text']).strip()}"
        for number, chunk in enumerate(sources, start=1)
    ]
    parts = [f"CONCEPT: {name}", "SOURCES:\n" + "\n\n".join(blocks)]
    if candidates:
        names = "\n".join(f"- {candidate['name']}" for candidate in candidates)
        parts.append("CANDIDATE CONCEPTS:\n" + names)
    else:
        parts.append("CANDIDATE CONCEPTS:\n(none yet — return an empty links array)")
    return "\n\n".join(parts)


def _shortlist(
    kb: KnowledgeBase,
    keyword: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    embedder: EmbedderLike | None,
) -> list[dict[str, Any]]:
    """Existing keywords closest to this note's material, at most LINK_CANDIDATES.

    The encoder is resolved only when there is something to rank: on a KB
    with a single keyword, loading an embedding model would buy nothing.
    """
    others = [k for k in kb.store.list_keywords() if int(k["id"]) != int(keyword["id"])]
    if not others:
        return []
    if len(others) <= LINK_CANDIDATES:
        return others
    if embedder is None:
        model = kb.store.config_get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        embedder = get_embedder(model)
    material = "\n".join(
        [str(keyword["name"])] + [str(chunk["text"]) for chunk in sources]
    )[:_SHORTLIST_MATERIAL_CHARS]
    query = embedder.encode_query(material)
    matrix = embedder.encode_passages([str(k["name"]) for k in others])
    scores = matrix @ query
    ranked = sorted(range(len(others)), key=lambda i: (-float(scores[i]), i))
    return [others[i] for i in ranked[:LINK_CANDIDATES]]


# --------------------------------------------------------------- reply -> rows


def _parse_reply(text: str) -> dict[str, Any]:
    data = json.loads(_JSON_FENCE.sub("", text or "").strip())
    if not isinstance(data, dict):
        raise json.JSONDecodeError("not a JSON object", str(text), 0)
    return data


def _cited_chunk_ids(
    data: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Source numbers the model returned, mapped back to chunk IDs."""
    cited: list[str] = []
    raw = data.get("sources")
    for number in raw if isinstance(raw, (list, tuple)) else []:
        chunk_id = _chunk_for_number(number, sources)
        if chunk_id is not None and chunk_id not in cited:
            cited.append(chunk_id)
    return cited


def _chunk_for_number(number: Any, sources: Sequence[Mapping[str, Any]]) -> str | None:
    if isinstance(number, bool) or not isinstance(number, (int, float, str)):
        return None
    try:
        index = int(number)
    except (TypeError, ValueError):
        return None
    if 1 <= index <= len(sources):
        return str(sources[index - 1]["id"])
    return None


def _resolve_links(
    data: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    known_targets: set[int],
) -> list[dict[str, Any]]:
    """Links to shortlisted keywords only; everything else is dropped.

    This is where "orphans are impossible by construction" is enforced: a
    target is a row from the shortlist, matched by canonical name, so a
    hallucinated concept has nowhere to resolve to.
    """
    by_name = {str(c["canonical_name"]): int(c["id"]) for c in candidates}
    links: list[dict[str, Any]] = []
    seen = set(known_targets)
    raw = data.get("links")
    for item in raw if isinstance(raw, (list, tuple)) else []:
        if not isinstance(item, Mapping):
            continue
        concept = item.get("concept")
        if not isinstance(concept, str):
            continue
        target = by_name.get(concept.strip().lower())
        if target is None or target in seen:
            continue
        seen.add(target)
        link_type = item.get("type")
        links.append(
            {
                "to_keyword_id": target,
                "type": link_type if link_type in LINK_TYPES else DEFAULT_LINK_TYPE,
                "grounding_chunk_id": _chunk_for_number(item.get("source"), sources),
            }
        )
        if len(links) == MAX_LINKS:
            break
    return links


# ------------------------------------------------------------------ synthesis


def synthesize_note(
    kb: KnowledgeBase,
    keyword_id: int,
    *,
    client: LLMClient,
    embedder: EmbedderLike | None = None,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_source_tokens: int = DEFAULT_SOURCE_TOKENS,
) -> int:
    """Build or refresh the L2 note for one keyword; returns its note ID.

    One LLM call (retried once on an unusable reply): numbered sources in,
    new prose plus the numbers it used out. The note is stored with
    ``kind="keyword"``, its selection recorded in ``note_sources``, and the
    links it chose — resolved against an embedding shortlist of existing
    keywords — appended to ``note_links``. Writing the note clears its
    stale flag (the store does that on upsert).

    Raises :class:`NoteError` when the keyword has no tagged chunks or the
    model returns nothing usable after :data:`MAX_NOTE_ATTEMPTS` attempts.
    ``AllKeysExhausted`` propagates untouched: the caller stops the batch.
    """
    store = kb.store
    keyword = store.get_keyword(keyword_id=keyword_id)
    if keyword is None:
        raise ValueError(f"no keyword {keyword_id}")
    sources = select_sources(
        kb, keyword_id, max_sources=max_sources, max_source_tokens=max_source_tokens
    )
    if not sources:
        raise NoteError(f"keyword {keyword['name']!r} has no tagged chunks to build on")

    candidates = _shortlist(kb, keyword, sources, embedder)
    user_message = _build_user_message(kb, str(keyword["name"]), sources, candidates)

    data: dict[str, Any] | None = None
    text = ""
    for attempt in range(1, MAX_NOTE_ATTEMPTS + 1):
        try:
            result = client.generate(
                system=NOTE_SYSTEM_PROMPT,
                user=user_message,
                json_schema=NOTE_RESPONSE_SCHEMA,
            )
            data = _parse_reply(result.text)
        except AllKeysExhausted:
            raise
        except json.JSONDecodeError:
            if attempt < MAX_NOTE_ATTEMPTS:
                continue
            raise NoteError(
                f"unusable reply for keyword {keyword['name']!r}: not JSON"
            ) from None
        candidate_text = data.get("text")
        text = candidate_text.strip() if isinstance(candidate_text, str) else ""
        if text:
            break
        if attempt == MAX_NOTE_ATTEMPTS:
            raise NoteError(f"empty note for keyword {keyword['name']!r}")

    assert data is not None
    cited = _cited_chunk_ids(data, sources)
    if not cited:
        # Grounding rule: prose the model could not tie to any of the material
        # it was given is not a note, it is a guess.
        raise NoteError(f"ungrounded note for keyword {keyword['name']!r}: no source cited")

    with store.transaction():
        note_id = store.upsert_note(
            kind=NOTE_KIND,
            keyword_id=keyword_id,
            text=text,
            sources=[str(chunk["id"]) for chunk in sources],
        )
        known = {int(row["to_keyword_id"]) for row in store.note_links_from(note_id)}
        links = _resolve_links(data, candidates, sources, known)
        if links:
            store.add_note_links(note_id, links)
    return note_id
