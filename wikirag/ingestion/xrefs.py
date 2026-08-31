"""Native cross-reference extraction. Pattern-based, no LLM (D10).

Prose has a native graph the way code has imports: explicit textual
cross-references ("see Chapter 5", "cfr. ArM5 p.83", "as described in
The Mysteries"). This stage extracts them from chunks into the xrefs
table with high precision and zero API cost — the only graph materialized
eagerly at ingest.

Resolution is best-effort: a raw reference is stored always; to_doc_id /
to_section_id are filled when the target matches a known document title,
section heading, or page mapping. Unresolved xrefs stay useful for the
agent (R3) as literal follow-up cues.

Pattern set starts small (see/cfr/chapter/page + known doc abbreviations,
per-KB extendable via kb_config) and grows from observed corpus forms.

Precision policy
----------------
A false xref pollutes the graph permanently, so every pattern demands at
least two independent signals. Nothing is emitted from a bare number, a
bare unit word, or a quoted phrase on its own.

1. **Cued locator** — a citation cue (`see`, `see also`, `cf.`, `cfr.`,
   `vedi`, `si veda`, `as described in`, `come descritto in`, ...), an
   optional article, a unit word, and a number:
   `see Chapter 5`, `cfr. sezione 2.1`, `vedi p. 83`, `cf. pp. 12-15`.
   The number must follow the unit word immediately, which is what keeps
   "the sea chapter of his life" and "he could see 5 people" out.
2. **Named locator** — a known document name (title, or an abbreviation
   from kb_config) immediately followed by a locator, with or without a
   cue: `ArM5 p.83`, `The Mysteries, chapter 4`. A known name touching a
   locator is a citation by construction, so no cue is required here.
3. **Cued document** — a cue followed by a known document name:
   `as described in The Mysteries`, `see ArM5`. Only names the KB already
   knows are matched; free-form title capture is not attempted.
4. **Quoted title** — a quoted string (`"..."`, `“...”`, `«...»`) that
   either resolves to a known document title / section heading, or is
   preceded by a cue (`see the chapter "Raw Vis"`). An unresolved quote
   with no cue is ordinary quoted prose and is dropped.

Overlapping matches are resolved by pattern specificity (named locator >
cued locator > cued document > quoted title), so `vedi ArM5 p.83` yields
one page xref, not a page xref plus a document xref.

Kind taxonomy (xrefs.kind)
--------------------------
- `chapter`  — a numbered chapter (`see Chapter 5`, `vedi capitolo 1`).
- `section`  — a numbered section/paragraph, or a named section heading
               (`cfr. sezione 2.1`, `see the chapter "Raw Vis"`).
- `page`     — a page or page range, optionally qualified by a document
               name (`p. 83`, `pp. 12-15`, `ArM5 p.83`).
- `document` — a work referenced as a whole, with no locator
               (`as described in The Mysteries`, `see ArM5`).

Resolution rules
----------------
- `to_doc_id`: set when the reference names a known document — by title
  or by a kb_config abbreviation, matched case-insensitively. A numbered
  chapter/section with no named target refers to the citing document, so
  it resolves against that document's own sections.
- `to_section_id`: set when a chapter/section number matches exactly one
  section of the target document (heading starting with that number,
  optionally prefixed by the unit word), or when a quoted heading matches
  exactly one known section title (same document preferred). Ambiguous
  matches resolve to nothing rather than to a guess.
- Page references never resolve to a section: the schema stores no page
  mapping (sections carry no page numbers), so a page ref resolves at
  most to its document.
- Unresolved references keep `(None, None)` and are still stored.

Configuration
-------------
`kb_config["xref_abbreviations"]`: a JSON object mapping an abbreviation
to a document title or a doc_id, e.g. `{"ArM5": "Ars Magica Fifth
Edition"}`. Empty by default. An abbreviation whose target is not (yet)
registered still produces xrefs, unresolved.

Public API
----------
- XREF_ABBREVIATIONS_KEY: the kb_config key holding the abbreviation map
- load_abbreviations(kb) -> dict[str, str]
- extract_xrefs(kb, doc_id) -> list[dict]   (pure read; no writes)
- apply_xrefs(kb, doc_id) -> int            (idempotent: replace + count)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoids importing config at runtime (store -> config -> here)
    from wikirag.config import KnowledgeBase

__all__ = [
    "XREF_ABBREVIATIONS_KEY",
    "load_abbreviations",
    "extract_xrefs",
    "apply_xrefs",
]

XREF_ABBREVIATIONS_KEY = "xref_abbreviations"

# A document name shorter than this is too collision-prone to match in prose.
_MIN_NAME_LEN = 3

# Citation cues, English and Italian. Longest alternatives first: regex
# alternation is leftmost-first, so "see also" must precede "see".
_CUE = (
    r"see\s+also|see|cf\.|cfr\.|"
    r"as\s+described\s+in|as\s+detailed\s+in|as\s+explained\s+in|"
    r"described\s+in|detailed\s+in|discussed\s+in|refer\s+to|"
    r"vedi\s+anche|vedi|vedasi|si\s+veda|"
    r"come\s+descritto\s+in|come\s+spiegato\s+in"
)
# Articles/prepositions tolerated between the cue and the target.
_FILLER = (
    r"(?:the|a|il|lo|la|le|i|gli|al|allo|alla|ai|nel|nello|nella|"
    r"in|di|del|dello|della|su|sul)\s+"
)

_CHAPTER_WORDS = r"chapters|chapter|chap\.|ch\.|capitoli|capitolo|cap\."
_SECTION_WORDS = (
    r"sections|section|sect\.|sec\.|sezioni|sezione|sez\.|"
    r"paragraphs|paragraph|paragrafi|paragrafo|par\."
)
_PAGE_WORDS = r"pp\.|pagg\.|pages|page|pagine|pagina|pag\.|p\."
_UNIT = rf"{_PAGE_WORDS}|{_CHAPTER_WORDS}|{_SECTION_WORDS}"

_NUM = r"\d{1,4}(?:\.\d{1,3})*"
_RANGE = rf"{_NUM}(?:\s*[-–—]\s*\d{{1,4}})?"

_CHAPTER_SET = {"chapters", "chapter", "chap", "ch", "capitoli", "capitolo", "cap"}
_PAGE_SET = {"pp", "pagg", "pages", "page", "pagine", "pagina", "pag", "p"}

# Pattern 1: cue + optional article + unit + number.
_CUED_LOCATOR_RE = re.compile(
    rf"\b(?:{_CUE})\s*(?:{_FILLER}){{0,2}}(?P<unit>{_UNIT})\s*(?P<num>{_RANGE})",
    re.IGNORECASE,
)
# A locator on its own; used as the optional tail of a named reference.
_LOCATOR_TAIL = rf"(?:\s*,)?\s*(?P<unit>{_UNIT})\s*(?P<num>{_RANGE})"

_QUOTED_RE = re.compile(
    r"\"(?P<d>[^\"\n]{2,120})\"|“(?P<c>[^”\n]{2,120})”|«(?P<g>[^»\n]{2,120})»"
)
# Anchored at the end of the text preceding a quote: cue, optional article,
# optional unit word ("see the chapter \"Raw Vis\"").
_CUE_BEFORE_QUOTE_RE = re.compile(
    rf"\b(?:{_CUE})\s*(?:{_FILLER}){{0,2}}"
    rf"(?:(?:{_CHAPTER_WORDS}|{_SECTION_WORDS})\s+)?[\s(]*$",
    re.IGNORECASE,
)

_WS = re.compile(r"\s+")

# Priorities: lower wins when two matches overlap (most specific first).
_P_NAMED_LOCATOR = 0
_P_CUED_LOCATOR = 1
_P_CUED_DOCUMENT = 2
_P_QUOTED = 3


def _norm(text: str) -> str:
    """Case- and whitespace-insensitive key for title matching."""
    return _WS.sub(" ", text).strip().strip(".,;:!?").lower()


def _kind_for_unit(unit: str) -> str:
    word = unit.strip().rstrip(".").lower()
    if word in _PAGE_SET:
        return "page"
    if word in _CHAPTER_SET:
        return "chapter"
    return "section"


@dataclass(frozen=True)
class _Candidate:
    """One pattern hit inside a chunk, before overlap resolution."""

    start: int
    end: int
    priority: int
    raw_ref: str
    kind: str
    to_doc_id: str | None
    to_section_id: str | None


def load_abbreviations(kb: KnowledgeBase) -> dict[str, str]:
    """Per-KB abbreviation map from kb_config; empty when unset.

    A malformed value is raised rather than ignored: silently dropping the
    map would make an ingest quietly lose every abbreviated xref.
    """
    raw = kb.store.config_get(XREF_ABBREVIATIONS_KEY)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"kb_config[{XREF_ABBREVIATIONS_KEY!r}] is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(
            f"kb_config[{XREF_ABBREVIATIONS_KEY!r}] must be a JSON object of "
            "abbreviation -> document title or doc_id"
        )
    return {k.strip(): v.strip() for k, v in data.items() if k.strip()}


class _CorpusIndex:
    """What the KB knows about targets: document names and section titles."""

    def __init__(self, kb: KnowledgeBase) -> None:
        store = kb.store
        documents = store.list_documents()
        doc_ids = {d["id"] for d in documents}
        by_title = {_norm(d["title"]): d["id"] for d in documents if d["title"]}

        # Surface form -> doc_id (None when the target is not registered yet).
        names: dict[str, str | None] = {}
        for doc in documents:
            if doc["title"] and len(doc["title"].strip()) >= _MIN_NAME_LEN:
                names[doc["title"].strip()] = doc["id"]
        for abbr, target in load_abbreviations(kb).items():
            if len(abbr) < _MIN_NAME_LEN:
                continue
            names[abbr] = target if target in doc_ids else by_title.get(_norm(target))

        self.doc_titles = by_title
        self.names = names
        self.sections: dict[str, list[dict[str, Any]]] = {
            doc["id"]: store.get_sections(doc["id"]) for doc in documents
        }
        self.section_titles: dict[str, list[tuple[str, str]]] = {}
        for doc_id, sections in self.sections.items():
            for section in sections:
                if section["title"]:
                    self.section_titles.setdefault(_norm(section["title"]), []).append(
                        (doc_id, section["id"])
                    )

        alternation = "|".join(
            re.escape(name) for name in sorted(self.names, key=len, reverse=True)
        )
        # Lookarounds rather than \b: names may end in punctuation ("ArM5.").
        self.named_locator_re = (
            re.compile(
                rf"(?<![\w])(?P<name>{alternation})(?![\w]){_LOCATOR_TAIL}",
                re.IGNORECASE,
            )
            if alternation
            else None
        )
        self.cued_document_re = (
            re.compile(
                rf"\b(?:{_CUE})\s*(?:{_FILLER}){{0,2}}"
                rf"(?<![\w])(?P<name>{alternation})(?![\w])",
                re.IGNORECASE,
            )
            if alternation
            else None
        )

    def doc_for_name(self, name: str) -> str | None:
        key = _norm(name)
        for surface, doc_id in self.names.items():
            if _norm(surface) == key:
                return doc_id
        return None

    def numbered_section(self, doc_id: str | None, number: str) -> str | None:
        """The section of `doc_id` whose heading starts with `number`.

        Headings are matched, not ordinals: a document's section ordinals are
        reading order, which is not chapter numbering. Ambiguity resolves to
        None — a wrong edge is worse than a missing one.
        """
        if doc_id is None:
            return None
        head = number.split("-")[0].split("–")[0].split("—")[0].strip()
        pattern = re.compile(
            rf"^\W*(?:(?:{_CHAPTER_WORDS}|{_SECTION_WORDS}|parte|part)\s*)?"
            rf"0*{re.escape(head)}(?![\d])",
            re.IGNORECASE,
        )
        hits = [
            s["id"]
            for s in self.sections.get(doc_id, [])
            if s["title"] and pattern.match(s["title"])
        ]
        return hits[0] if len(hits) == 1 else None

    def resolve_title(
        self, title: str, from_doc_id: str
    ) -> tuple[str, str | None, str | None] | None:
        """Match a bare/quoted title against known documents and headings."""
        key = _norm(title)
        if key in self.doc_titles:
            return "document", self.doc_titles[key], None
        hits = self.section_titles.get(key, [])
        same_doc = [h for h in hits if h[0] == from_doc_id]
        chosen = same_doc or hits
        if len(chosen) == 1:
            doc_id, section_id = chosen[0]
            return "section", doc_id, section_id
        return None


def _candidates(chunk: dict[str, Any], index: _CorpusIndex) -> list[_Candidate]:
    text: str = chunk["text"]
    from_doc_id: str = chunk["doc_id"]
    found: list[_Candidate] = []

    def raw(start: int, end: int) -> str:
        return _WS.sub(" ", text[start:end]).strip()

    if index.named_locator_re is not None:
        for m in index.named_locator_re.finditer(text):
            kind = _kind_for_unit(m.group("unit"))
            to_doc = index.doc_for_name(m.group("name"))
            to_section = (
                None
                if kind == "page"
                else index.numbered_section(to_doc, m.group("num"))
            )
            found.append(
                _Candidate(
                    m.start(),
                    m.end(),
                    _P_NAMED_LOCATOR,
                    raw(m.start(), m.end()),
                    kind,
                    to_doc,
                    to_section,
                )
            )

    for m in _CUED_LOCATOR_RE.finditer(text):
        kind = _kind_for_unit(m.group("unit"))
        to_section = (
            None
            if kind == "page"
            else index.numbered_section(from_doc_id, m.group("num"))
        )
        # An unqualified chapter/section number belongs to the citing document,
        # but only claim that when the heading actually exists there.
        to_doc = from_doc_id if to_section else None
        found.append(
            _Candidate(
                m.start(),
                m.end(),
                _P_CUED_LOCATOR,
                raw(m.start(), m.end()),
                kind,
                to_doc,
                to_section,
            )
        )

    if index.cued_document_re is not None:
        for m in index.cued_document_re.finditer(text):
            found.append(
                _Candidate(
                    m.start(),
                    m.end(),
                    _P_CUED_DOCUMENT,
                    raw(m.start(), m.end()),
                    "document",
                    index.doc_for_name(m.group("name")),
                    None,
                )
            )

    for m in _QUOTED_RE.finditer(text):
        quoted = next(g for g in m.groups() if g is not None)
        resolved = index.resolve_title(quoted, from_doc_id)
        cue = _CUE_BEFORE_QUOTE_RE.search(text[: m.start()])
        if resolved is None and cue is None:
            continue  # ordinary quoted prose, not a reference
        start = cue.start() if cue else m.start()
        kind, to_doc, to_section = resolved or ("document", None, None)
        found.append(
            _Candidate(
                start,
                m.end(),
                _P_QUOTED,
                raw(start, m.end()),
                kind,
                to_doc,
                to_section,
            )
        )
    return found


def _select(found: list[_Candidate]) -> list[_Candidate]:
    """Keep the most specific of any overlapping matches, in text order."""
    kept: list[_Candidate] = []
    for cand in sorted(found, key=lambda c: (c.priority, c.start, -c.end)):
        if any(cand.start < k.end and k.start < cand.end for k in kept):
            continue
        kept.append(cand)
    return sorted(kept, key=lambda c: c.start)


def extract_xrefs(kb: KnowledgeBase, doc_id: str) -> list[dict[str, Any]]:
    """Cross-references found in a document's chunks, in reading order.

    Rows are store-shaped ({from_chunk_id, raw_ref, to_doc_id, to_section_id,
    kind}) and ready for Store.add_xrefs. Nothing is written here; identical
    references repeated inside one chunk collapse to a single row.
    """
    index = _CorpusIndex(kb)
    rows: list[dict[str, Any]] = []
    for chunk in kb.store.get_document_chunks(doc_id):
        seen: set[tuple[str, str | None, str | None, str]] = set()
        for cand in _select(_candidates(chunk, index)):
            key = (cand.raw_ref, cand.to_doc_id, cand.to_section_id, cand.kind)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "from_chunk_id": chunk["id"],
                    "raw_ref": cand.raw_ref,
                    "to_doc_id": cand.to_doc_id,
                    "to_section_id": cand.to_section_id,
                    "kind": cand.kind,
                }
            )
    return rows


def apply_xrefs(kb: KnowledgeBase, doc_id: str) -> int:
    """Re-extract a document's xrefs and replace its stored ones; count them.

    The ingest stage entry point. Delete and insert share one transaction, so
    a retry never observes (or leaves) a half-replaced reference set.
    """
    rows = extract_xrefs(kb, doc_id)
    with kb.store.transaction():
        kb.store.delete_xrefs(doc_id)
        kb.store.add_xrefs(rows)
    return len(rows)
