"""Vault export: render the human-facing markdown view from the store.

The vault is a *product*, not a store: one file per keyword note, per
document summary, plus overview.md; wikilinks rendered as [[name]] for
Obsidian. Deleting vault/ loses nothing; export is idempotent and fast
enough to re-run after every ingest.

Notes carry their provenance (note_sources), so export renders footnote
-style source references per note — documents and sections, never chunk
text (the vault is reading material, not a dump).

Two-way sync (user edits flowing back into the store) is explicitly out of
scope for now — docs/decisions.md.

Layout
------
    vault/
      overview.md               the atlas note (kind "overview")
      concepts/<Keyword>.md     keyword notes (kind "keyword")
      documents/<Title>.md      per-document summaries (kind "summary")
      notes/<kind>-<subject>.md any other kind (hub pages, future kinds)

Only those four locations are managed: `clean=True` deletes the `*.md`
files there that no subject in the store claims any more, and nothing
else, so attachments or hand-written files elsewhere in the vault survive.

Filenames
---------
A file stem is its subject's display name with the characters Obsidian and
Windows reject (``\\ / : * ? " < > | # ^ [ ]``) replaced by ``-``,
whitespace collapsed, leading/trailing dots and spaces dropped, truncated
to 80 characters; an empty or Windows-reserved result becomes ``untitled``
resp. gets a ``-note`` suffix.

Stems are claimed from one vault-wide table, compared case-insensitively
(case-insensitive filesystems, and Obsidian resolves ``[[name]]`` by
basename across folders — one file per stem keeps every link unambiguous).
Two keywords whose names sanitize alike therefore cannot share a file: the
first claimant keeps the plain stem and the next gets ``<stem>-<id>``
appended (its keyword id, doc id or note id). Claims run in a fixed order —
``overview``, then every keyword by id, then documents by id, then the
remaining notes by note id — so a stem is stable across runs and only
changes if the name itself does. Keywords claim their stem whether or not
they have a note yet, which is what makes a link to a not-yet-synthesized
concept resolve the moment its note appears.

Wikilinks render as ``[[stem]]``, or ``[[stem|display name]]`` when
sanitizing changed the name, so the rendered text stays human while the
link resolves to a real file.

API
---
- export_vault(kb, *, clean=False) -> ExportReport
- ExportReport(vault_dir, written, updated, unchanged, removed)
- sanitize(name) -> str
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wikirag.config import KnowledgeBase

CONCEPTS_DIR = "concepts"
DOCUMENTS_DIR = "documents"
OTHER_DIR = "notes"
OVERVIEW_FILE = "overview.md"
MANAGED_DIRS = (CONCEPTS_DIR, DOCUMENTS_DIR, OTHER_DIR)

MAX_STEM = 80

# Illegal in Obsidian note names and/or on Windows/macOS filesystems.
_ILLEGAL = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]')
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

STALE_CALLOUT = (
    "> [!warning] Stale\n"
    "> The sources this note was built from have changed since it was written."
)


def sanitize(name: str) -> str:
    """Turn a display name into a filesystem- and Obsidian-safe file stem."""
    cleaned = " ".join(_ILLEGAL.sub("-", name).split())
    cleaned = cleaned.strip(" .")[:MAX_STEM].strip(" .")
    if not cleaned:
        return "untitled"
    # Windows refuses these stems whatever the extension; the vault may be
    # synced to a Windows Obsidian.
    return f"{cleaned}-note" if cleaned.upper() in _RESERVED else cleaned


@dataclass
class ExportReport:
    """What one export did with the vault. Every path is absolute."""

    vault_dir: Path
    written: list[Path] = field(default_factory=list)
    updated: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)

    @property
    def files(self) -> int:
        """Files the vault holds for this store after the export."""
        return len(self.written) + len(self.updated) + len(self.unchanged)


class _Stems:
    """Vault-wide file-stem table; the first claimant keeps the plain stem."""

    def __init__(self) -> None:
        self._taken: set[str] = set()

    def claim(self, label: str, unique: str) -> str:
        base = sanitize(label)
        stem = base
        attempt = 0
        while stem.casefold() in self._taken:
            attempt += 1
            suffix = unique if attempt == 1 else f"{unique}-{attempt}"
            stem = sanitize(f"{base}-{suffix}")
        self._taken.add(stem.casefold())
        return stem


def export_vault(kb: KnowledgeBase, *, clean: bool = False) -> ExportReport:
    """Render `kb.vault_dir` from the store; return what changed.

    Without `clean` nothing is ever deleted: the export only writes files
    whose content differs from what is on disk (so re-running is a no-op,
    and file timestamps stay meaningful). With `clean=True` the managed
    markdown files that no note claims any more are removed as well.
    """
    store = kb.store
    vault = kb.vault_dir
    report = ExportReport(vault_dir=vault)

    keywords = {int(k["id"]): k for k in store.list_keywords()}
    documents = {str(d["id"]): d for d in store.list_documents()}

    stems = _Stems()
    # "overview" is reserved first: the atlas has a fixed path, and a second
    # file with that basename would make [[overview]] ambiguous.
    stems.claim("overview", "note")
    keyword_stems = {
        kid: stems.claim(str(keywords[kid]["name"]), str(kid)) for kid in sorted(keywords)
    }

    context = _Context(kb=kb, keyword_stems=keyword_stems, keywords=keywords, documents=documents)
    planned: dict[Path, str] = {}
    for note in _sorted_notes(_all_notes(kb)):
        path = _path_for(note, vault, stems, context)
        planned[path] = _render(note, context)

    for path in sorted(planned):
        text = planned[path]
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == text:
            report.unchanged.append(path)
            continue
        _atomic_write(path, text)
        (report.updated if current is not None else report.written).append(path)

    if clean:
        for path in _managed_files(vault):
            if path not in planned:
                path.unlink()
                report.removed.append(path)
        for name in MANAGED_DIRS:
            directory = vault / name
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    _fsync_dirs({path.parent for path in report.written} | {path.parent for path in report.removed})
    return report


# ------------------------------------------------------------------ note input


def _all_notes(kb: KnowledgeBase) -> list[dict[str, Any]]:
    """Every note in the store, whatever its kind.

    The frozen store API addresses notes by subject (kind, keyword_id,
    doc_id), which cannot enumerate kinds the exporter does not already know
    about — and dropping an unknown kind would silently lose a product the
    user paid a model to write. So the notes table is read directly, through
    a second connection: SQLite is a concurrent reader (WAL), and this is a
    read of columns documented in docs/storage.md, not a second writer.
    """
    db = sqlite3.connect(str(kb.db_path))
    try:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT id, kind, keyword_id, doc_id, text, updated_at, stale FROM notes ORDER BY id"
        ).fetchall()
    finally:
        db.close()
    return [dict(row) for row in rows]


#: Claim order: fixed so stems stay stable across runs (see module docstring).
_KIND_ORDER = {"overview": 0, "keyword": 1, "summary": 2}


def _sorted_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        notes,
        key=lambda n: (
            _KIND_ORDER.get(str(n["kind"]), 3),
            n["keyword_id"] if n["keyword_id"] is not None else 0,
            str(n["doc_id"] or ""),
            int(n["id"]),
        ),
    )


@dataclass
class _Context:
    """Lookups shared by every rendered note."""

    kb: KnowledgeBase
    keyword_stems: dict[int, str]
    keywords: dict[int, dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    _sections: dict[str, dict[str, str]] = field(default_factory=dict)

    def document_title(self, doc_id: str | None) -> str:
        if doc_id is None:
            return "unknown document"
        document = self.documents.get(doc_id)
        if document is None:
            return doc_id
        return str(document["title"] or Path(str(document["filename"])).stem or doc_id)

    def section_title(self, doc_id: str | None, section_id: str | None) -> str | None:
        if doc_id is None or section_id is None:
            return None
        if doc_id not in self._sections:
            self._sections[doc_id] = {
                str(s["id"]): str(s["title"] or "")
                for s in self.kb.store.get_sections(doc_id)
            }
        return self._sections[doc_id].get(section_id) or None

    def link(self, keyword_id: int, name: str) -> str:
        """A wikilink to a keyword, aliased when its stem is not its name."""
        stem = self.keyword_stems.get(keyword_id) or sanitize(name)
        return f"[[{stem}]]" if stem == name else f"[[{stem}|{name}]]"


def _path_for(
    note: dict[str, Any], vault: Path, stems: _Stems, context: _Context
) -> Path:
    """Where a note's file goes; unknown kinds land in notes/ by their kind."""
    kind = str(note["kind"])
    keyword_id = note["keyword_id"]
    doc_id = note["doc_id"]
    if kind == "overview":
        return vault / OVERVIEW_FILE
    if kind == "keyword" and keyword_id is not None:
        return vault / CONCEPTS_DIR / f"{context.keyword_stems[int(keyword_id)]}.md"
    if kind == "summary" and doc_id is not None:
        stem = stems.claim(context.document_title(str(doc_id)), str(doc_id))
        return vault / DOCUMENTS_DIR / f"{stem}.md"
    subject = _subject_label(note, context)
    label = f"{kind} - {subject}" if subject else kind
    return vault / OTHER_DIR / f"{stems.claim(label, str(note['id']))}.md"


def _subject_label(note: dict[str, Any], context: _Context) -> str:
    keyword_id = note["keyword_id"]
    if keyword_id is not None:
        keyword = context.keywords.get(int(keyword_id))
        return str(keyword["name"]) if keyword else str(keyword_id)
    if note["doc_id"] is not None:
        return context.document_title(str(note["doc_id"]))
    return ""


# -------------------------------------------------------------------- rendering


def _yaml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _front_matter(fields: dict[str, Any]) -> str:
    lines = ["---"]
    lines += [f"{key}: {_yaml(value)}" for key, value in fields.items() if value is not None]
    lines.append("---")
    return "\n".join(lines)


def _title(note: dict[str, Any], context: _Context) -> str:
    kind = str(note["kind"])
    if kind == "overview":
        return "Overview"
    subject = _subject_label(note, context)
    if kind in ("keyword", "summary") or not subject:
        return subject or kind.capitalize()
    return f"{subject} ({kind})"


def _render(note: dict[str, Any], context: _Context) -> str:
    store = context.kb.store
    note_id = int(note["id"])
    kind = str(note["kind"])
    keyword_id = note["keyword_id"]
    doc_id = note["doc_id"]
    stale = bool(note["stale"])
    sources = store.note_sources(note_id)
    title = _title(note, context)

    fields: dict[str, Any] = {"kind": kind, "title": title, "note_id": note_id}
    if keyword_id is not None:
        keyword = context.keywords.get(int(keyword_id))
        fields["keyword"] = str(keyword["name"]) if keyword else None
        fields["keyword_id"] = int(keyword_id)
    if doc_id is not None:
        fields["doc_id"] = str(doc_id)
        fields["document"] = context.document_title(str(doc_id))
    fields["stale"] = stale
    fields["updated_at"] = str(note["updated_at"])
    fields["sources"] = len(sources)

    blocks = [_front_matter(fields)]
    text = str(note["text"]).strip()
    if not text.startswith("# "):
        blocks.append(f"# {title}")
    if stale:
        blocks.append(STALE_CALLOUT)
    if text:
        blocks.append(text)

    links = _link_lines(store.note_links_from(note_id), context)
    if links:
        blocks.append("\n".join(["## Links", "", *links]))
    citations = _source_lines(sources, context)
    if citations:
        blocks.append("\n".join(["## Sources", "", *citations]))
    return "\n\n".join(blocks) + "\n"


def _link_lines(links: list[dict[str, Any]], context: _Context) -> list[str]:
    lines: list[str] = []
    for link in links:
        rendered = context.link(int(link["to_keyword_id"]), str(link["to_keyword"]))
        line = f"- {link['type']}: {rendered}"
        if line not in lines:
            lines.append(line)
    return lines


def _source_lines(chunk_ids: list[str], context: _Context) -> list[str]:
    """Cite the documents and sections a note was built from, never chunks."""
    lines: list[str] = []
    for chunk in context.kb.store.get_chunks(chunk_ids):
        doc_id = chunk["doc_id"]
        title = context.document_title(doc_id)
        section = context.section_title(doc_id, chunk["section_id"])
        line = f"- {title} — {section}" if section else f"- {title}"
        if line not in lines:
            lines.append(line)
    return sorted(lines)


# ---------------------------------------------------------------------- on disk


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp file + fsync + replace (same discipline as vectors.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_dirs(directories: set[Path]) -> None:
    """Make the renames themselves durable, not just the file contents."""
    for directory in directories:
        if not directory.is_dir():
            continue
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _managed_files(vault: Path) -> list[Path]:
    """The markdown files this exporter owns, if the vault exists."""
    found: list[Path] = []
    overview = vault / OVERVIEW_FILE
    if overview.is_file():
        found.append(overview)
    for name in MANAGED_DIRS:
        directory = vault / name
        if directory.is_dir():
            found += sorted(p for p in directory.glob("*.md") if p.is_file())
    return found
