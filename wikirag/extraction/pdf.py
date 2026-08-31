"""PDF extraction: structure first, then clean text per section.

Ported concepts from v1 (they worked):
- PyMuPDF TOC as the primary structure source; font-size heuristics with
  per-page title aggregation as fallback.
- Boilerplate removal by repeated-line detection across pages (not fixed
  geometric crops — v1 learned this on sparse slide decks).

New in v2:
- Output is the `sections` structure consumed by the store: ordered,
  hierarchical (level from TOC depth), with page ranges — because chunk
  IDs derive from the structural path (docs/storage.md).
- Extraction quality is inspectable standalone (debug CLI like v1's
  pdf_tester) since everything downstream depends on it.

Open (docs/ingestion.md): ColPali-class visual indexing for layout-heavy
sourcebooks would bypass this module entirely for the index; extraction
would still run for the vault/synthesis side.

How structure is found
----------------------
1. TOC (`doc.get_toc`) when present: entries with an out-of-range page are
   dropped, document order is authoritative, level jumps are clamped so the
   tree stays well formed.
2. Otherwise font-size heuristics: spans at >= 80% of the largest size on a
   page become that page's heading (v1's per-page aggregation), yielding a
   flat level-1 sequence; consecutive pages under the same title are one
   section, not one per page. A document with a single font size has no font
   structure and gets none.
3. Otherwise a single synthetic section holding the whole text.

Text is assigned by *anchoring* each heading to the page line that repeats
its title (searched forward, within the heading page and the next one), so
several headings on one page still split that page's text; when the title
line is not found the section simply starts at its page. A section's text is
its own body only, children excluded. Text before the first heading becomes
a synthetic "Front matter" section, so nothing is silently dropped.

An entirely text-free PDF (scan without a text layer, blank file) yields
zero sections on purpose: that is the signal that extraction found nothing,
not something for the caller to chunk.

Public API
----------
- class Section: path, title, level, text, page_start, page_end
- class ExtractedDoc: title, page_count, sections
- extract(pdf_path) -> ExtractedDoc
- main(argv=None) -> int  — debug CLI, also `python -m wikirag.extraction.pdf`
"""

from __future__ import annotations

import argparse
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# A line is boilerplate only if it repeats IDENTICALLY on a substantial
# fraction of the pages: a section title repeated on a few consecutive slides
# stays well under the threshold and survives (v1 lesson: a fixed geometric
# crop deleted the single content line of sparse slides).
BOILERPLATE_PAGE_FRACTION = 0.3
BOILERPLATE_MIN_PAGES = 3

HEADING_FONT_RATIO = 0.8  # v1: spans at >= 80% of the largest size are titles
HEADING_MIN_CHARS = 4  # v1: len(text) > 3
HEADING_SEARCH_PAGE_SPAN = 1  # look for a heading's title line one page past its own

PREAMBLE_TITLE = "Front matter"

_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s*/\s*\d+|\s+of\s+\d+)?$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-zÀ-ÿ])-\n([a-zà-ÿ])")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


@dataclass
class Section:
    path: str        # dotted 1-based structural position: "3", "3.2", "3.2.1"; level == number of components
    title: str
    level: int       # 1 = top level
    text: str        # clean body text of this section only (children's text excluded)
    page_start: int  # 1-based, inclusive
    page_end: int


@dataclass
class ExtractedDoc:
    title: str | None
    page_count: int
    sections: list[Section]  # document order (pre-order traversal)


@dataclass
class _Heading:
    level: int
    title: str
    page: int  # 1-based


def extract(pdf_path: Path | str) -> ExtractedDoc:
    """Extract the section tree and clean per-section text from a PDF."""
    path = Path(pdf_path)
    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        raw_pages = [_page_lines(page) for page in doc]
        boilerplate = _boilerplate_lines(raw_pages)
        lines = _content_lines(raw_pages, boilerplate)
        headings = _toc_headings(doc, page_count)
        if not headings:
            headings = _font_headings(doc, boilerplate)
        title = _document_title(doc)

    sections = _build_sections(
        headings, lines, page_count, fallback_title=title or path.stem
    )
    return ExtractedDoc(title=title, page_count=page_count, sections=sections)


# --------------------------------------------------------------------- text


def _page_lines(page: pymupdf.Page) -> list[str]:
    """Non-empty lines with an empty-string marker between text blocks.

    The markers survive into section text as blank lines: the chunker splits
    paragraphs on blank lines, so block boundaries are the paragraph signal
    extraction must not destroy.
    """
    lines: list[str] = []
    for block in page.get_text("blocks"):
        if block[6] != 0:  # image block
            continue
        block_lines = [ln.strip() for ln in block[4].split("\n") if ln.strip()]
        if not block_lines:
            continue
        lines.extend(block_lines)
        lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _boilerplate_lines(raw_pages: list[list[str]]) -> set[str]:
    """Header/footer lines repeated across pages, by exact match."""
    counts: Counter[str] = Counter()
    for lines in raw_pages:
        counts.update(set(lines) - {""})  # once per page; markers are not lines
    threshold = max(BOILERPLATE_MIN_PAGES, int(len(raw_pages) * BOILERPLATE_PAGE_FRACTION))
    return {line for line, count in counts.items() if count >= threshold}


def _content_lines(
    raw_pages: list[list[str]], boilerplate: set[str]
) -> list[tuple[int, str]]:
    """Flat (1-based page, line) stream with boilerplate and page numbers gone.

    Bare numeric lines are dropped as page numbers (v1 behaviour); this costs
    the occasional standalone number in a numeric table.
    """
    out: list[tuple[int, str]] = []
    for page_index, lines in enumerate(raw_pages, start=1):
        for line in lines:
            if not line:  # paragraph marker: keep one, never leading
                if out and out[-1][1]:
                    out.append((page_index, ""))
                continue
            if line in boilerplate or _PAGE_NUMBER_RE.match(line):
                continue
            out.append((page_index, line))
    return out


# ---------------------------------------------------------------- structure


def _document_title(doc: pymupdf.Document) -> str | None:
    title = (doc.metadata or {}).get("title") or ""
    title = _WS_RE.sub(" ", title).strip()
    return title or None


def _toc_headings(doc: pymupdf.Document, page_count: int) -> list[_Heading]:
    headings: list[_Heading] = []
    for entry in doc.get_toc(simple=True) or []:
        level, title, page = entry[0], entry[1], entry[2]
        title = _WS_RE.sub(" ", str(title)).strip()
        # Unresolved destinations come back as page -1 or out of range.
        if not title or not isinstance(page, int) or not 1 <= page <= page_count:
            continue
        headings.append(_Heading(level=max(1, int(level)), title=title, page=page))
    return headings


def _font_headings(doc: pymupdf.Document, boilerplate: set[str]) -> list[_Heading]:
    """Fallback: largest-font text per page, aggregated into one title.

    v1's per-page aggregation ("Chapter 1" + "The beginning" -> one heading);
    the result is a flat level-1 sequence, which is all a font signal supports.
    """
    per_page: list[tuple[int, list[tuple[float, str]]]] = []
    sizes: set[float] = set()
    for page_index, page in enumerate(doc, start=1):
        page_spans: list[tuple[float, str]] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                text = _WS_RE.sub(" ", text).strip()
                if not text:
                    continue
                size = round(max(span["size"] for span in line["spans"]), 1)
                sizes.add(size)
                page_spans.append((size, text))
        per_page.append((page_index, page_spans))

    # A single font size carries no structure: better one flat section than a
    # heading on every page.
    if len(sizes) < 2:
        return []

    threshold = max(sizes) * HEADING_FONT_RATIO
    headings: list[_Heading] = []
    for page_index, page_spans in per_page:
        parts: list[str] = []
        for size, text in page_spans:
            if size < threshold or len(text) < HEADING_MIN_CHARS:
                continue
            if text in boilerplate or text in parts:
                continue
            parts.append(text)
        if not parts:
            continue
        title = " - ".join(parts)
        # A slide title repeated over consecutive slides is one division, not
        # one per page: merging keeps sections chunk-sized instead of 30 chars.
        if headings and headings[-1].title == title:
            continue
        headings.append(_Heading(level=1, title=title, page=page_index))
    return headings


def _paths(levels: Sequence[int], *, first_top: int) -> list[str]:
    """Dotted 1-based positions; level jumps (1 -> 3) are clamped to 1 -> 2."""
    counters: list[int] = [first_top - 1] if first_top > 1 else []
    out: list[str] = []
    for raw_level in levels:
        level = min(max(1, raw_level), len(counters) + 1)
        if level <= len(counters):
            del counters[level:]
            counters[-1] += 1
        else:
            counters.append(1)
        out.append(".".join(str(c) for c in counters))
    return out


def _anchors(
    headings: Sequence[_Heading], lines: Sequence[tuple[int, str]]
) -> list[tuple[int, int]]:
    """For each heading, (cut, body_start) indices into the line stream.

    `cut` ends the previous section's body (the heading's own title line is
    excluded from both sections); `body_start` opens this one. Both are
    monotonic, so sections never overlap even when the TOC pages are not.
    """
    pages = [page for page, _ in lines]
    result: list[tuple[int, int]] = []
    floor = 0
    for heading in headings:
        page_first = bisect_left(pages, heading.page)
        page_last = bisect_right(pages, heading.page + HEADING_SEARCH_PAGE_SPAN)
        start = max(page_first, floor)
        found = None
        norm_title = _norm(heading.title)
        for index in range(start, page_last):
            if _same_heading(_norm(lines[index][1]), norm_title):
                found = index
                break
        if found is None:
            cut = body_start = start
        else:
            cut, body_start = found, found + 1
        result.append((cut, body_start))
        floor = body_start
    return result


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _same_heading(line: str, title: str) -> bool:
    """A title may be split over physical lines, or carry extra numbering."""
    if not line or not title:
        return False
    if line == title:
        return True
    shorter, longer = sorted((line, title), key=len)
    return len(shorter) >= HEADING_MIN_CHARS and shorter in longer


# ------------------------------------------------------------------ assembly


def _build_sections(
    headings: Sequence[_Heading],
    lines: Sequence[tuple[int, str]],
    page_count: int,
    *,
    fallback_title: str,
) -> list[Section]:
    if not lines and not headings:
        return []

    anchors = _anchors(headings, lines)
    sections: list[Section] = []

    preamble_end = anchors[0][0] if anchors else len(lines)
    preamble = lines[:preamble_end]
    has_preamble = any(line for _, line in preamble)
    if has_preamble:
        sections.append(
            _section(
                path="1",
                title=PREAMBLE_TITLE if headings else fallback_title,
                span=preamble,
                heading_page=None,
                page_count=page_count,
            )
        )

    paths = _paths([h.level for h in headings], first_top=2 if has_preamble else 1)
    for index, heading in enumerate(headings):
        body_start = anchors[index][1]
        body_end = anchors[index + 1][0] if index + 1 < len(anchors) else len(lines)
        sections.append(
            _section(
                path=paths[index],
                title=heading.title,
                span=lines[body_start:max(body_start, body_end)],
                heading_page=heading.page,
                page_count=page_count,
            )
        )
    return sections


def _section(
    *,
    path: str,
    title: str,
    span: Sequence[tuple[int, str]],
    heading_page: int | None,
    page_count: int,
) -> Section:
    # Paragraph markers at the span's edges are layout residue, not content;
    # they must not stretch the page range or produce leading blank lines.
    start_idx, end_idx = 0, len(span)
    while start_idx < end_idx and not span[start_idx][1]:
        start_idx += 1
    while end_idx > start_idx and not span[end_idx - 1][1]:
        end_idx -= 1
    span = span[start_idx:end_idx]
    # The range covers the heading itself, so a heading at the foot of a page
    # is cited with that page and not only with the page its body spills onto.
    if span and heading_page is not None:
        page_start, page_end = min(heading_page, span[0][0]), span[-1][0]
    elif span:
        page_start, page_end = span[0][0], span[-1][0]
    else:
        page_start = page_end = min(max(1, heading_page or 1), max(1, page_count))
    # A line repeating the section's own title inside its body is a running
    # title (a slide header carried across the section's pages).
    norm_title = _norm(title)
    text = "\n".join(line for _, line in span if not line or _norm(line) != norm_title)
    # De-hyphenate words broken across lines, only when the continuation is
    # lowercase (conservative: legitimate mid-word hyphens at line end lose
    # the hyphen too, a rare and harmless cost for search quality).
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _BLANK_RUN_RE.sub("\n\n", text).strip()
    return Section(
        path=path,
        title=title,
        level=path.count(".") + 1,
        text=text,
        page_start=page_start,
        page_end=page_end,
    )


# ----------------------------------------------------------------- debug CLI


def _print_tree(doc: ExtractedDoc, source: Path) -> None:
    total = sum(len(s.text) for s in doc.sections)
    empty = sum(1 for s in doc.sections if not s.text)
    print(f"{source.name}")
    print(
        f"title: {doc.title or '-'} | pages: {doc.page_count} | "
        f"sections: {len(doc.sections)} | chars: {total} | empty sections: {empty}"
    )
    print("-" * 72)
    for section in doc.sections:
        indent = "  " * (section.level - 1)
        label = f"{section.path} {indent}{section.title}"
        pages = f"p{section.page_start}-{section.page_end}"
        print(f"{label[:52]:<54}{pages:>10}{len(section.text):>8} chars")


def _find_section(doc: ExtractedDoc, key: str) -> Section | None:
    for section in doc.sections:
        if section.path == key:
            return section
    if key.isdigit() and 1 <= int(key) <= len(doc.sections):
        return doc.sections[int(key) - 1]
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wikirag.extraction.pdf",
        description="Inspect what extraction makes of a PDF.",
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--sections",
        action="store_true",
        help="print the structure tree (default)",
    )
    parser.add_argument(
        "--text",
        metavar="N",
        help="print the full text of one section, by path ('3.2') or position",
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        print(f"no such file: {args.pdf}", file=sys.stderr)
        return 2

    doc = extract(args.pdf)
    if args.text and not args.sections:
        section = _find_section(doc, args.text)
        if section is None:
            print(f"no section {args.text!r}", file=sys.stderr)
            return 1
        print(
            f"[{section.path}] {section.title} "
            f"(p{section.page_start}-{section.page_end}, {len(section.text)} chars)"
        )
        print("-" * 72)
        print(section.text)
        return 0

    _print_tree(doc, args.pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
