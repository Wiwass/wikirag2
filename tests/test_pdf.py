"""Extraction tests. Fixture PDFs are built programmatically with PyMuPDF —
no real files, no network."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pymupdf
import pytest

from wikirag.extraction.pdf import ExtractedDoc, Section, extract, main

BODY_SIZE = 11.0
TITLE_SIZE = 24.0


def _write_pdf(
    path: Path,
    pages: Sequence[Sequence[tuple[float, str]]],
    *,
    toc: Sequence[list] | None = None,
    title: str | None = None,
) -> Path:
    """Each page is a sequence of (font size, line) drawn top to bottom."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 90.0
        for size, text in lines:
            page.insert_text((72, y), text, fontsize=size)
            y += size * 1.8
    if toc:
        doc.set_toc([list(entry) for entry in toc])
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def toc_pdf(tmp_path: Path) -> Path:
    pages = [
        [(BODY_SIZE, "Preface line one"), (BODY_SIZE, "Preface line two")],
        [(BODY_SIZE, "Chapter One"), (BODY_SIZE, "Alpha body text on page two.")],
        [(BODY_SIZE, "Section One A"), (BODY_SIZE, "Beta body text on page three.")],
        [(BODY_SIZE, "Chapter Two"), (BODY_SIZE, "Gamma body text on page four.")],
    ]
    toc = [
        [1, "Chapter One", 2],
        [2, "Section One A", 3],
        [1, "Chapter Two", 4],
    ]
    return _write_pdf(tmp_path / "toc.pdf", pages, toc=toc, title="A Structured Book")


@pytest.fixture
def font_pdf(tmp_path: Path) -> Path:
    pages = [
        [(TITLE_SIZE, "Introduction"), (BODY_SIZE, "First body paragraph.")],
        [(TITLE_SIZE, "Methods"), (BODY_SIZE, "Second body paragraph.")],
        [(TITLE_SIZE, "Results"), (BODY_SIZE, "Third body paragraph.")],
    ]
    return _write_pdf(tmp_path / "font.pdf", pages)


@pytest.fixture
def flat_pdf(tmp_path: Path) -> Path:
    pages = [
        [(BODY_SIZE, "Uniform line one."), (BODY_SIZE, "Uniform line two.")],
        [(BODY_SIZE, "Uniform line three.")],
    ]
    return _write_pdf(tmp_path / "flat.pdf", pages)


@pytest.fixture
def boilerplate_pdf(tmp_path: Path) -> Path:
    header = "CONFIDENTIAL DRAFT - DO NOT CIRCULATE"
    pages = [
        [
            (BODY_SIZE, header),
            (BODY_SIZE, f"Unique body sentence number {n}."),
            (BODY_SIZE, str(n)),
        ]
        for n in range(1, 5)
    ]
    return _write_pdf(tmp_path / "boiler.pdf", pages)


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    return _write_pdf(tmp_path / "empty.pdf", [[], []])


def _paths(doc: ExtractedDoc) -> list[str]:
    return [s.path for s in doc.sections]


def _by_path(doc: ExtractedDoc, path: str) -> Section:
    return next(s for s in doc.sections if s.path == path)


def test_toc_structure_paths_levels_and_order(toc_pdf: Path) -> None:
    doc = extract(toc_pdf)

    assert doc.title == "A Structured Book"
    assert doc.page_count == 4
    # "1" is the synthetic preamble section; TOC top levels continue at 2.
    assert _paths(doc) == ["1", "2", "2.1", "3"]
    assert [s.level for s in doc.sections] == [1, 1, 2, 1]
    assert [s.title for s in doc.sections][1:] == [
        "Chapter One",
        "Section One A",
        "Chapter Two",
    ]


def test_toc_section_text_is_own_body_only(toc_pdf: Path) -> None:
    doc = extract(toc_pdf)

    assert "Preface line one" in _by_path(doc, "1").text
    chapter_one = _by_path(doc, "2")
    assert "Alpha body text on page two." in chapter_one.text
    # The child's body and both heading lines stay out of the parent's text.
    assert "Beta" not in chapter_one.text
    assert "Chapter One" not in chapter_one.text
    assert "Beta body text on page three." in _by_path(doc, "2.1").text
    assert "Gamma body text on page four." in _by_path(doc, "3").text


def test_page_ranges_are_sane(toc_pdf: Path) -> None:
    doc = extract(toc_pdf)

    for section in doc.sections:
        assert 1 <= section.page_start <= section.page_end <= doc.page_count
    assert (_by_path(doc, "1").page_start, _by_path(doc, "1").page_end) == (1, 1)
    assert (_by_path(doc, "2").page_start, _by_path(doc, "2").page_end) == (2, 2)
    assert (_by_path(doc, "3").page_start, _by_path(doc, "3").page_end) == (4, 4)


def test_page_range_covers_the_heading_page(tmp_path: Path) -> None:
    pages = [
        [(BODY_SIZE, "Preface.")],
        [(BODY_SIZE, "Chapter One")],  # heading alone at the end of its page
        [(BODY_SIZE, "Body that spilled over.")],
    ]
    pdf = _write_pdf(tmp_path / "spill.pdf", pages, toc=[[1, "Chapter One", 2]])

    section = _by_path(extract(pdf), "2")

    assert (section.page_start, section.page_end) == (2, 3)
    assert section.text == "Body that spilled over."


def test_repeated_slide_title_becomes_one_section(tmp_path: Path) -> None:
    # Two slides share a title (below the boilerplate threshold, so the title
    # itself survives), then the deck moves on.
    pages = [
        [(TITLE_SIZE, "Word alignment"), (BODY_SIZE, f"Slide body {n}.")]
        for n in range(1, 3)
    ] + [
        [(TITLE_SIZE, "Decoding"), (BODY_SIZE, "Third slide body.")],
        [(TITLE_SIZE, "Evaluation"), (BODY_SIZE, "Fourth slide body.")],
    ]
    pdf = _write_pdf(tmp_path / "slides.pdf", pages)

    doc = extract(pdf)

    assert _paths(doc) == ["1", "2", "3"]
    merged = doc.sections[0]
    assert merged.title == "Word alignment"
    assert (merged.page_start, merged.page_end) == (1, 2)
    # The running title must not survive inside the merged body.
    assert "Word alignment" not in merged.text
    # Separate slides stay separate paragraphs (blank-line boundary).
    assert merged.text.splitlines() == ["Slide body 1.", "", "Slide body 2."]


def test_font_size_headings_without_toc(font_pdf: Path) -> None:
    doc = extract(font_pdf)

    assert _paths(doc) == ["1", "2", "3"]
    assert [s.title for s in doc.sections] == ["Introduction", "Methods", "Results"]
    assert all(s.level == 1 for s in doc.sections)
    assert "First body paragraph." in doc.sections[0].text
    assert "Methods" not in doc.sections[1].text


def test_flat_document_degrades_to_one_section(flat_pdf: Path) -> None:
    doc = extract(flat_pdf)

    assert _paths(doc) == ["1"]
    section = doc.sections[0]
    assert section.level == 1
    assert "Uniform line one." in section.text
    assert "Uniform line three." in section.text
    assert (section.page_start, section.page_end) == (1, 2)


def test_repeated_header_and_page_numbers_are_removed(boilerplate_pdf: Path) -> None:
    doc = extract(boilerplate_pdf)

    assert doc.sections, "a document with text must never yield zero sections"
    all_text = "\n".join(s.text for s in doc.sections)
    assert "CONFIDENTIAL" not in all_text
    for n in range(1, 5):
        assert f"Unique body sentence number {n}." in all_text
    assert [line for line in all_text.splitlines() if line.strip().isdigit()] == []


def test_empty_document_yields_no_sections(empty_pdf: Path) -> None:
    doc = extract(empty_pdf)

    assert doc.page_count == 2
    assert doc.sections == []


def test_extraction_is_deterministic(toc_pdf: Path, font_pdf: Path) -> None:
    for path in (toc_pdf, font_pdf):
        first, second = extract(path), extract(path)
        assert first == second
        assert repr(first) == repr(second)


def test_accepts_str_path(flat_pdf: Path) -> None:
    assert extract(str(flat_pdf)) == extract(flat_pdf)


def test_cli_prints_tree(toc_pdf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(toc_pdf)]) == 0

    out = capsys.readouterr().out
    assert "A Structured Book" in out
    assert "Chapter Two" in out
    assert "p4-4" in out


def test_cli_prints_one_section_text(
    toc_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(toc_pdf), "--text", "2.1"]) == 0

    out = capsys.readouterr().out
    assert "Beta body text on page three." in out
    assert "Gamma" not in out


def test_cli_reports_missing_file_and_section(
    tmp_path: Path, toc_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "nope.pdf")]) == 2
    assert main([str(toc_pdf), "--text", "99.9"]) == 1
