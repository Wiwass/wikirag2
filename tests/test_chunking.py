"""Property-style tests for the deterministic chunker.

The extraction dataclasses are duplicated here on purpose: the chunker
duck-types its input, and the tests must not depend on
``wikirag.extraction.pdf`` being implemented.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import pytest

from wikirag.ingestion.chunking import (
    MIN_CHUNK_DIVISOR,
    chunk_document,
    estimate_tokens,
    section_rows,
)

CHUNK_ID = re.compile(r"^(?P<doc>[^#]+)#(?P<path>\d+(?:\.\d+)*)#(?P<n>\d+)$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"'’”)\]]*\s+")

WORDS = (
    "hermetic magus covenant vis aura ritual spell certamen tribunal grog "
    "laboratory enchantment familiar sigil arcane connection parma magica "
    "storyguide saga troupe virtue flaw longevity mystery cult faerie regio"
).split()


# ------------------------------------------------------------------ fixtures


@dataclass
class Section:
    path: str
    title: str
    level: int
    text: str
    page_start: int
    page_end: int


@dataclass
class ExtractedDoc:
    title: str | None
    page_count: int
    sections: list[Section] = field(default_factory=list)


def _sentence(rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(4, 22))]
    return " ".join(words).capitalize() + rng.choice([".", ".", "?", "!"])


def _paragraph(rng: random.Random, sentences: int) -> str:
    return " ".join(_sentence(rng) for _ in range(sentences))


def _section_text(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.10:
        return rng.choice(["", "   ", "\n\n\t\n"])
    if kind < 0.25:  # single short paragraph -> whole section under min
        return _paragraph(rng, 1)
    if kind < 0.45:  # one paragraph far above any test max_tokens
        return _paragraph(rng, rng.randint(60, 140))
    paragraphs = [
        _paragraph(rng, rng.randint(1, 30)) for _ in range(rng.randint(1, 12))
    ]
    # PDF-style hard wrapping inside paragraphs.
    paragraphs = [p.replace(" ", "\n", 1) if rng.random() < 0.3 else p for p in paragraphs]
    return "\n\n".join(paragraphs)


def make_doc(seed: int, n_sections: int = 8) -> ExtractedDoc:
    """Deterministic pseudo-document with a real (dotted) section tree."""
    rng = random.Random(seed)
    sections: list[Section] = []
    top = 0
    while len(sections) < n_sections:
        top += 1
        path = str(top)
        sections.append(
            Section(path, f"Chapter {top}", 1, _section_text(rng), top, top + 1)
        )
        for sub in range(1, rng.randint(1, 4)):
            if len(sections) >= n_sections:
                break
            sub_path = f"{top}.{sub}"
            sections.append(
                Section(sub_path, f"Section {sub_path}", 2, _section_text(rng), top, top + 2)
            )
    return ExtractedDoc(title=f"Doc {seed}", page_count=top + 2, sections=sections)


SEEDS = [1, 2, 3, 7, 11, 42, 99, 1234]
MAXES = [64, 200, 600]


def norm(text: str) -> str:
    return " ".join(text.split())


def strip_overlap(previous: str, current: str) -> str:
    """Remove the repeated leading sentence the chunker may have added."""
    sentences = SENTENCE_SPLIT.split(current)
    if sentences and previous.rstrip().endswith(sentences[0].strip()):
        return current[len(sentences[0]) :].lstrip()
    return current


# ---------------------------------------------------------------- estimator


def test_estimate_tokens_basics() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") >= 1
    prose = " ".join(WORDS * 10)
    assert estimate_tokens(prose) == estimate_tokens(prose)
    # Conservative: never below either component estimate.
    assert estimate_tokens(prose) >= len(prose.split()) * 4 // 3
    assert estimate_tokens(prose) >= len(prose) // 4
    assert estimate_tokens(prose + " more") > estimate_tokens(prose)


# ----------------------------------------------------------------- properties


@pytest.mark.parametrize("seed", SEEDS)
def test_determinism(seed: int) -> None:
    doc = make_doc(seed)
    first = chunk_document("doc1", doc, max_tokens=200)
    second = chunk_document("doc1", doc, max_tokens=200)
    # Also from an independently rebuilt but identical document.
    third = chunk_document("doc1", make_doc(seed), max_tokens=200)
    assert first == second == third
    assert section_rows("doc1", doc) == section_rows("doc1", make_doc(seed))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_tokens", MAXES)
def test_hard_token_bound(seed: int, max_tokens: int) -> None:
    for row in chunk_document("d", make_doc(seed), max_tokens=max_tokens):
        assert row["token_count"] == estimate_tokens(row["text"])
        assert row["token_count"] <= max_tokens


@pytest.mark.parametrize("seed", SEEDS)
def test_no_empty_chunks(seed: int) -> None:
    rows = chunk_document("d", make_doc(seed), max_tokens=200)
    assert rows
    for row in rows:
        assert row["text"].strip()
        assert row["token_count"] > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_ids_unique_and_well_formed(seed: int) -> None:
    doc = make_doc(seed)
    rows = chunk_document("kb-doc.1", doc, max_tokens=200)
    ids = [row["id"] for row in rows]
    assert len(set(ids)) == len(ids)
    per_section: dict[str, list[int]] = {}
    for row in rows:
        match = CHUNK_ID.match(row["id"])
        assert match, row["id"]
        assert match.group("doc") == "kb-doc.1"
        assert row["section_id"] == f"kb-doc.1#{match.group('path')}"
        assert row["doc_id"] == "kb-doc.1"
        per_section.setdefault(row["section_id"], []).append(int(match.group("n")))
    for ordinals in per_section.values():
        assert ordinals == list(range(len(ordinals)))  # 0-based, contiguous


@pytest.mark.parametrize("seed", SEEDS)
def test_global_ordinals_follow_document_order(seed: int) -> None:
    doc = make_doc(seed)
    rows = chunk_document("d", doc, max_tokens=200)
    assert [row["ordinal"] for row in rows] == list(range(len(rows)))
    order = {section.path: i for i, section in enumerate(doc.sections)}
    section_positions = [order[row["section_id"].split("#", 1)[1]] for row in rows]
    assert section_positions == sorted(section_positions)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_tokens", MAXES)
def test_tail_merge_leaves_no_fragments(seed: int, max_tokens: int) -> None:
    """A sub-threshold chunk is only allowed when no merge fits the bound."""
    min_tokens = max(1, max_tokens // MIN_CHUNK_DIVISOR)
    rows = chunk_document("d", make_doc(seed), max_tokens=max_tokens)
    by_section: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_section.setdefault(str(row["section_id"]), []).append(row)
    for section_chunks in by_section.values():
        if len(section_chunks) == 1:
            continue  # whole section is smaller than the threshold: allowed
        for i, row in enumerate(section_chunks):
            if int(row["token_count"]) >= min_tokens:
                continue
            neighbours = [
                section_chunks[j]
                for j in (i - 1, i + 1)
                if 0 <= j < len(section_chunks)
            ]
            assert neighbours
            for other in neighbours:
                merged = estimate_tokens(f"{other['text']} {row['text']}")
                assert merged > max_tokens, (
                    f"chunk {row['id']} ({row['token_count']} tokens) should have "
                    f"been merged with {other['id']}"
                )


def test_long_paragraph_is_sentence_split_with_overlap() -> None:
    rng = random.Random(5)
    paragraph = _paragraph(rng, 200)
    doc = ExtractedDoc("t", 1, [Section("1", "One", 1, paragraph, 1, 1)])
    rows = chunk_document("d", doc, max_tokens=600)
    assert len(rows) > 1
    assert all(row["token_count"] <= 600 for row in rows)
    # Every continuation chunk repeats the previous chunk's last sentence.
    overlaps = 0
    for previous, current in zip(rows, rows[1:]):
        first_sentence = SENTENCE_SPLIT.split(current["text"])[0].strip()
        if previous["text"].rstrip().endswith(first_sentence):
            overlaps += 1
    assert overlaps == len(rows) - 1
    # Canonical content survives the overlap.
    rebuilt = [rows[0]["text"]]
    for previous, current in zip(rows, rows[1:]):
        rebuilt.append(strip_overlap(previous["text"], current["text"]))
    assert norm(" ".join(rebuilt)) == norm(paragraph)


def test_short_paragraphs_do_not_get_overlap() -> None:
    rng = random.Random(6)
    paragraphs = [_paragraph(rng, 2) for _ in range(12)]
    doc = ExtractedDoc("t", 1, [Section("1", "One", 1, "\n\n".join(paragraphs), 1, 1)])
    assert all(estimate_tokens(p) < 150 for p in paragraphs)
    rows = chunk_document("d", doc, max_tokens=150)
    assert len(rows) > 1
    joined = "\n\n".join(str(row["text"]) for row in rows)
    assert norm(joined) == norm("\n\n".join(paragraphs))  # nothing repeated


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_tokens", MAXES)
def test_canonical_text_integrity(seed: int, max_tokens: int) -> None:
    doc = make_doc(seed)
    rows = chunk_document("d", doc, max_tokens=max_tokens)
    by_section: dict[str, list[str]] = {}
    for row in rows:
        by_section.setdefault(str(row["section_id"]), []).append(str(row["text"]))
    for section in doc.sections:
        texts = by_section.get(f"d#{section.path}", [])
        if not section.text.strip():
            assert not texts
            continue
        rebuilt = [texts[0]]
        for previous, current in zip(texts, texts[1:]):
            rebuilt.append(strip_overlap(previous, current))
        assert norm(" ".join(rebuilt)) == norm(section.text)


def test_whitespace_sections_yield_no_chunks() -> None:
    doc = ExtractedDoc(
        "t",
        2,
        [
            Section("1", "Empty", 1, "", 1, 1),
            Section("2", "Blank", 1, "   \n\n \t ", 2, 2),
        ],
    )
    assert chunk_document("d", doc, max_tokens=200) == []
    assert len(section_rows("d", doc)) == 2  # structure survives empty text


def test_single_token_monster_word_still_respects_bound() -> None:
    doc = ExtractedDoc("t", 1, [Section("1", "One", 1, "x" * 5000, 1, 1)])
    rows = chunk_document("d", doc, max_tokens=64)
    assert len(rows) > 1
    assert all(row["token_count"] <= 64 for row in rows)
    assert "".join(str(row["text"]) for row in rows).replace(" ", "") == "x" * 5000


# --------------------------------------------------------------- section rows


def test_section_rows_parenting() -> None:
    doc = ExtractedDoc(
        "t",
        3,
        [
            Section("1", "One", 1, "a", 1, 1),
            Section("1.1", "One One", 2, "b", 1, 1),
            Section("1.1.2", "Deep", 3, "c", 1, 1),
            Section("2.3", "Orphan", 2, "d", 2, 2),
        ],
    )
    rows = section_rows("doc", doc)
    assert [row["id"] for row in rows] == [
        "doc#1",
        "doc#1.1",
        "doc#1.1.2",
        "doc#2.3",
    ]
    assert [row["parent_id"] for row in rows] == [None, "doc#1", "doc#1.1", None]
    assert [row["ordinal"] for row in rows] == [0, 1, 2, 3]
    assert [row["level"] for row in rows] == [1, 2, 3, 2]
    assert rows[0]["title"] == "One"
    # Parents precede children, as Store.add_sections requires.
    seen: set[str] = set()
    for row in rows:
        assert row["parent_id"] is None or row["parent_id"] in seen
        seen.add(str(row["id"]))


def test_duplicate_section_paths_are_rejected() -> None:
    doc = ExtractedDoc(
        "t",
        1,
        [Section("1", "a", 1, "x", 1, 1), Section("1", "b", 1, "y", 1, 1)],
    )
    with pytest.raises(ValueError):
        section_rows("d", doc)
    with pytest.raises(ValueError):
        chunk_document("d", doc, max_tokens=100)


def test_custom_token_counter_is_honoured() -> None:
    rng = random.Random(9)
    doc = ExtractedDoc("t", 1, [Section("1", "One", 1, _paragraph(rng, 40), 1, 1)])
    words = lambda text: len(text.split())  # noqa: E731 - test-local counter
    rows = chunk_document("d", doc, max_tokens=50, token_counter=words)
    assert len(rows) > 1
    assert all(words(str(row["text"])) <= 50 for row in rows)
    assert all(row["token_count"] == words(str(row["text"])) for row in rows)
