"""Benchmark harness: run the S x R x budget grid, judge blind, report.

Ported/adapted from v1's benchmark/ package:
- gold datasets per KB (questions + gold answers + key points,
  simple/complex split) — the Ars Magica dataset carries over;
- blind judge (question, gold, key points, candidate; never system
  identity), 0-30 scale: correctness + completeness + groundedness;
- run artifacts kept (contexts, answers, judgments); baseline freezing;
  judge model pinned per dataset version;
- standing regression suite = one S row x all R x budgets; full grid on
  ingest-method changes;
- retrieval-only micro-benchmarks (recall@k on annotated chunks) for
  component choices (embedding model, reranker, vector backend) without
  paying judge costs.

The grid
--------
``run_grid`` runs one cell per (depth, budget) pair against the KB it is
handed. The S axis is the KB itself — one KB per synthesis level, recorded
in the manifest — so a full S x R x budget cube is one ``run_grid`` call
per S level; the runner never re-ingests anything. Each cell reuses
``answer_run`` + ``judge_answers`` (it does not re-implement either) and
writes its own artifact, ``answers_{depth}_{budget}.json``, named
deterministically from the cell so ``resume=True`` can skip cells that
already finished: a grid is expensive and must survive interruption, the
same discipline ingestion applies to its jobs. A cell is "complete" only
if it holds a result for every question, was not aborted, and — when a
judge client was given — carries a judgment on every non-errored result;
anything less is re-run. ``AllKeysExhausted`` (raised, or recorded in an
artifact's meta by ``answer_run``/``judge_answers``) stops the grid and
leaves the manifest saying which cells finished and which were never
attempted.

Reports
-------
``manifest.json`` is rewritten after every cell — kb, synthesis level,
corpus, dataset path and hash, git commit, the four models
(answer/judge/embedding/reranker), the axes, timestamps and per-cell
status — so an interrupted grid still describes itself. ``compare`` diffs
a run against a frozen baseline and calls the difference against
``NOISE_THRESHOLD``; ``report_markdown``/``write_report`` render
``summary.md`` in v1's shape: per-cell tables with SEM, the note on
significance, cost per cell from the ``llm_calls`` meter, and which of
docs/evaluation.md's standing questions the run's axes can answer.

Retrieval metrics are a proxy, and the names say so
---------------------------------------------------
``doc_hit@k`` and ``page_hit@k`` are not recall@k: the gold dataset
annotates *source documents* and *printed page citations*, not the chunks
that answer the question, so "the right page was retrieved" is the closest
observable. Two caveats that the numbers cannot show on their own:

- a citation names the book's **printed** page number, while a section's
  ``page_start``/``page_end`` are **physical PDF** pages. Front matter
  shifts one against the other, which is what ``page_tolerance`` absorbs —
  it is a fudge factor, not a relevance window, and a systematic offset
  larger than the tolerance shows up as a floor-zero page_hit rate;
- a citation lists pages for the question as a whole, not per source
  document, so a page hit only requires the candidate to sit in *some*
  gold source document at *some* cited page.

Questions whose citation cannot be parsed are scored doc-only (their
page_hit equals their doc_hit) and counted separately in the report, so a
page_hit rate is always read next to ``n_with_pages``.

Blind judge
-----------
Ported from v1 ``benchmark/judge.py``, prompt and rubric unchanged
(0-12 correctness, 0-12 completeness, 0-6 groundedness): the judge sees the
question, the gold answer, the key points and the candidate answer — never
the retrieved context, the depth, the budget or which system produced the
answer. Validation is a pure function with per-field clamping and a total
*recomputed* from the sub-scores, so a model that ignores the schema
degrades into a valid Judgment instead of a corrupt score. A judge call
that keeps failing degrades to ``fallback=True`` (score 0) rather than
killing a run; ``AllKeysExhausted`` is the exception that does stop the
batch — swallowing it would mark every remaining answer as a zero.

Artifacts
---------
``answer_run`` writes one JSON file::

    {"meta": {"kb", "depth", "budget_tokens", "model", "ts", "aborted"?},
     "results": [{"id", "type", "question", "answer", "no_answer",
                  "context_tokens", "usage": {...},
                  "provenance": [{"doc_id", "section_title", "ref"}],
                  "error"?, "judgment"?}]}

``judge_answers`` adds ``judgment`` to each result in place; ``summarize``
reads the file back. Results carry ``type`` so the simple/complex split
survives without the dataset.

Every LLM call goes through the client the caller hands in (the CLI wraps
it in MeteredClient); this module never constructs a provider.

Public API
----------
- ``class GoldQuestion``: id, type, question, gold_answer, key_points,
  source_docs, citation, cited_pages
- ``parse_cited_pages(citation) -> list[int]``
- ``load_dataset(path) -> list[GoldQuestion]``
- ``slugify(title) -> str``
- ``match_doc_ids(kb, titles) -> dict[str, str | None]``
- ``class QuestionRetrieval`` / ``class RetrievalReport`` (``.rows()``)
- ``retrieval_eval(kb, questions, ks=(5, 10, 20), *, embedder=None,
  reranker=None, page_tolerance=2) -> RetrievalReport``
- ``answer_run(kb, questions, *, client, depth=R1, budget_tokens=10_000,
  out_path) -> dict`` (the artifact, also written to ``out_path``)
- ``JUDGE_SYSTEM_PROMPT``, ``JUDGE_RESPONSE_SCHEMA``, ``MAX_JUDGE_ATTEMPTS``
- ``class Judgment``, ``judge_answer(...) -> Judgment``
- ``judge_answers(artifact, questions, *, judge_client, max_attempts=2)
  -> dict[str, Judgment]``
- ``summarize(artifact) -> list[dict]``
- ``DEFAULT_DEPTHS``, ``DEFAULT_BUDGETS``, ``NOISE_THRESHOLD``,
  ``MANIFEST_FILENAME``, ``SUMMARY_FILENAME``
- ``cell_filename(depth, budget) -> str``
- ``class CellResult`` / ``class GridReport`` (``.rows()``,
  ``.completed()``, ``.manifest()``, ``.manifest_path``)
- ``run_grid(kb, questions, *, client, depths=DEFAULT_DEPTHS,
  budgets=DEFAULT_BUDGETS, out_dir, judge_client=None, resume=True,
  dataset_path=None, corpus="") -> GridReport``
- ``compare(baseline_path, run_path) -> list[dict]``
- ``report_markdown(run_dir) -> str``
- ``write_report(run_dir) -> Path``
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wikirag.answer import answer
from wikirag.config import KnowledgeBase, RetrievalDepth
from wikirag.llm.client import AllKeysExhausted, LLMClient
from wikirag.retrieval.rerank import no_answer, rerank
from wikirag.retrieval.search import Candidate, hybrid_search

__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_DEPTHS",
    "DEFAULT_KS",
    "DEFAULT_PAGE_TOLERANCE",
    "JUDGE_RESPONSE_SCHEMA",
    "JUDGE_SYSTEM_PROMPT",
    "MANIFEST_FILENAME",
    "MAX_JUDGE_ATTEMPTS",
    "NOISE_THRESHOLD",
    "SUMMARY_FILENAME",
    "CellResult",
    "GoldQuestion",
    "GridReport",
    "Judgment",
    "QuestionRetrieval",
    "RetrievalReport",
    "answer_run",
    "cell_filename",
    "compare",
    "judge_answer",
    "judge_answers",
    "load_dataset",
    "match_doc_ids",
    "parse_cited_pages",
    "report_markdown",
    "retrieval_eval",
    "run_grid",
    "slugify",
    "summarize",
    "write_report",
]

DEFAULT_KS: tuple[int, ...] = (5, 10, 20)

#: Printed page numbers vs physical PDF pages: front matter shifts them by a
#: few pages in either direction. See "Retrieval metrics are a proxy".
DEFAULT_PAGE_TOLERANCE = 2

DEFAULT_BUDGET_TOKENS = 10_000

VALID_TYPES = ("simple", "complex")


# --------------------------------------------------------------- gold dataset


@dataclass
class GoldQuestion:
    """One authored benchmark question (v1 dataset schema + parsed pages)."""

    id: str
    type: str
    question: str
    gold_answer: str
    key_points: list[str] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)
    citation: str = ""
    cited_pages: list[int] = field(default_factory=list)


# A citation must actually say "page"/"pages"; a bare number elsewhere in the
# string ("chapter 3") is not a page reference and must not become one.
_PAGE_KEYWORD = re.compile(r"\bpages?\b", re.IGNORECASE)
# Ranges may be written with an ASCII hyphen or any of the Unicode dashes a
# word processor substitutes for one (U+2010..U+2015).
_PAGE_NUMBERS = re.compile(r"(\d+)\s*(?:[-‐-―]\s*(\d+))?")

#: Longest page range expanded literally. A wider span is far more likely a
#: typo or an OCR artefact than an annotation of 200 relevant pages, so only
#: its endpoints are kept.
_MAX_RANGE_SPAN = 50


def parse_cited_pages(citation: str | None) -> list[int]:
    """Pages named by a citation string, ranges expanded, sorted and unique.

    Handles the three shapes the gold dataset uses — ``"page 95"``,
    ``"pages 64, 110, 111"``, ``"pages 10-11, 25"`` — and returns ``[]`` for
    anything else (missing, empty, or prose without a page keyword), which
    the retrieval report treats as "score this question doc-only".
    """
    if not citation or not _PAGE_KEYWORD.search(citation):
        return []
    pages: set[int] = set()
    for start_text, end_text in _PAGE_NUMBERS.findall(citation):
        start = int(start_text)
        pages.add(start)
        if not end_text:
            continue
        end = int(end_text)
        if start <= end <= start + _MAX_RANGE_SPAN:
            pages.update(range(start, end + 1))
        else:
            pages.add(end)
    return sorted(pages)


def load_dataset(path: str | Path) -> list[GoldQuestion]:
    """Load ``benchmarks/<kb>/dataset.json`` into GoldQuestions.

    Structurally invalid questions (missing id/question/gold answer, unknown
    type) raise: a silently dropped question would quietly change every
    aggregate that follows.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(f"{path}: no 'questions' list in dataset")

    questions: list[GoldQuestion] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_questions):
        label = f"{path}: questions[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: not an object")
        qid = raw.get("id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{label}: 'id' must be a non-empty string")
        if qid in seen:
            raise ValueError(f"{label}: duplicate id {qid!r}")
        seen.add(qid)
        qtype = raw.get("type")
        if qtype not in VALID_TYPES:
            raise ValueError(f"{label} (id={qid}): 'type' must be one of {VALID_TYPES}")
        for required in ("question", "gold_answer"):
            if not isinstance(raw.get(required), str) or not raw[required].strip():
                raise ValueError(f"{label} (id={qid}): {required!r} must be a non-empty string")
        citation = raw.get("citation") or ""
        questions.append(
            GoldQuestion(
                id=qid,
                type=qtype,
                question=raw["question"],
                gold_answer=raw["gold_answer"],
                key_points=[str(kp) for kp in raw.get("key_points") or []],
                source_docs=[str(doc) for doc in raw.get("source_docs") or []],
                citation=str(citation),
                cited_pages=parse_cited_pages(str(citation)),
            )
        )
    return questions


_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """The doc_id shape ingestion derives from a filename: ``ag0205-ars-magica``."""
    return _NON_SLUG.sub("-", title.lower()).strip("-")


def match_doc_ids(kb: KnowledgeBase, titles: Iterable[str]) -> dict[str, str | None]:
    """Map gold ``source_docs`` titles onto the KB's doc_ids.

    The dataset names documents the way v1 did — by PDF stem ("AG0205 - Ars
    Magica") — while v2 stores slugs. Matching is by slug: exact, then
    prefix, then containment either way (a KB whose filename carries an
    extra suffix, or a dataset title that abbreviates one). Unmatched titles
    map to ``None`` so the caller can report KB coverage instead of reading
    a missing book as a retrieval failure.
    """
    documents = kb.store.list_documents()
    candidates: list[tuple[str, str]] = []
    for row in documents:
        doc_id = str(row["id"])
        candidates.append((doc_id, doc_id))
        filename = row.get("filename")
        if filename:
            candidates.append((slugify(Path(str(filename)).stem), doc_id))

    matched: dict[str, str | None] = {}
    for title in titles:
        wanted = slugify(str(title))
        matched[title] = _best_doc_match(wanted, candidates) if wanted else None
    return matched


def _best_doc_match(wanted: str, candidates: Sequence[tuple[str, str]]) -> str | None:
    for slug, doc_id in candidates:
        if slug == wanted:
            return doc_id
    for slug, doc_id in candidates:
        if slug.startswith(wanted) or wanted.startswith(slug):
            return doc_id
    for slug, doc_id in candidates:
        if wanted in slug or slug in wanted:
            return doc_id
    return None


# ----------------------------------------------------------- retrieval report


@dataclass
class QuestionRetrieval:
    """Per-question retrieval outcome; ``doc_hit``/``page_hit`` are keyed by k."""

    id: str
    type: str
    gold_doc_ids: list[str]
    unmatched_source_docs: list[str]
    cited_pages: list[int]
    doc_hit: dict[int, bool]
    page_hit: dict[int, bool]
    top_score: float | None
    would_no_answer: bool
    n_candidates: int


@dataclass
class RetrievalReport:
    """Retrieval-only micro-benchmark result (docs/evaluation.md, Practices)."""

    ks: tuple[int, ...]
    page_tolerance: int
    questions: list[QuestionRetrieval] = field(default_factory=list)

    def rows(self) -> list[dict[str, Any]]:
        """One printable row per (subset, k): simple, complex, then overall."""
        rows: list[dict[str, Any]] = []
        for subset in ("simple", "complex", "overall"):
            selected = [
                q for q in self.questions if subset == "overall" or q.type == subset
            ]
            if not selected:
                continue
            with_pages = [q for q in selected if q.cited_pages]
            for k in self.ks:
                rows.append(
                    {
                        "subset": subset,
                        "k": k,
                        "n": len(selected),
                        "doc_hit": _mean([q.doc_hit[k] for q in selected]),
                        "page_hit": _mean([q.page_hit[k] for q in selected]),
                        "n_with_pages": len(with_pages),
                        "no_answer_rate": _mean([q.would_no_answer for q in selected]),
                        "mean_top_score": _mean(
                            [q.top_score for q in selected if q.top_score is not None]
                        ),
                        "n_without_gold_docs": sum(
                            1 for q in selected if not q.gold_doc_ids
                        ),
                    }
                )
        return rows


def retrieval_eval(
    kb: KnowledgeBase,
    questions: Sequence[GoldQuestion],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    embedder: Any = None,
    reranker: Any = None,
    page_tolerance: int = DEFAULT_PAGE_TOLERANCE,
) -> RetrievalReport:
    """Score retrieval alone: no generation, no judge, no API cost.

    One ``hybrid_search`` at ``max(ks)`` plus one ``rerank`` per question;
    every k is a prefix of that single reranked pool, so the ks share the
    work and stay mutually consistent.

    A question hits at k when a top-k candidate belongs to one of its gold
    source documents (``doc_hit``), and — for questions with parsable page
    citations — when such a candidate's section page range overlaps a cited
    page within ``page_tolerance`` (``page_hit``). Questions without parsable
    pages take their doc_hit as their page_hit, and are counted in the
    report's ``n_with_pages`` so the rate stays readable.
    """
    ordered_ks = tuple(sorted({int(k) for k in ks}))
    if not ordered_ks or ordered_ks[0] < 1:
        raise ValueError(f"ks must be positive integers, got {list(ks)}")
    if page_tolerance < 0:
        raise ValueError(f"page_tolerance must be >= 0, got {page_tolerance}")

    pool_size = ordered_ks[-1]
    section_pages: dict[str, dict[str, tuple[int, int]]] = {}
    report = RetrievalReport(ks=ordered_ks, page_tolerance=page_tolerance)

    for question in questions:
        matches = match_doc_ids(kb, question.source_docs)
        gold_doc_ids = [doc_id for doc_id in matches.values() if doc_id]
        unmatched = [title for title, doc_id in matches.items() if doc_id is None]

        candidates = hybrid_search(kb, [question.question], k=pool_size, embedder=embedder)
        scored = rerank(question.question, candidates, reranker=reranker)

        doc_hit: dict[int, bool] = {}
        page_hit: dict[int, bool] = {}
        for k in ordered_ks:
            top = scored[:k]
            in_gold = [c for c in top if c.doc_id in gold_doc_ids]
            doc_hit[k] = bool(in_gold)
            if question.cited_pages:
                page_hit[k] = any(
                    _overlaps_cited_page(
                        _page_range(kb, candidate, section_pages),
                        question.cited_pages,
                        page_tolerance,
                    )
                    for candidate in in_gold
                )
            else:
                page_hit[k] = doc_hit[k]

        report.questions.append(
            QuestionRetrieval(
                id=question.id,
                type=question.type,
                gold_doc_ids=gold_doc_ids,
                unmatched_source_docs=unmatched,
                cited_pages=list(question.cited_pages),
                doc_hit=doc_hit,
                page_hit=page_hit,
                top_score=max((float(c.score) for c in scored if c.score is not None), default=None),
                would_no_answer=no_answer(scored),
                n_candidates=len(scored),
            )
        )
    return report


def _page_range(
    kb: KnowledgeBase,
    candidate: Candidate,
    cache: dict[str, dict[str, tuple[int, int]]],
) -> tuple[int, int] | None:
    """The physical page span of a candidate's section, if the store knows it."""
    if candidate.doc_id is None or candidate.section_id is None:
        return None
    pages = cache.get(candidate.doc_id)
    if pages is None:
        pages = {}
        for section in kb.store.get_sections(candidate.doc_id):
            start, end = section.get("page_start"), section.get("page_end")
            if start is None and end is None:
                continue
            start = int(start if start is not None else end)
            end = int(end if end is not None else start)
            pages[str(section["id"])] = (min(start, end), max(start, end))
        cache[candidate.doc_id] = pages
    return pages.get(candidate.section_id)


def _overlaps_cited_page(
    page_range: tuple[int, int] | None, cited_pages: Sequence[int], tolerance: int
) -> bool:
    if page_range is None:
        return False
    start, end = page_range
    return any(start - tolerance <= page <= end + tolerance for page in cited_pages)


def _mean(values: Sequence[Any]) -> float | None:
    numeric = [float(v) for v in values]
    return sum(numeric) / len(numeric) if numeric else None


# ------------------------------------------------------------- answer artifact


def answer_run(
    kb: KnowledgeBase,
    questions: Sequence[GoldQuestion],
    *,
    client: LLMClient,
    depth: RetrievalDepth | str = RetrievalDepth.R1,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    out_path: str | Path,
) -> dict[str, Any]:
    """Answer every question once and write the run artifact to ``out_path``.

    ``client`` is used as given (the caller meters it). One question raising
    records ``error`` on its result and the run continues — a single bad
    question must not cost the whole run — while ``AllKeysExhausted`` stops
    the run, marks ``meta['aborted']`` and still writes what was answered:
    resuming later is cheaper than re-answering, and every remaining
    question would only re-fail.
    """
    level = RetrievalDepth.parse(depth)
    artifact: dict[str, Any] = {
        "meta": {
            "kb": kb.name,
            "depth": level.value,
            "budget_tokens": int(budget_tokens),
            "model": str(getattr(client, "model", "unknown")),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "results": [],
    }

    for question in questions:
        record: dict[str, Any] = {"id": question.id, "type": question.type, "question": question.question}
        try:
            result = answer(
                kb,
                question.question,
                depth=level,
                client=client,
                budget_tokens=budget_tokens,
            )
        except AllKeysExhausted as exc:
            artifact["meta"]["aborted"] = f"AllKeysExhausted: {exc}"
            break
        except Exception as exc:  # one question's failure, not the run's
            record["error"] = f"{type(exc).__name__}: {exc}"
            artifact["results"].append(record)
            continue
        record.update(
            answer=result.text,
            no_answer=result.no_answer,
            context_tokens=result.context_tokens,
            usage=asdict(result.usage),
            provenance=[
                {
                    "doc_id": entry.doc_id,
                    "section_title": entry.section_title,
                    "ref": entry.ref,
                }
                for entry in result.provenance
            ],
        )
        artifact["results"].append(record)

    _write_artifact(out_path, artifact)
    return artifact


def _write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------- blind judge

MAX_JUDGE_ATTEMPTS = 2

#: Plain-dict JSON schema (the LLMClient contract); providers translate it.
JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer"},
        "completeness": {"type": "integer"},
        "groundedness": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["correctness", "completeness", "groundedness", "rationale"],
}

JUDGE_SYSTEM_PROMPT = """You are grading an AI assistant's answer against a reference answer, for a personal knowledge-base RAG system about the Ars Magica tabletop RPG.

You will receive: the question, a reference (gold) answer, a list of key points the answer should cover, and the candidate answer to grade. You do NOT see how the candidate retrieved its information, and you do NOT know which system produced it — grade only the final answer text, on its own merits.

Score three sub-scores as integers, then a short rationale:

- "correctness" (0-12): factual accuracy of claims made, judged against the reference answer. 12 = fully accurate, 0 = factually wrong or contradicts the reference.
- "completeness" (0-12): how many of the listed key points are actually covered by the candidate answer. 12 = all key points present, 0 = none present.
- "groundedness" (0-6): penalize invented mechanics, rules, or details not supported by the reference answer or key points, and penalize the candidate inventing facts instead of admitting missing context. 6 = no fabrication, 0 = significant fabrication.

An answer that correctly says the context does not contain enough information, instead of guessing, should NOT be penalized on groundedness for that honesty — but should still score low on completeness/correctness if the information was actually needed and available in a real knowledge base.

Output the JSON object only, no markdown fences, no commentary."""


@dataclass
class Judgment:
    """One blind judgment; ``total`` is always the sum of the sub-scores."""

    correctness: int = 0
    completeness: int = 0
    groundedness: int = 0
    total: int = 0
    rationale: str = ""
    fallback: bool = False  # the judge failed and this is the zero default


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, number))


def _validate_judgment(data: Mapping[str, Any]) -> Judgment:
    """Normalize the judge's raw JSON into a coherent Judgment. Pure.

    Each sub-score is clamped to its own range (a model that ignores the
    schema or extrapolates cannot inflate a run), and the total is ALWAYS
    recomputed from the sub-scores instead of trusting a ``total`` field.
    """
    correctness = _clamp_int(data.get("correctness", 0), 0, 12)
    completeness = _clamp_int(data.get("completeness", 0), 0, 12)
    groundedness = _clamp_int(data.get("groundedness", 0), 0, 6)

    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""

    return Judgment(
        correctness=correctness,
        completeness=completeness,
        groundedness=groundedness,
        total=correctness + completeness + groundedness,
        rationale=rationale.strip(),
    )


def _build_judge_message(
    question: str, gold_answer: str, key_points: Sequence[str], candidate_answer: str
) -> str:
    key_points_block = "\n".join(f"- {kp}" for kp in key_points) if key_points else "(none listed)"
    return (
        f"Question:\n{question}\n\n"
        f"Reference (gold) answer:\n{gold_answer}\n\n"
        f"Key points expected:\n{key_points_block}\n\n"
        f"Candidate answer to grade:\n{candidate_answer}"
    )


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse_json_object(text: str) -> Any:
    """Parse the judge's output, tolerating markdown fences a model adds anyway."""
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def judge_answer(
    judge_client: LLMClient,
    question: str,
    gold_answer: str,
    key_points: Sequence[str],
    candidate_answer: str,
    *,
    max_attempts: int = MAX_JUDGE_ATTEMPTS,
) -> Judgment:
    """Grade one candidate answer blind.

    Retries malformed JSON up to ``max_attempts``; any other error, or
    exhausted attempts, degrades to ``Judgment(fallback=True)`` (score 0)
    rather than failing the run. ``AllKeysExhausted`` propagates: inside a
    batch it means every remaining question would fall back too.
    """
    user_message = _build_judge_message(question, gold_answer, key_points, candidate_answer)

    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            result = judge_client.generate(
                system=JUDGE_SYSTEM_PROMPT,
                user=user_message,
                json_schema=JUDGE_RESPONSE_SCHEMA,
            )
            data = _parse_json_object(result.text)
        except json.JSONDecodeError:
            if attempt < max_attempts:
                continue
            return Judgment(fallback=True)
        except AllKeysExhausted:
            raise
        except Exception:
            return Judgment(fallback=True)

        if not isinstance(data, dict):
            if attempt < max_attempts:
                continue
            return Judgment(fallback=True)
        return _validate_judgment(data)

    return Judgment(fallback=True)


def judge_answers(
    artifact: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    questions: Sequence[GoldQuestion],
    *,
    judge_client: LLMClient,
    max_attempts: int = MAX_JUDGE_ATTEMPTS,
) -> dict[str, Judgment]:
    """Judge a run's answers blind and write the judgments back into it.

    ``artifact`` is a path to a run artifact, an already-loaded artifact
    dict, or its bare ``results`` list. Each judged result gains a
    ``judgment`` field; when a path was given the file is rewritten.
    Results carrying an ``error``, and results whose id is not in
    ``questions``, are skipped (there is nothing blind to grade).

    Returns the judgments by question id. ``AllKeysExhausted`` stops the
    pass, keeps what was judged, and marks ``meta['judge_aborted']``.
    """
    path = artifact if isinstance(artifact, (str, Path)) else None
    if path is not None:
        loaded: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        loaded = artifact
    results = loaded["results"] if isinstance(loaded, Mapping) else list(loaded)
    meta = loaded.get("meta") if isinstance(loaded, Mapping) else None

    by_id = {question.id: question for question in questions}
    judgments: dict[str, Judgment] = {}
    for record in results:
        question = by_id.get(record.get("id"))
        if question is None or record.get("error"):
            continue
        try:
            judgment = judge_answer(
                judge_client,
                question.question,
                question.gold_answer,
                question.key_points,
                record.get("answer", ""),
                max_attempts=max_attempts,
            )
        except AllKeysExhausted as exc:
            if isinstance(meta, dict):
                meta["judge_aborted"] = f"AllKeysExhausted: {exc}"
            break
        judgments[question.id] = judgment
        record["judgment"] = asdict(judgment)

    if path is not None:
        _write_artifact(path, loaded)
    return judgments


# --------------------------------------------------------------------- report


def summarize(artifact: str | Path | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Aggregate a judged run into printable rows: simple, complex, overall.

    Each row carries n, judged, mean total and its SEM (stdev / sqrt(n),
    ``0.0`` below two judged answers — a single sample has no spread to
    report), the mean sub-scores, and the counts that qualify them:
    no-answer replies, judge fallbacks and errored questions. The SEM is the
    noise band docs/evaluation.md compares cells against.
    """
    if isinstance(artifact, (str, Path)):
        loaded: Any = json.loads(Path(artifact).read_text(encoding="utf-8"))
    else:
        loaded = artifact
    results = loaded["results"] if isinstance(loaded, Mapping) else list(loaded)

    rows: list[dict[str, Any]] = []
    for subset in ("simple", "complex", "overall"):
        selected = [
            r for r in results if subset == "overall" or r.get("type") == subset
        ]
        if not selected:
            continue
        judged = [r["judgment"] for r in selected if isinstance(r.get("judgment"), Mapping)]
        totals = [float(j["total"]) for j in judged]
        rows.append(
            {
                "subset": subset,
                "n": len(selected),
                "judged": len(judged),
                "mean_total": _mean(totals),
                "sem_total": _sem(totals),
                "mean_correctness": _mean([float(j["correctness"]) for j in judged]),
                "mean_completeness": _mean([float(j["completeness"]) for j in judged]),
                "mean_groundedness": _mean([float(j["groundedness"]) for j in judged]),
                "no_answer": sum(1 for r in selected if r.get("no_answer")),
                "fallback": sum(1 for j in judged if j.get("fallback")),
                "errors": sum(1 for r in selected if r.get("error")),
            }
        )
    return rows


def _sem(values: Sequence[float]) -> float | None:
    """Standard error of the mean; 0.0 for a single value, None for none."""
    if not values:
        return None
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / (len(values) ** 0.5)


# ----------------------------------------------------------------- grid runner

#: The standing regression suite's axes (docs/evaluation.md, "v2 measurement
#: grid"). R3 is not a default: it raises NotImplementedError until Wave 5.
DEFAULT_DEPTHS: tuple[RetrievalDepth, ...] = (
    RetrievalDepth.R0,
    RetrievalDepth.R1,
    RetrievalDepth.R2,
)
DEFAULT_BUDGETS: tuple[int, ...] = (2_500, 10_000, 25_000)

MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "summary.md"

#: Judge-score difference (0-30 scale) below which two cells are not
#: distinguishable on this dataset. Not a p-value: docs/evaluation.md
#: ("Practices", and v1's benchmark summary note) fixes the practice that with
#: ~15 questions per subset the SEM is around 1.5-3 points, so differences
#: smaller than roughly 3 points read as noise rather than as a winner.
NOISE_THRESHOLD = 3.0

_VERDICT_IMPROVED = "improved"
_VERDICT_REGRESSED = "regressed"
_VERDICT_NOISE = "within noise"
_VERDICT_NO_DATA = "no data"


def cell_filename(depth: RetrievalDepth | str, budget_tokens: int) -> str:
    """The artifact name for one grid cell: ``answers_R1_10k.json``.

    Deterministic in (depth, budget) — this is what makes resume possible
    and what keeps the frozen baseline's filename reproducible.
    """
    level = RetrievalDepth.parse(depth)
    budget = int(budget_tokens)
    label = f"{budget // 1000}k" if budget >= 1000 and budget % 1000 == 0 else str(budget)
    return f"answers_{level.value}_{label}.json"


@dataclass
class CellResult:
    """One (depth, budget) cell: where its artifact is and how it ended.

    ``status`` is one of ``done`` (run now), ``resumed`` (a complete
    artifact was already on disk), ``aborted`` (keys exhausted mid-cell) or
    ``pending`` (never attempted, because an earlier cell aborted).
    ``cost`` is ``None`` for a cell this run did not pay for.
    """

    depth: str
    budget_tokens: int
    artifact: str
    status: str = "pending"
    n: int = 0
    judged: int = 0
    mean_total: float | None = None
    sem_total: float | None = None
    cost: float | None = None
    error: str = ""


@dataclass
class GridReport:
    """The outcome of one ``run_grid``: the manifest, in memory."""

    kb: str
    synthesis_level: str
    out_dir: Path
    started_at: str
    depths: list[str] = field(default_factory=list)
    budgets: list[int] = field(default_factory=list)
    cells: list[CellResult] = field(default_factory=list)
    finished_at: str = ""
    aborted: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_FILENAME

    def completed(self) -> list[CellResult]:
        """Cells with a usable artifact: run now, or resumed from disk."""
        return [cell for cell in self.cells if cell.status in ("done", "resumed")]

    def rows(self) -> list[dict[str, Any]]:
        """One printable row per cell, in grid order."""
        return [asdict(cell) for cell in self.cells]

    def manifest(self) -> dict[str, Any]:
        """The manifest as written to disk (baseline shape plus the axes)."""
        return {
            **self.meta,
            "kb": self.kb,
            "synthesis_level": self.synthesis_level,
            "depths": list(self.depths),
            "budgets": list(self.budgets),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "aborted": self.aborted,
            "cells": self.rows(),
        }


def run_grid(
    kb: KnowledgeBase,
    questions: Sequence[GoldQuestion],
    *,
    client: LLMClient,
    depths: Sequence[RetrievalDepth | str] = DEFAULT_DEPTHS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    out_dir: str | Path,
    judge_client: LLMClient | None = None,
    resume: bool = True,
    dataset_path: str | Path | None = None,
    corpus: str = "",
) -> GridReport:
    """Run the depth x budget grid for this KB into ``out_dir``.

    One cell per (depth, budget), in that order; the S axis is the KB's own
    synthesis level, recorded in the manifest — nothing is re-ingested here,
    a second S level means a second KB and a second call.

    Each cell delegates to ``answer_run`` (and to ``judge_answers`` when a
    ``judge_client`` is given) and writes ``cell_filename(depth, budget)``
    in ``out_dir``. With ``resume=True`` a cell whose artifact is already
    complete is not paid for again; an incomplete one is re-run from
    scratch, since a half-answered artifact is cheaper to redo than to
    stitch. ``manifest.json`` is rewritten after every cell, and
    ``summary.md`` once at the end.

    Exhausted keys — raised, or recorded in the artifact meta by
    ``answer_run``/``judge_answers`` — stop the grid: the aborted cell keeps
    its partial artifact, later cells stay ``pending``, and the manifest
    says so. Rerunning with ``resume=True`` when quota is back picks up
    exactly there.
    """
    levels = [RetrievalDepth.parse(depth) for depth in depths]
    sizes = [int(budget) for budget in budgets]
    if not levels or not sizes:
        raise ValueError("run_grid needs at least one depth and one budget")
    if any(size < 1 for size in sizes):
        raise ValueError(f"budgets must be positive, got {sizes}")
    if not questions:
        raise ValueError("run_grid needs at least one gold question")

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    report = GridReport(
        kb=kb.name,
        synthesis_level=kb.synthesis_level.value,
        out_dir=directory,
        started_at=_utc_now(),
        depths=[level.value for level in levels],
        budgets=sizes,
    )
    report.meta = _run_metadata(
        kb,
        client=client,
        judge_client=judge_client,
        dataset_path=dataset_path,
        corpus=corpus,
        questions=questions,
        run_id=directory.name,
        resume=resume,
    )

    for level in levels:
        for size in sizes:
            path = directory / cell_filename(level, size)
            cell = CellResult(depth=level.value, budget_tokens=size, artifact=path.name)
            report.cells.append(cell)
            if report.aborted:
                continue  # recorded as pending: quota, not this cell, stopped it
            _run_cell(
                kb,
                questions,
                cell=cell,
                report=report,
                path=path,
                client=client,
                judge_client=judge_client,
                depth=level,
                budget_tokens=size,
                resume=resume,
            )
            _write_artifact(report.manifest_path, report.manifest())

    report.finished_at = _utc_now()
    _write_artifact(report.manifest_path, report.manifest())
    write_report(directory)
    return report


def _run_cell(
    kb: KnowledgeBase,
    questions: Sequence[GoldQuestion],
    *,
    cell: CellResult,
    report: GridReport,
    path: Path,
    client: LLMClient,
    judge_client: LLMClient | None,
    depth: RetrievalDepth,
    budget_tokens: int,
    resume: bool,
) -> None:
    """Answer + judge one cell, or reuse a complete artifact already on disk."""
    if resume:
        existing = _complete_artifact(path, questions, judged=judge_client is not None)
        if existing is not None:
            _score_cell(cell, existing, status="resumed", cost=None)
            return

    calls_before = len(kb.store.get_llm_calls())
    try:
        artifact = answer_run(
            kb,
            questions,
            client=client,
            depth=depth,
            budget_tokens=budget_tokens,
            out_path=path,
        )
        if judge_client is not None:
            judge_answers(artifact, questions, judge_client=judge_client)
            _write_artifact(path, artifact)
    except AllKeysExhausted as exc:
        # answer_run/judge_answers record this in meta themselves; catching it
        # here covers a client that raises outside their guarded sections.
        cell.status = "aborted"
        cell.error = f"AllKeysExhausted: {exc}"
        report.aborted = cell.error
        return

    cost = sum(float(call["cost"]) for call in kb.store.get_llm_calls()[calls_before:])
    _score_cell(cell, artifact, status="done", cost=cost)

    abort_note = artifact["meta"].get("aborted") or artifact["meta"].get("judge_aborted")
    if abort_note:
        cell.status = "aborted"
        cell.error = str(abort_note)
        report.aborted = str(abort_note)


def _score_cell(
    cell: CellResult, artifact: Mapping[str, Any], *, status: str, cost: float | None
) -> None:
    overall = next((row for row in summarize(artifact) if row["subset"] == "overall"), None)
    cell.status = status
    cell.cost = cost
    if overall is not None:
        cell.n = int(overall["n"])
        cell.judged = int(overall["judged"])
        cell.mean_total = overall["mean_total"]
        cell.sem_total = overall["sem_total"]


def _complete_artifact(
    path: Path, questions: Sequence[GoldQuestion], *, judged: bool
) -> dict[str, Any] | None:
    """A finished cell artifact at ``path``, or None if it must be re-run.

    Complete means: readable, not aborted, one result per question, and —
    when the grid judges — a judgment on every result that has an answer.
    """
    if not path.is_file():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(artifact, dict) or not isinstance(artifact.get("results"), list):
        return None
    meta = artifact.get("meta")
    if not isinstance(meta, dict) or meta.get("aborted") or meta.get("judge_aborted"):
        return None

    results = artifact["results"]
    answered = {r.get("id") for r in results}
    if any(question.id not in answered for question in questions):
        return None
    if judged and any(
        not isinstance(r.get("judgment"), Mapping) for r in results if not r.get("error")
    ):
        return None
    return artifact


def _run_metadata(
    kb: KnowledgeBase,
    *,
    client: LLMClient,
    judge_client: LLMClient | None,
    dataset_path: str | Path | None,
    corpus: str,
    questions: Sequence[GoldQuestion],
    run_id: str,
    resume: bool,
) -> dict[str, Any]:
    """The static half of the manifest: what produced these numbers.

    Judge and embedding/reranker models are recorded because a change to any
    of them invalidates cross-run comparison (docs/evaluation.md: the judge
    is pinned per dataset version).
    """
    from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL

    dataset = Path(dataset_path) if dataset_path is not None else None
    return {
        "run_id": run_id,
        "corpus": corpus,
        "dataset_path": str(dataset) if dataset else "",
        "dataset_sha256": _file_sha256(dataset) if dataset else "",
        "n_questions": len(questions),
        "git_commit": _git_commit(),
        "answer_model": str(getattr(client, "model", "unknown")),
        "judge_model": str(getattr(judge_client, "model", "")) if judge_client else "",
        "embedding_model": kb.store.config_get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "reranker_model": kb.store.config_get("reranker_model", DEFAULT_RERANKER_MODEL),
        "resume": bool(resume),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """Dataset hash: the manifest must pin the questions, not just their path."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_commit() -> str:
    """The commit the run was produced by; ``unknown`` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


# --------------------------------------------------------- baseline comparison


def compare(
    baseline_path: str | Path | Mapping[str, Any],
    run_path: str | Path | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Diff a judged run against a frozen baseline, per subset.

    Each side is a judged run artifact — a path to one, a run directory
    holding exactly one ``answers_*.json``, or an already-loaded artifact.
    One row per subset (simple, complex, overall) carries both means with
    their SEMs, the delta (run minus baseline), the combined SEM
    (``sqrt(sem_b^2 + sem_r^2)``, the two runs being independent samples)
    and a ``verdict``.

    The verdict is ``improved`` / ``regressed`` only when the delta clears
    NOISE_THRESHOLD, else ``within noise`` (``no data`` when a side has
    nothing judged). The threshold is not computed from these two runs: it
    is the practice docs/evaluation.md fixes for this dataset — with ~15
    questions per subset the SEM is 1.5-3 points, so a difference under
    ~3 points on the 0-30 scale is not a result.
    """
    baseline_rows = {row["subset"]: row for row in summarize(_artifact_source(baseline_path))}
    run_rows = {row["subset"]: row for row in summarize(_artifact_source(run_path))}

    rows: list[dict[str, Any]] = []
    for subset in ("simple", "complex", "overall"):
        base = baseline_rows.get(subset)
        current = run_rows.get(subset)
        if base is None and current is None:
            continue
        base_mean = base["mean_total"] if base else None
        run_mean = current["mean_total"] if current else None
        base_sem = (base["sem_total"] if base else None) or 0.0
        run_sem = (current["sem_total"] if current else None) or 0.0
        delta = None if base_mean is None or run_mean is None else run_mean - base_mean
        rows.append(
            {
                "subset": subset,
                "baseline_n": base["judged"] if base else 0,
                "run_n": current["judged"] if current else 0,
                "baseline_mean": base_mean,
                "run_mean": run_mean,
                "baseline_sem": base["sem_total"] if base else None,
                "run_sem": current["sem_total"] if current else None,
                "delta": delta,
                "delta_sem": None if delta is None else (base_sem**2 + run_sem**2) ** 0.5,
                "noise_threshold": NOISE_THRESHOLD,
                "verdict": _verdict(delta),
            }
        )
    return rows


def _verdict(delta: float | None) -> str:
    if delta is None:
        return _VERDICT_NO_DATA
    if abs(delta) < NOISE_THRESHOLD:
        return _VERDICT_NOISE
    return _VERDICT_IMPROVED if delta > 0 else _VERDICT_REGRESSED


def _artifact_source(
    source: str | Path | Mapping[str, Any],
) -> Mapping[str, Any] | Path:
    """Accept an artifact, its path, or a run directory with one artifact."""
    if not isinstance(source, (str, Path)):
        return source
    path = Path(source)
    if path.is_dir():
        artifacts = sorted(path.glob("answers_*.json"))
        if len(artifacts) != 1:
            raise ValueError(
                f"{path}: expected exactly one answers_*.json to compare, found "
                f"{[p.name for p in artifacts]}; name the cell artifact explicitly"
            )
        return artifacts[0]
    return path


# ------------------------------------------------------------ markdown report

#: docs/evaluation.md, "Standing questions the harness must answer": the id,
#: the question, what blocks it in a single grid run (empty when nothing
#: does), and the depths whose cells it needs.
_STANDING_QUESTIONS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "Q1",
        "Does the reranker beat plain RRF ordering at equal budget?",
        "needs a rerank-ablation run; the grid varies depth and budget only",
        (),
    ),
    (
        "Q2",
        "Does any S level beat S0 at realistic budgets?",
        "needs a second grid on a KB at another synthesis level",
        (),
    ),
    ("Q3", "Does orientation (R1) help or hurt vs skipping it?", "", ("R0", "R1")),
    (
        "Q4",
        "Does the repair loop (R2) pay for its extra call on complex questions?",
        "",
        ("R1", "R2"),
    ),
    (
        "Q5",
        "Does graph expansion at R2 add anything over R1 + structural expansion?",
        "needs a graph-ablation run at R2, which this grid does not vary",
        (),
    ),
    ("Q6", "Is R3 worth its latency on hard questions?", "", ("R1", "R3")),
)


def report_markdown(run_dir: str | Path) -> str:
    """Render ``summary.md`` for a run directory (v1's benchmark summary shape).

    Reads the cell artifacts on disk — the manifest only supplies metadata
    and the per-cell cost — so a partial run (some cells missing, some
    unjudged, no manifest at all) renders what exists and says what is
    missing instead of raising.
    """
    directory = Path(run_dir)
    manifest = _read_manifest(directory)
    costs = {
        str(cell.get("artifact")): cell
        for cell in manifest.get("cells", [])
        if isinstance(cell, Mapping)
    }
    artifacts = sorted(
        (path for path in directory.glob("answers_*.json")),
        key=lambda path: path.name,
    )

    lines: list[str] = [f"# Benchmark run: {manifest.get('run_id', directory.name)}", ""]
    lines += _report_header(manifest, len(artifacts))

    cells = [_cell_summary(path, costs.get(path.name)) for path in artifacts]
    if not cells:
        lines += ["## Cells", "", "No cell artifacts in this run directory yet.", ""]
    for subset, title in (
        ("simple", "Simple questions (direct information retrieval)"),
        ("complex", "Complex questions (multi-hop reasoning)"),
        ("overall", "All questions"),
    ):
        table = _score_table(cells, subset)
        if table:
            lines += [f"## {title}", "", *table, ""]

    lines += [
        "**Note on significance:** with ~15 questions per subset the standard error "
        f"of the mean is wide; score differences smaller than {NOISE_THRESHOLD:.0f} points "
        "on the 0-30 scale should be read as noise, not as a winner "
        "(docs/evaluation.md, Practices). The context-token column is what makes "
        "cells comparable at a given cost: the nominal budget is an envelope, the "
        "tokens actually packed are what the model saw.",
        "",
    ]
    lines += _cost_table(cells)
    lines += _standing_questions_section(manifest, cells)
    return "\n".join(lines).rstrip() + "\n"


def write_report(run_dir: str | Path) -> Path:
    """Write ``report_markdown`` to ``run_dir/summary.md`` and return the path."""
    path = Path(run_dir) / SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(run_dir), encoding="utf-8")
    return path


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _report_header(manifest: Mapping[str, Any], n_artifacts: int) -> list[str]:
    fields = [
        ("Knowledge base", manifest.get("kb")),
        ("Synthesis level", manifest.get("synthesis_level")),
        ("Corpus", manifest.get("corpus")),
        ("Dataset", manifest.get("dataset_path")),
        ("Dataset sha256", (manifest.get("dataset_sha256") or "")[:12]),
        ("Questions", manifest.get("n_questions")),
        ("Answer model", manifest.get("answer_model")),
        ("Judge model", manifest.get("judge_model")),
        ("Embedding model", manifest.get("embedding_model")),
        ("Reranker model", manifest.get("reranker_model")),
        ("Commit", manifest.get("git_commit")),
        ("Started", manifest.get("started_at")),
        ("Finished", manifest.get("finished_at")),
    ]
    lines = [f"- **{label}**: {value}" for label, value in fields if value]
    lines.append(f"- **Cells with artifacts**: {n_artifacts}")
    if manifest.get("aborted"):
        lines.append(
            f"- **Aborted**: {manifest['aborted']} — cells left `pending` were never run; "
            "rerun with resume to finish them."
        )
    if manifest.get("note"):
        # A frozen baseline's caveats (a swapped judge, a quota-forced model
        # change) belong in the report that quotes its numbers.
        lines.append(f"- **Note**: {manifest['note']}")
    if not manifest:
        lines.append("- **Manifest**: missing (metadata and per-cell cost unavailable)")
    lines.append("")
    lines.append(
        "Answers were graded by a blind judge (question, gold answer, key points and "
        "candidate answer only — never the depth, the budget or which cell produced it) "
        "on a 0-30 scale: correctness (0-12) + completeness (0-12) + groundedness (0-6)."
    )
    lines.append("")
    return lines


def _cell_summary(path: Path, manifest_cell: Mapping[str, Any] | None) -> dict[str, Any]:
    """One cell: its meta, its per-subset summarize rows, and what it cost."""
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        rows = {row["subset"]: row for row in summarize(artifact)}
        meta = artifact.get("meta", {}) if isinstance(artifact, dict) else {}
        results = artifact.get("results", []) if isinstance(artifact, dict) else []
    except (OSError, json.JSONDecodeError):
        rows, meta, results = {}, {}, []
    tokens = [
        float(r["context_tokens"]) for r in results if isinstance(r.get("context_tokens"), int)
    ]
    return {
        "artifact": path.name,
        "depth": meta.get("depth", "?"),
        "budget": meta.get("budget_tokens", 0),
        "rows": rows,
        "mean_context_tokens": _mean(tokens),
        "cost": (manifest_cell or {}).get("cost"),
        "status": (manifest_cell or {}).get("status", "unknown"),
    }


def _score_table(cells: Sequence[Mapping[str, Any]], subset: str) -> list[str]:
    body = []
    for cell in cells:
        row = cell["rows"].get(subset)
        if row is None:
            continue
        body.append(
            f"| {cell['depth']} | {cell['budget']} | {row['n']} | {row['judged']} | "
            f"{_fmt(cell['mean_context_tokens'], 0)} | {_fmt(row['mean_total'])} | "
            f"±{_fmt(row['sem_total'])} |"
        )
    if not body:
        return []
    return [
        "| Depth | Budget | N | Judged | Avg. context tokens | Avg. score | SEM |",
        "|---|---|---|---|---|---|---|",
        *body,
    ]


def _cost_table(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    if not cells:
        return []
    lines = [
        "## Query cost per cell",
        "",
        "From the KB's `llm_calls` meter (answers + judge). A resumed cell was "
        "not paid for in this run.",
        "",
        "| Depth | Budget | Status | Cost |",
        "|---|---|---|---|",
    ]
    total = 0.0
    for cell in cells:
        cost = cell["cost"]
        total += float(cost or 0.0)
        shown = "—" if cost is None else f"${float(cost):.4f}"
        lines.append(f"| {cell['depth']} | {cell['budget']} | {cell['status']} | {shown} |")
    lines += ["", f"Total metered this run: ${total:.4f}.", ""]
    return lines


def _standing_questions_section(
    manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> list[str]:
    depths = {str(cell["depth"]) for cell in cells}
    lines = [
        "## Standing questions (docs/evaluation.md)",
        "",
        "| # | Question | This run |",
        "|---|---|---|",
    ]
    for qid, question, blocker, needed in _STANDING_QUESTIONS:
        if blocker:
            verdict = f"no — {blocker}"
        elif all(depth in depths for depth in needed):
            verdict = f"yes — compare the {' vs '.join(needed)} rows above"
        else:
            missing = ", ".join(depth for depth in needed if depth not in depths)
            verdict = f"no — this grid has no {missing} cell"
        lines.append(f"| {qid} | {question} | {verdict} |")
    level = manifest.get("synthesis_level")
    if level:
        lines += [
            "",
            f"The S axis is fixed at {level} here: every S question needs a second grid "
            "on a KB ingested at another level (docs/evaluation.md, v2 measurement grid).",
        ]
    lines.append("")
    return lines


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"
