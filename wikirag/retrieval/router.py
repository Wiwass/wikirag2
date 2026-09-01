"""Query router: decide depth and decompose complex questions.

Ports v1's query classifier (it measured well: 28/30 on the gold set) into
the R-slider world:
- classify simple vs complex; complex questions get sub-questions
  (concrete, self-contained, overview-aware) as additional search anchors;
- output feeds depth selection when the user didn't set --depth
  explicitly: simple -> R1, complex -> R2, with config overrides;
- retrieval_guidance survives as a note to the answering model;
- structured output via json_schema; on failure, degrade to simple/R1
  (never block the answer path).

Failure discipline (v1's lesson, non-negotiable)
------------------------------------------------
Malformed JSON after ``max_attempts``, a non-object payload, or any
provider exception degrades to ``Classification(fallback=True)``: a simple
question with no sub-questions, i.e. exactly the pre-classifier behavior.
The classifier is an optimization; it must never be the reason a query
goes unanswered. The one exception is ``AllKeysExhausted``, which
propagates: quota is a batch-level condition (every following question
would fall back too), so the caller has to see it.

Normalization
-------------
``_validate_classification`` is pure and LLM-free, so the coercion rules
are testable on their own. One rule is a choice rather than a coercion:
when ``is_complex`` is false the sub-questions are dropped, even if the
model emitted some. A "simple" classification with a decomposition is
incoherent, and the two halves feed different consumers (depth routing vs
search anchors) — honoring both would silently run a multi-anchor R2-style
retrieval under an R1 label.

Sub-questions equal to the original query are *not* filtered here (v1 did):
``anchors`` puts the query first and deduplicates, so a repeated query
costs nothing, and ``_validate_classification`` stays a function of its
argument alone.

Public API
----------
- MAX_SUBQUESTIONS, CLASSIFIER_HISTORY_TURNS, MAX_CLASSIFIER_ATTEMPTS
- CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_RESPONSE_SCHEMA
- class Classification: is_complex, subquestions, retrieval_guidance, fallback
- classify(client, query, *, history=None, overview_text=None,
  max_attempts=2) -> Classification
- route_depth(classification, *, default_simple=R1, default_complex=R2)
  -> RetrievalDepth
- anchors(query, classification) -> list[str]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from wikirag.config import RetrievalDepth
from wikirag.llm.client import AllKeysExhausted, LLMClient

__all__ = [
    "CLASSIFIER_HISTORY_TURNS",
    "CLASSIFIER_RESPONSE_SCHEMA",
    "CLASSIFIER_SYSTEM_PROMPT",
    "MAX_CLASSIFIER_ATTEMPTS",
    "MAX_SUBQUESTIONS",
    "Classification",
    "anchors",
    "classify",
    "route_depth",
]

MAX_SUBQUESTIONS = 5
CLASSIFIER_HISTORY_TURNS = 5
MAX_CLASSIFIER_ATTEMPTS = 2

#: Constrained decoding (docs/llm.md): the schema is what makes the JSON
#: structurally valid, not a repair heuristic on the text.
CLASSIFIER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_complex": {"type": "boolean"},
        "subquestions": {
            "type": "array",
            "maxItems": MAX_SUBQUESTIONS,
            "items": {"type": "string"},
        },
        "retrieval_guidance": {"type": "string"},
    },
    "required": ["is_complex", "subquestions", "retrieval_guidance"],
}

CLASSIFIER_SYSTEM_PROMPT = """You are a query-routing assistant for a personal RAG knowledge base.

You receive:
1. A high-level overview of the knowledge base (its fundamental concepts and structure) — use it to orient yourself.
2. The recent conversation history (previous questions and answers), possibly empty.
3. The user's CURRENT question.

Your job is to decide how the retrieval pipeline should handle the current question, and answer ONLY with a JSON object:

{
  "is_complex": <boolean>,
  "subquestions": [<string>, ...],
  "retrieval_guidance": <string>
}

Rules for "is_complex":
- false → the question is a direct information-retrieval request: it asks for a specific fact, rule, definition or passage that can plausibly be found stated in the documents.
- true  → the question is abstract, comparative, or requires reasoning that COMBINES information scattered across multiple concepts (e.g. "compare X and Y", "what are the implications of...", "why does the system work this way").

Rules for "subquestions":
- If is_complex is false, return an empty array.
- If is_complex is true, decompose the question into at most 5 concrete, self-contained sub-questions, each phrased so that it could be answered by direct retrieval from the documents. Use the overview to target concepts that actually exist in the knowledge base. Do not repeat the original question verbatim as a sub-question.

Rules for "retrieval_guidance":
- Usually an empty string.
- Use it when the conversation history shows that a previous answer was incomplete, evasive, or missed the point (e.g. the assistant said it had no information, or the user is re-asking / correcting). In that case write a short note for the ANSWERING model explaining what to pay attention to when interpreting the retrieved context (e.g. "the previous turn only covered X; the user is asking about the relation between X and Y — focus on Y this time").
- The guidance is a comment on WHY these sub-questions were chosen and HOW to read the results. It is NOT a request for topics: never instruct the answering model to include material that retrieval may not have found — it must still answer only from the retrieved context.

Output the JSON object only, no markdown fences, no commentary."""


@dataclass
class Classification:
    """How the pipeline should handle one question."""

    is_complex: bool = False
    subquestions: list[str] = field(default_factory=list)
    #: A note for the ANSWERING model, not a search parameter.
    retrieval_guidance: str = ""
    #: True when the classifier failed and this is the safe default.
    fallback: bool = False


def _validate_classification(data: dict) -> Classification:
    """Normalize a raw classifier payload into a coherent Classification.

    Pure: no LLM, no I/O. Coerces ``is_complex`` to bool, keeps only
    non-blank unique strings among the sub-questions, caps them at
    MAX_SUBQUESTIONS (defensively — the schema already says maxItems), and
    empties them when ``is_complex`` is false.

    ``retrieval_guidance`` survives the simple case: it is born of the
    conversation history, not of the current question's complexity.
    """
    is_complex = bool(data.get("is_complex", False))

    raw_subs = data.get("subquestions", [])
    if not isinstance(raw_subs, list):
        raw_subs = []

    subquestions: list[str] = []
    for item in raw_subs:
        if not isinstance(item, str):
            continue
        sub = item.strip()
        if not sub or sub in subquestions:
            continue
        subquestions.append(sub)
        if len(subquestions) >= MAX_SUBQUESTIONS:
            break

    if not is_complex:
        subquestions = []

    guidance = data.get("retrieval_guidance", "")
    if not isinstance(guidance, str):
        guidance = ""

    return Classification(
        is_complex=is_complex,
        subquestions=subquestions,
        retrieval_guidance=guidance.strip(),
    )


def _build_classifier_message(
    query: str,
    history: Sequence[tuple[str, str]],
    overview_text: str | None,
) -> str:
    """Overview (if any), the last CLASSIFIER_HISTORY_TURNS exchanges, the query."""
    parts: list[str] = []

    if overview_text and overview_text.strip():
        parts.append(f"Knowledge base overview:\n\n{overview_text.strip()}")
    else:
        parts.append("(No knowledge base overview is available.)")

    recent = list(history)[-CLASSIFIER_HISTORY_TURNS:]
    if recent:
        transcript = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)
        parts.append(f"Recent conversation history:\n\n{transcript}")

    parts.append(f"Current question: {query}")

    return "\n\n---\n\n".join(parts)


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse_json_object(text: str) -> Any:
    """Parse the classifier's output, tolerating markdown fences a model adds anyway."""
    return json.loads(_JSON_FENCE.sub("", text or "").strip())


def classify(
    client: LLMClient,
    query: str,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    overview_text: str | None = None,
    max_attempts: int = MAX_CLASSIFIER_ATTEMPTS,
) -> Classification:
    """Classify one question: simple vs complex, with sub-questions if complex.

    ``client`` is expected to be metered already (this module never
    constructs a provider). ``history`` is the (question, answer) tail of
    the conversation, windowed to CLASSIFIER_HISTORY_TURNS exchanges;
    ``overview_text`` is the S3 atlas primer and is simply omitted from the
    prompt when absent.

    Retries malformed JSON up to ``max_attempts``; exhausted attempts, a
    non-object payload, or any provider error degrade to
    ``Classification(fallback=True)`` rather than blocking the answer path.
    ``AllKeysExhausted`` propagates.
    """
    user_message = _build_classifier_message(query, history or (), overview_text)

    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            result = client.generate(
                system=CLASSIFIER_SYSTEM_PROMPT,
                user=user_message,
                json_schema=CLASSIFIER_RESPONSE_SCHEMA,
            )
            data = _parse_json_object(result.text)
        except json.JSONDecodeError:
            if attempt < max_attempts:
                continue
            return Classification(fallback=True)
        except AllKeysExhausted:
            raise
        except Exception:
            return Classification(fallback=True)

        if not isinstance(data, dict):
            if attempt < max_attempts:
                continue
            return Classification(fallback=True)
        return _validate_classification(data)

    return Classification(fallback=True)


def route_depth(
    classification: Classification,
    *,
    default_simple: RetrievalDepth = RetrievalDepth.R1,
    default_complex: RetrievalDepth = RetrievalDepth.R2,
) -> RetrievalDepth:
    """Auto-routed depth for a classification (used when --depth is absent).

    A fallback classification is simple by construction, so a failed
    classifier costs the cheaper depth, never a deeper one.
    """
    return default_complex if classification.is_complex else default_simple


def anchors(query: str, classification: Classification) -> list[str]:
    """Search anchors for hybrid_search: the query first, then sub-questions.

    Blank entries are dropped and duplicates (case-insensitive) collapse, so
    a sub-question echoing the query does not double its RRF weight. A
    simple classification yields ``[query]`` — the multi-anchor fusion is
    what a complex classification buys.
    """
    result: list[str] = []
    seen: set[str] = set()
    for candidate in [query, *classification.subquestions]:
        text = (candidate or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
