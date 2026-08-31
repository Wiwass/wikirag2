"""Answering: turn a query + retrieved context into a grounded reply.

Shared by `wikirag query` and `wikirag chat` (and by eval runners):
- system prompts: grounding rules (facts only from context, cite sources,
  say so when the context is insufficient), ported from v1;
- history is continuity-only, never evidence; fresh retrieval per turn;
  history windowed to a cap (v1 let it grow unboundedly);
- the no-answer path: when rerank's top score is below threshold, reply
  "not in the KB" without stuffing weak context.

Depth dispatch: R0-R2 -> retrieval pipeline + single generate call;
R3 -> wikirag.agent.loop.

Two prompt features of the spec belong to later waves and have no code
here yet: the complex-question addendum listing sub-questions (needs
`retrieval.router`, Wave 3) and the overview primer (needs an S3 atlas,
Wave 4, and `assembly` reserves its slot).

Pipeline (the R0 path, currently the only one)
---------------------------------------------
hybrid_search([query], k=pool_size) -> rerank -> no-answer gate ->
assemble under the context budget -> one client.generate.

R0, R1 and R2 all run this path today: R1's orientation and R2's
decomposition/repair arrive in Wave 3, and until they exist the three
levels are deliberately equivalent (the level is still accepted, logged
and reported, so nothing downstream has to change when they diverge).
R3 raises NotImplementedError until Wave 5 wires `agent.loop`.

Budget split
------------
`budget_tokens` is the whole prompt+answer envelope. A fifth of it, capped
at RESERVED_ANSWER_TOKENS, is reserved for the model's own output (and
passed as `max_output_tokens`); the rest is the context budget handed to
`assemble`. The proportional cap keeps small budgets usable: an eval run
with a 2k envelope still retrieves, it just gets a shorter answer.

The system prompt and the history block are not charged against the
budget — they are bounded by construction (a constant, and HISTORY_WINDOW
exchanges), while the context is the part that would otherwise grow with
the pool.

Grounding
---------
The context entries are numbered by `assembly` ([1], [2], ...); the
answer cites those numbers. History is included for continuity only,
windowed to the last HISTORY_WINDOW exchanges and marked as history
rather than evidence. Every query is logged (including no-answer
outcomes): usage is what drives lazy consolidation (D11).

The client is provided by the caller and is expected to be metered
already (the CLI wraps it): this module never constructs providers.

Public API
----------
- GROUNDING_SYSTEM_PROMPT, NO_ANSWER_TEXT
- HISTORY_WINDOW, RESERVED_ANSWER_TOKENS, DEFAULT_BUDGET_TOKENS,
  DEFAULT_POOL_SIZE
- class Answer: text, provenance, usage, no_answer, depth, context_tokens
- answer(kb, query, *, depth=R1, client, history=None,
  budget_tokens=10_000, pool_size=50) -> Answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from wikirag.config import KnowledgeBase, RetrievalDepth
from wikirag.llm.client import LLMClient, Usage
from wikirag.retrieval.assembly import ContextEntry, assemble
from wikirag.retrieval.rerank import no_answer, rerank
from wikirag.retrieval.search import hybrid_search

__all__ = [
    "DEFAULT_BUDGET_TOKENS",
    "DEFAULT_POOL_SIZE",
    "GROUNDING_SYSTEM_PROMPT",
    "HISTORY_WINDOW",
    "NO_ANSWER_TEXT",
    "RESERVED_ANSWER_TOKENS",
    "Answer",
    "answer",
]

#: Exchanges (question + reply) kept from the conversation history. v1 let
#: the transcript grow for the whole session, which quietly ate the budget
#: and dragged old topics into fresh retrieval.
HISTORY_WINDOW = 6

#: Upper bound on the tokens reserved for the answer itself; the effective
#: reservation is min(this, budget_tokens // 5). See "Budget split".
RESERVED_ANSWER_TOKENS = 2_000

DEFAULT_BUDGET_TOKENS = 10_000
DEFAULT_POOL_SIZE = 50

GROUNDING_SYSTEM_PROMPT = """You are a research assistant answering questions from a personal knowledge base.

You may receive two kinds of input, and they are not equal:
1. Conversation history — earlier questions and your own earlier replies, given only so you can resolve what the user is referring to ("it", "that one", follow-ups). It is never evidence.
2. Knowledge-base context — numbered entries ([1], [2], ...) retrieved for the current question. These are your only source of facts.

Rules:
- Answer the current question using only the facts stated in the numbered context entries.
- Cite the entries you rely on inline, as [1] or [2][5], next to the claim they support.
- Never cite a number that is not in the context, and never invent a source, a document title or a quotation.
- If the context does not contain enough to answer, say so plainly and name what is missing. A partial answer is fine as long as you mark where the context stops.
- Be concise and precise: no filler, no restating the question."""

NO_ANSWER_TEXT = (
    "I could not find anything relevant to this question in the knowledge base. "
    "Nothing retrieved was close enough to answer from, so rather than guess from "
    "weak material: the knowledge base does not seem to cover this. Try rephrasing "
    "it, or ingest a document that covers the topic."
)

_HISTORY_HEADER = "Conversation history (for continuity only, not evidence):"
_CONTEXT_HEADER = "Knowledge-base context:"
_EMPTY_CONTEXT = "(No knowledge-base context could be assembled for this question.)"
_PART_SEPARATOR = "\n\n---\n\n"


@dataclass
class Answer:
    """One answered query: the reply, what grounds it, and what it cost."""

    text: str
    provenance: list[ContextEntry] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    no_answer: bool = False
    depth: RetrievalDepth = RetrievalDepth.R1
    context_tokens: int = 0


def answer(
    kb: KnowledgeBase,
    query: str,
    *,
    depth: RetrievalDepth | str = RetrievalDepth.R1,
    client: LLMClient,
    history: Sequence[tuple[str, str]] | None = None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> Answer:
    """Retrieve, ground, and generate one reply.

    `history` is a sequence of (question, reply) exchanges, oldest first;
    only the last HISTORY_WINDOW are shown to the model, and never as
    evidence — retrieval runs fresh on `query` alone every turn.

    `client` must already be metered (docs/llm.md: every LLM call goes
    through MeteredClient); `budget_tokens` is the prompt+answer envelope,
    split as described in the module docstring.

    Raises NotImplementedError for R3, which needs the agent loop.
    """
    level = RetrievalDepth.parse(depth)
    if level is RetrievalDepth.R3:
        raise NotImplementedError("R3 arrives in Wave 5")

    candidates = hybrid_search(kb, [query], k=pool_size)
    scored = rerank(query, candidates)
    if no_answer(scored):
        # Logged before returning: a query that found nothing is exactly the
        # signal that tells the KB what it is missing.
        kb.store.log_query(query=query, depth=level.value, no_answer=True)
        return Answer(text=NO_ANSWER_TEXT, no_answer=True, depth=level)

    reserved = _reserved_tokens(budget_tokens)
    context = assemble(kb, scored, budget_tokens - reserved)

    # Logged before the call, so a provider failure still leaves the trace.
    kb.store.log_query(query=query, depth=level.value, no_answer=False)
    result = client.generate(
        system=GROUNDING_SYSTEM_PROMPT,
        user=build_user_message(query, context.text, history),
        max_output_tokens=reserved,
    )
    return Answer(
        text=result.text,
        provenance=list(context.entries),
        usage=result.usage,
        depth=level,
        context_tokens=context.token_count,
    )


def _reserved_tokens(budget_tokens: int) -> int:
    """Tokens held back from the budget for the model's own output."""
    return max(1, min(RESERVED_ANSWER_TOKENS, budget_tokens // 5))


def build_user_message(
    query: str,
    context_text: str,
    history: Sequence[tuple[str, str]] | None = None,
) -> str:
    """The user turn: windowed history, numbered context, then the question."""
    parts: list[str] = []
    recent = list(history or ())[-HISTORY_WINDOW:]
    if recent:
        transcript = "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in recent)
        parts.append(f"{_HISTORY_HEADER}\n\n{transcript}")
    if context_text.strip():
        parts.append(f"{_CONTEXT_HEADER}\n\n{context_text}")
    else:
        parts.append(_EMPTY_CONTEXT)
    parts.append(f"Question: {query}")
    return _PART_SEPARATOR.join(parts)
