"""Enrich stage — OPEN RESEARCH AREA (docs/ingestion.md).

What is fixed is only the contract:
- Input: a document's chunks (numbered) + the KB's existing keyword index
  (embedding-shortlisted, never the full index in a prompt — v1's O(index)
  cost and runaway-filter failure mode must not return).
- Output: chunk_keywords tags, optional per-chunk context strings for
  embedding, new keyword proposals. All by reference; no copied text.

Candidate methods to implement and benchmark (each behind this interface):
- contextual chunk enrichment (Anthropic-style), possibly via prompt
  caching or a local small model;
- local 4-8B model tagging with API spot-checks;
- deferred/lazy variants that tag coarsely at ingest and refine on demand.

Implemented: ``embedding_shortlist`` (the first method)
------------------------------------------------------
The cheapest candidate of docs/ingestion.md, and the default. Per batch of
chunks:

1. Every existing keyword name is embedded (locally, cached per process);
   every chunk vector is read back from the KB's vector index, which the
   `index` stage already filled — only chunks missing from it are encoded.
2. Each chunk shortlists its ``SHORTLIST_PER_CHUNK`` nearest keywords by
   cosine; the batch prompt carries the union, capped at ``SHORTLIST_MAX``
   by best similarity. Prompt size is therefore O(batch), never O(index).
3. One structured call per batch returns, per chunk number, the candidates
   it picked plus at most ``MAX_NEW_PER_CHUNK`` proposals for concepts the
   shortlist genuinely lacked. Nothing else: the model is never asked for
   source text, and its answer carries none (rule 1).
4. The reply is validated before anything is written: a name in
   ``keywords`` that was not on the shortlist is a hallucination and is
   dropped; a ``new`` name must look like a keyword (non-empty, at most
   ``MAX_KEYWORD_WORDS`` words / ``MAX_KEYWORD_CHARS`` characters), which
   also makes an echoed passage unusable as a tag. This is the replacement
   for v1's runaway filter: the failure mode costs one dropped tag, never
   a polluted index.

Tags are persisted per batch, so a run stopped by quota exhaustion keeps
every batch it paid for.

Canonicalization (conservative, stdlib-only)
--------------------------------------------
``canonical_keyword`` folds case, collapses whitespace, strips edge
punctuation and singularizes *only the last word* of the phrase by the
trivial English rules (``-ies`` -> ``-y``, ``-sses/-shes/-ches/-xes/-zes``
-> drop ``-es``, trailing ``-s`` unless the word ends in ``ss``, ``us`` or
``is``). So "Covenant" / "covenants" / " Covenants " are one concept, and
"vis", "Ars", "status" are left alone. Nothing else is folded: no
stemming, no article stripping, no synonym merging — near-duplicate
merging is S3 link curation's job (docs/synthesis.md), not a silent
ingest-time rewrite. The fold is passed to ``store.upsert_keyword`` as the
canonical name, so the display name stays the form first seen.

Idempotency
-----------
Per chunk, not per document: a chunk that already has at least one
keyword row is considered tagged and is skipped. A document whose chunks
are all tagged makes no LLM call at all (and constructs no provider), and
a run interrupted mid-document resumes at the first untagged chunk —
the same "existing rows prove completion" discipline the other stages use.
A chunk the model returned nothing for stays untagged and is retried on
the next run, which is the intended behavior for a bad reply.

Cost attribution (docs/ingestion.md evaluation criterion 2)
-----------------------------------------------------------
The stage never builds a provider: ``_client_factory`` hands it a
``MeteredClient`` labelled ``stage="enrich"``, ``role="enrich"``, so every
call lands in ``llm_calls`` and ``wikirag costs <kb> --by stage,role,model``
prints the enrich row (calls, tokens in/out, cost, wall time) for the
build. One KB build runs exactly one method — the one named by the
``enrich_method`` kb_config row — so comparing methods means building the
same corpus under each and diffing that row; the KB's config records which
method produced it.

Public API
----------
- ``canonical_keyword(name) -> str``
- ``class EnrichResult``: tags, new_keywords, rejected, calls
- ``class EnrichMethod(Protocol)``: ``(kb, doc_id, chunks, *, client) ->
  EnrichResult``; the method persists what it accepts, per batch
- ``METHODS``, ``register_method(name, fn)``, ``get_method(name)``
- ``embedding_shortlist_tagging(...)``: the default method
- ``enrich_stage(kb, doc_id) -> None``: the runner-compatible stage
- kb_config keys: ``ENRICH_METHOD_KEY``, ``ENRICH_MODEL_KEY``
- tuning constants: ``SHORTLIST_PER_CHUNK``, ``SHORTLIST_MAX``,
  ``BATCH_CHUNKS``, ``BATCH_MAX_TOKENS``, ``MAX_KEYWORDS_PER_CHUNK``,
  ``MAX_NEW_PER_CHUNK``, ``MAX_KEYWORD_WORDS``, ``MAX_KEYWORD_CHARS``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from wikirag.ingestion.chunking import estimate_tokens
from wikirag.llm.client import LLMClient
from wikirag.llm.meter import MeteredClient

if TYPE_CHECKING:
    from wikirag.config import KnowledgeBase
    from wikirag.llm.embeddings import EmbedderLike

log = logging.getLogger(__name__)

#: kb_config keys this stage honors (with their defaults).
ENRICH_METHOD_KEY = "enrich_method"
ENRICH_MODEL_KEY = "enrich_model"
DEFAULT_METHOD = "embedding_shortlist"
DEFAULT_ENRICH_MODEL = "gemini-3.5-flash-lite"

ENRICH_STAGE = "enrich"
ENRICH_ROLE = "enrich"

#: Shortlist size per chunk, and the cap on the batch-wide union that
#: actually reaches the prompt. A few dozen names is what "the LLM only
#: ever sees a few dozen candidates" means; the cap is what keeps prompt
#: cost O(batch) instead of O(index).
SHORTLIST_PER_CHUNK = 20
SHORTLIST_MAX = 60

#: Chunks per call, and the input-token ceiling that overrides the count
#: when chunks are unusually large. Batching is where the per-call
#: overhead (system prompt + shortlist) is amortized.
BATCH_CHUNKS = 8
BATCH_MAX_TOKENS = 4_000

MAX_KEYWORDS_PER_CHUNK = 6
MAX_NEW_PER_CHUNK = 2
#: A keyword is a short noun phrase. These bounds are also what makes an
#: echoed passage unusable as a tag (rule 1: no verbatim echo).
MAX_KEYWORD_WORDS = 6
MAX_KEYWORD_CHARS = 60

_JSON_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_EDGE_PUNCT = " \t\r\n\"'`*_.,;:!?()[]{}"

SYSTEM_PROMPT = (
    "You tag passages from a reference library with concept keywords, so "
    "that a reader can find every passage about a concept.\n\n"
    "Rules:\n"
    "1. For each numbered chunk, pick the CANDIDATES the chunk is really "
    "about, copied exactly as listed. Pick none rather than a weak one.\n"
    "2. Only when a central concept of the chunk is missing from the "
    f"candidates, propose it in \"new\": at most {MAX_NEW_PER_CHUNK} per "
    f"chunk, each a noun phrase of at most {MAX_KEYWORD_WORDS} words "
    "naming a concept, not a sentence and not a fact.\n"
    f"3. At most {MAX_KEYWORDS_PER_CHUNK} keywords per chunk in total.\n"
    "4. Answer with chunk numbers and keyword names only. Never repeat, "
    "quote, translate or summarize the passage text."
)

#: Chunk numbers and keyword names, nothing else — the schema is where
#: "no verbatim echo" is enforced structurally rather than by asking.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk": {"type": "integer"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "new": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["chunk", "keywords"],
            },
        }
    },
    "required": ["chunks"],
}


# ------------------------------------------------------------ canonicalization


def _singular(word: str) -> str:
    """Trivial English plural fold. Conservative: unknown shapes stay."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def canonical_keyword(name: str) -> str:
    """The identity a keyword is deduped by: case, spacing, trivial plural.

    Returns "" for anything that is not a usable name; callers treat that
    as a rejection.
    """
    text = " ".join(str(name).strip().lower().split()).strip(_EDGE_PUNCT)
    if not text:
        return ""
    words = text.split(" ")
    words[-1] = _singular(words[-1])
    return " ".join(w for w in words if w)


def _display_name(name: str) -> str:
    """The form stored as the keyword's label: trimmed, spacing collapsed."""
    return " ".join(str(name).strip().split()).strip(_EDGE_PUNCT)


def _is_keyword_shaped(name: str) -> bool:
    """Shape gate for a proposed new keyword (also the anti-echo bound)."""
    display = _display_name(name)
    return (
        bool(canonical_keyword(display))
        and len(display) <= MAX_KEYWORD_CHARS
        and len(display.split(" ")) <= MAX_KEYWORD_WORDS
    )


# -------------------------------------------------------------------- registry


class EnrichMethod(Protocol):
    """One candidate enrichment method (docs/ingestion.md).

    Receives the chunks that still need tagging and an already-metered
    client; persists what it accepts (per batch, so an interrupted run
    keeps completed work) and returns a summary of what it did.
    """

    def __call__(
        self,
        kb: "KnowledgeBase",
        doc_id: str,
        chunks: Sequence[Mapping[str, Any]],
        *,
        client: LLMClient,
    ) -> "EnrichResult": ...


METHODS: dict[str, EnrichMethod] = {}


def register_method(name: str, fn: EnrichMethod) -> None:
    """Bind a method name to its implementation; `enrich_method` selects it."""
    METHODS[name] = fn


def get_method(name: str) -> EnrichMethod:
    method = METHODS.get(name)
    if method is None:
        raise RuntimeError(
            f"unknown enrich method {name!r}; registered: {sorted(METHODS)}"
        )
    return method


@dataclass
class EnrichResult:
    """What one enrichment pass did — the method's measurable surface.

    `tags` maps chunk ID to the keyword names written for it, `rejected`
    holds every name dropped by validation (hallucinated candidates,
    malformed proposals): the failure-surface number of criterion 4.
    """

    tags: dict[str, list[str]] = field(default_factory=dict)
    new_keywords: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    calls: int = 0


# ------------------------------------------------------- embeddings + shortlist

# Keyword-name vectors are pure functions of (model, canonical name), so they
# are cached for the process: without this every document re-encodes the whole
# keyword index locally, which is the dominant CPU cost of this method.
_keyword_vector_cache: dict[tuple[str, str], np.ndarray] = {}


def _resolve_embedder(kb: "KnowledgeBase") -> tuple[str, "EmbedderLike"]:
    """The KB's embedding model, through the package-level embedder seam."""
    from wikirag import ingestion
    from wikirag.llm.embeddings import DEFAULT_EMBEDDING_MODEL

    model = kb.store.config_get(ingestion.EMBEDDING_MODEL_KEY, DEFAULT_EMBEDDING_MODEL)
    assert model is not None
    # Read at call time: tests and offline smoke runs swap this one seam for
    # both the index stage and this one.
    return model, ingestion._embedder_factory(model)


def _keyword_matrix(
    embedder: "EmbedderLike", model: str, canonicals: Sequence[str]
) -> np.ndarray:
    missing = [c for c in canonicals if (model, c) not in _keyword_vector_cache]
    if missing:
        for canonical, vector in zip(missing, embedder.encode_passages(missing)):
            _keyword_vector_cache[(model, canonical)] = vector
    if not canonicals:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.stack([_keyword_vector_cache[(model, c)] for c in canonicals])


def _chunk_matrix(
    kb: "KnowledgeBase",
    chunks: Sequence[Mapping[str, Any]],
    embedder: "EmbedderLike",
    model: str,
) -> np.ndarray:
    """Chunk vectors, reused from the index stage's work where possible."""
    from wikirag.storage.vectors import VectorIndex

    index = VectorIndex.open(kb.vectors_dir, model, embedder.dim)
    vectors: list[np.ndarray | None] = [
        index.vector_of(("chunk", str(c["id"]))) for c in chunks
    ]
    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        encoded = embedder.encode_passages([str(chunks[i]["text"]) for i in missing])
        for i, vector in zip(missing, encoded):
            vectors[i] = vector
    return np.stack([v for v in vectors if v is not None])


def _shortlist(
    chunk_vectors: np.ndarray, names: Sequence[str], keyword_vectors: np.ndarray
) -> list[str]:
    """Union of each chunk's nearest keywords, best similarity first."""
    if not names or chunk_vectors.size == 0:
        return []
    scores = keyword_vectors @ chunk_vectors.T  # (keywords, chunks) cosines
    per_chunk = min(SHORTLIST_PER_CHUNK, len(names))
    picked: set[int] = set()
    for column in range(scores.shape[1]):
        column_scores = scores[:, column]
        top = np.argpartition(-column_scores, per_chunk - 1)[:per_chunk]
        picked.update(int(row) for row in top)
    best = scores.max(axis=1)
    ranked = sorted(picked, key=lambda row: (-float(best[row]), names[row]))
    return [names[row] for row in ranked[:SHORTLIST_MAX]]


def _batches(chunks: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Split into calls: BATCH_CHUNKS chunks, or fewer under the token cap."""
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    tokens = 0
    for chunk in chunks:
        size = int(chunk["token_count"] or 0) or estimate_tokens(str(chunk["text"]))
        if current and (len(current) >= BATCH_CHUNKS or tokens + size > BATCH_MAX_TOKENS):
            batches.append(current)
            current, tokens = [], 0
        current.append(chunk)
        tokens += size
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------- prompt


def build_user_message(
    chunks: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> str:
    """The batch prompt: the shortlist, then the numbered chunks."""
    listing = (
        "\n".join(f"- {name}" for name in candidates)
        if candidates
        else "(none yet — this knowledge base has no keywords, propose new ones)"
    )
    body = "\n\n".join(
        f"[{n}] {str(chunk['text']).strip()}" for n, chunk in enumerate(chunks, 1)
    )
    return f"CANDIDATES:\n{listing}\n\nCHUNKS:\n{body}"


def _parse_reply(text: str) -> list[Mapping[str, Any]]:
    """Chunk entries from the model's JSON; tolerates a fenced block."""
    data = json.loads(_JSON_FENCE.sub("", text or "").strip())
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    entries = data.get("chunks")
    if not isinstance(entries, list):
        raise ValueError("reply has no 'chunks' array")
    return [e for e in entries if isinstance(e, Mapping)]


def _names(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


# ---------------------------------------------------------------- persistence


def _persist(kb: "KnowledgeBase", tags: Mapping[str, Sequence[str]]) -> None:
    """Write one batch's tags by reference, canonicalizing every name."""
    pairs: list[tuple[str, int]] = []
    for chunk_id, names in tags.items():
        for name in names:
            keyword_id = kb.store.upsert_keyword(name, canonical_keyword(name))
            pairs.append((chunk_id, keyword_id))
    if pairs:
        kb.store.tag_chunks(pairs)


# ------------------------------------------------------------- the first method


def embedding_shortlist_tagging(
    kb: "KnowledgeBase",
    doc_id: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    client: LLMClient,
) -> EnrichResult:
    """Tag chunks against an embedding-shortlisted slice of the keyword index.

    One structured call per batch of chunks; the prompt carries only the
    shortlisted candidate names, and every returned name is validated
    against that shortlist (or the new-keyword shape rules) before a tag
    is written. Provider failures propagate — including AllKeysExhausted,
    which stops the runner's batch — while a malformed reply costs only
    its own batch, whose chunks stay untagged and are retried next run.
    """
    result = EnrichResult()
    if not chunks:
        return result

    model, embedder = _resolve_embedder(kb)
    known = {str(row["canonical_name"]): str(row["name"]) for row in kb.store.list_keywords()}
    canonicals = list(known)
    keyword_vectors = _keyword_matrix(embedder, model, canonicals)
    chunk_vectors = _chunk_matrix(kb, chunks, embedder, model)

    offset = 0
    for batch in _batches(chunks):
        vectors = chunk_vectors[offset : offset + len(batch)]
        offset += len(batch)
        candidate_canonicals = _shortlist(vectors, canonicals, keyword_vectors)
        # The prompt shows display names; validation happens on canonicals.
        shortlist = {c: known[c] for c in candidate_canonicals}

        response = client.generate(
            system=SYSTEM_PROMPT,
            user=build_user_message(batch, list(shortlist.values())),
            json_schema=RESPONSE_SCHEMA,
        )
        result.calls += 1
        try:
            entries = _parse_reply(response.text)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("enrich: unusable reply for %s (%s); chunks left untagged", doc_id, exc)
            continue

        batch_tags: dict[str, list[str]] = {}
        for entry in entries:
            number = entry.get("chunk")
            if not isinstance(number, int) or not 1 <= number <= len(batch):
                result.rejected.append(str(number))
                continue
            chunk_id = str(batch[number - 1]["id"])
            if chunk_id in batch_tags:
                continue  # duplicate entry for the same chunk: first one wins
            accepted: list[str] = []
            seen: set[str] = set()

            for raw in _names(entry.get("keywords")):
                canonical = canonical_keyword(raw)
                if canonical not in shortlist:
                    # Not offered: a hallucinated tag, dropped before it can
                    # pollute the index (v1's runaway filter, priced at one tag).
                    result.rejected.append(_display_name(raw))
                    continue
                if canonical not in seen and len(accepted) < MAX_KEYWORDS_PER_CHUNK:
                    seen.add(canonical)
                    accepted.append(shortlist[canonical])

            proposed = 0
            for raw in _names(entry.get("new")):
                canonical = canonical_keyword(raw)
                if canonical in known or canonical in seen:
                    # Already a concept of this KB: a merge, not a proposal.
                    if canonical not in seen and len(accepted) < MAX_KEYWORDS_PER_CHUNK:
                        seen.add(canonical)
                        accepted.append(known.get(canonical, _display_name(raw)))
                    continue
                if not _is_keyword_shaped(raw):
                    result.rejected.append(_display_name(raw))
                    continue
                if proposed >= MAX_NEW_PER_CHUNK or len(accepted) >= MAX_KEYWORDS_PER_CHUNK:
                    result.rejected.append(_display_name(raw))
                    continue
                display = _display_name(raw)
                proposed += 1
                seen.add(canonical)
                accepted.append(display)
                # Visible to the rest of this run: a new concept is reused by
                # later chunks and batches instead of being proposed twice.
                known[canonical] = display
                canonicals.append(canonical)
                keyword_vectors = _keyword_matrix(embedder, model, canonicals)
                result.new_keywords.append(display)

            if accepted:
                batch_tags[chunk_id] = accepted

        _persist(kb, batch_tags)
        result.tags.update(batch_tags)

    return result


register_method(DEFAULT_METHOD, embedding_shortlist_tagging)


# ----------------------------------------------------------------- the stage


def _default_client(kb: "KnowledgeBase") -> LLMClient:
    """The metered client for the `enrich` role of this KB."""
    # Imported here so a KB whose enrich never runs need not have google-genai.
    from wikirag.llm.gemini import GeminiClient

    model = kb.store.config_get(ENRICH_MODEL_KEY, DEFAULT_ENRICH_MODEL)
    assert model is not None
    return MeteredClient(
        GeminiClient(model), store=kb.store, stage=ENRICH_STAGE, role=ENRICH_ROLE
    )


# Seam for tests and offline smoke runs, mirroring ingestion._embedder_factory:
# the stage never constructs a provider itself.
_client_factory: Callable[["KnowledgeBase"], LLMClient] = _default_client


def enrich_stage(kb: "KnowledgeBase", doc_id: str) -> None:
    """Chunks -> chunk_keywords tags, with the KB's configured method.

    Idempotent per chunk: chunks that already carry a keyword are skipped,
    so a re-run of a tagged document makes no call and builds no client,
    and an interrupted run resumes at the first untagged chunk.
    """
    store = kb.store
    todo = [c for c in store.get_document_chunks(doc_id) if not store.keywords_for_chunk(str(c["id"]))]
    if not todo:
        return
    method_name = store.config_get(ENRICH_METHOD_KEY, DEFAULT_METHOD)
    assert method_name is not None
    method = get_method(method_name)
    result = method(kb, doc_id, todo, client=_client_factory(kb))
    if result.rejected:
        log.info(
            "enrich: %s dropped %d invalid keyword(s) from %d call(s)",
            doc_id,
            len(result.rejected),
            result.calls,
        )
