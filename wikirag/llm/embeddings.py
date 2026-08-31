"""Local models: embedding encoder and cross-encoder reranker.

One lazily-loaded shared instance of each per process (v1's
SharedEmbeddingModel pattern, kept), preloadable at chat startup so the
first query doesn't pay the load.

Constraints (v1 lesson): chunking targets ~400-600 tokens precisely so
that content fits the encoder's window with margin — the encoder's max
sequence length is a hard input to the chunker, exposed here as a
property, never assumed.

Models (D14, docs/decisions.md): ``BAAI/bge-m3`` for embedding and
``BAAI/bge-reranker-v2-m3`` for reranking. Cross-lingual by design
(Italian queries over English corpora), 8192-token window, no prefix or
instruction convention required. Per-KB config may override both names;
callers pass them in, the defaults here are only defaults.

Scales
------
- Embeddings: L2-normalized float32, so a dot product is a cosine in
  ``[-1, 1]`` — what ``storage.vectors.VectorIndex`` scores with.
- Reranker: a sigmoid of the cross-encoder logit, in ``(0, 1)``, monotone
  in relevance and stable across pools (unlike a rank). This is the scale
  the no-answer threshold is set on (docs/retrieval.md): relevant pairs
  land above ~0.5, clearly irrelevant ones below ~0.1. The threshold
  itself is calibrated at Checkpoint A, not guessed here.

API
---
- ``class Embedder``: ``encode_passages(texts) -> (n, dim) float32``,
  ``encode_query(text) -> (dim,) float32``, ``model_name``, ``dim``,
  ``max_seq_tokens``, ``preload()``. Prefix/prompt conventions and
  normalization are handled inside.
- ``class Reranker``: ``score(query, texts) -> list[float]``,
  ``model_name``, ``max_seq_tokens``, ``preload()``.
- ``get_embedder(model_name=None)`` / ``get_reranker(model_name=None)``:
  shared per-process instances, themselves lazy (no model is loaded until
  something is encoded or a model-derived property is read).
- ``preload(...)``: warm both, for chat startup. ``reset_shared()`` drops
  the cached instances (tests, model switches).
- ``class FakeEmbedder`` / ``class FakeReranker``: same interfaces,
  deterministic, no download — the offline stand-ins for retrieval tests.
- ``EmbedderLike`` / ``RerankerLike``: the protocols both pairs satisfy.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # heavy imports stay out of module import time
    from sentence_transformers import CrossEncoder, SentenceTransformer

#: Defaults; a KB overrides them through kb_config (docs/storage.md).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

#: encode() has a fixed per-call cost (tokenization, forward, bookkeeping) that
#: dominates on tens of thousands of items, so callers hand whole batches over
#: (v1 lesson: one-at-a-time embedding was the historic ingestion bottleneck).
DEFAULT_BATCH_SIZE = 32

# Prompt conventions. Modern sentence-transformers models that need a query or
# passage prefix ship it in their config as a named prompt, so the convention is
# read off the loaded model rather than hardcoded. E5 is the notable family
# whose published prefixes are documented but not shipped that way.
_QUERY_PROMPT_NAMES = ("query", "search_query", "s2p_query")
_PASSAGE_PROMPT_NAMES = ("document", "passage", "search_document", "corpus")
_FALLBACK_PREFIXES: dict[str, tuple[str, str]] = {"e5": ("query: ", "passage: ")}

_DTYPE = np.float32


@runtime_checkable
class EmbedderLike(Protocol):
    """What retrieval and ingestion need from an encoder."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def max_seq_tokens(self) -> int: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


@runtime_checkable
class RerankerLike(Protocol):
    """What retrieval needs from a cross-encoder."""

    @property
    def model_name(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


# --------------------------------------------------------------------- helpers


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function."""
    decay = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + decay), decay / (1.0 + decay))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero row would only come from a degenerate model output; leaving it as
    # zeros would make VectorIndex.upsert reject it, so keep it finite instead.
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(_DTYPE)


def _resolve_prompts(model: "SentenceTransformer", model_name: str) -> tuple[str, str]:
    """(query prefix, passage prefix) required by ``model``, possibly empty."""
    prompts: dict[str, str] = dict(getattr(model, "prompts", None) or {})
    query = next((prompts[name] for name in _QUERY_PROMPT_NAMES if prompts.get(name)), "")
    passage = next((prompts[name] for name in _PASSAGE_PROMPT_NAMES if prompts.get(name)), "")
    if not query and not passage:
        lowered = model_name.lower()
        for family, pair in _FALLBACK_PREFIXES.items():
            if family in lowered:
                return pair
    return query, passage


# -------------------------------------------------------------------- embedder


class Embedder:
    """Sentence-transformers encoder, loaded on first use.

    Reading :attr:`dim` or :attr:`max_seq_tokens` loads the model: both are
    read off it, never guessed — the chunker sizes chunks against
    :attr:`max_seq_tokens`, so a wrong value is a silently truncated corpus.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: "SentenceTransformer | None" = None
        self._query_prefix = ""
        self._passage_prefix = ""
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        """Whether the underlying model has been loaded yet."""
        return self._model is not None

    def preload(self) -> None:
        """Load the model now instead of at the first encode."""
        self._load()

    def _load(self) -> "SentenceTransformer":
        # Locked: ingestion embeds from a worker thread while the main thread
        # queries, and loading this model twice costs seconds and gigabytes.
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self._model_name, device=self._device)
                self._query_prefix, self._passage_prefix = _resolve_prompts(
                    model, self._model_name
                )
                self._model = model
            return self._model

    @property
    def dim(self) -> int:
        """Embedding dimension, from the loaded model."""
        dim = self._load().get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"model {self._model_name!r} reports no embedding dimension")
        return int(dim)

    @property
    def max_seq_tokens(self) -> int:
        """Input window in tokens — the hard bound the chunker targets."""
        window = self._load().max_seq_length
        if not window:
            raise RuntimeError(f"model {self._model_name!r} reports no max sequence length")
        return int(window)

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        # The prefixes come off the model, so they can only be read after the
        # load — never from the caller's side of this call.
        model = self._load()
        prefix = self._query_prefix if is_query else self._passage_prefix
        vectors = model.encode(
            list(texts),
            prompt=prefix or None,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float64).reshape(len(texts), -1)
        # The model is asked to normalize and then the rows are normalized
        # again: normalized float32 is this class's contract, and re-dividing
        # already-unit rows costs nothing next to a transformer forward pass.
        return _normalize_rows(matrix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Encode corpus texts: ``(len(texts), dim)``, L2-normalized float32."""
        if not texts:
            return np.zeros((0, self.dim), dtype=_DTYPE)
        return self._encode(texts, is_query=False)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one query: ``(dim,)``, L2-normalized float32."""
        return self._encode([text], is_query=True)[0]

    def __repr__(self) -> str:
        return f"Embedder(model_name={self._model_name!r}, loaded={self.loaded})"


class Reranker:
    """Cross-encoder relevance scorer, loaded on first use.

    :meth:`score` returns absolute scores in ``(0, 1)`` (sigmoid of the raw
    logit), comparable across queries and pools, which is what the no-answer
    gate needs. The activation is applied here rather than left to the
    library default so the scale is a property of this class.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: "CrossEncoder | None" = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def preload(self) -> None:
        self._load()

    def _load(self) -> "CrossEncoder":
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name, device=self._device)
            return self._model

    @property
    def max_seq_tokens(self) -> int:
        """Window for the concatenated (query, passage) pair."""
        model = self._load()
        # CrossEncoder leaves max_length unset when the checkpoint does not
        # pin one; the tokenizer then carries the real bound.
        window = model.max_length or getattr(model.tokenizer, "model_max_length", None)
        if not window:
            raise RuntimeError(f"model {self._model_name!r} reports no max sequence length")
        return int(window)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to ``query``, in ``(0, 1)``, order preserved."""
        if not texts:
            return []
        model = self._load()
        logits = model.predict(
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            # Identity: the raw logit is turned into a probability below, so the
            # scale does not depend on the library's default activation.
            activation_fn=lambda tensor: tensor,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in _sigmoid(np.asarray(logits, dtype=np.float64).ravel())]

    def __repr__(self) -> str:
        return f"Reranker(model_name={self._model_name!r}, loaded={self.loaded})"


# ------------------------------------------------------------------ shared use

_embedders: dict[tuple[str, str | None], Embedder] = {}
_rerankers: dict[tuple[str, str | None], Reranker] = {}
_registry_lock = threading.Lock()


def get_embedder(model_name: str | None = None, *, device: str | None = None) -> Embedder:
    """Shared :class:`Embedder` for ``model_name`` (default from constants).

    One instance per (model, device) in the process, so several KBs on the
    same model share one loaded copy. The instance is lazy: this call loads
    nothing.
    """
    key = (model_name or DEFAULT_EMBEDDING_MODEL, device)
    with _registry_lock:
        embedder = _embedders.get(key)
        if embedder is None:
            embedder = Embedder(key[0], device=device)
            _embedders[key] = embedder
        return embedder


def get_reranker(model_name: str | None = None, *, device: str | None = None) -> Reranker:
    """Shared :class:`Reranker` for ``model_name`` (default from constants)."""
    key = (model_name or DEFAULT_RERANKER_MODEL, device)
    with _registry_lock:
        reranker = _rerankers.get(key)
        if reranker is None:
            reranker = Reranker(key[0], device=device)
            _rerankers[key] = reranker
        return reranker


def preload(
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    *,
    device: str | None = None,
) -> None:
    """Load both shared models now (chat startup), so no query pays the load."""
    get_embedder(embedding_model, device=device).preload()
    get_reranker(reranker_model, device=device).preload()


def reset_shared() -> None:
    """Drop the shared instances (their models are freed when unreferenced)."""
    with _registry_lock:
        _embedders.clear()
        _rerankers.clear()


# ----------------------------------------------------------------------- fakes

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

FAKE_DIM = 32
FAKE_MAX_SEQ_TOKENS = 512


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """Deterministic offline stand-in for :class:`Embedder`.

    A text is the L2-normalized mean of one pseudo-random unit vector per
    token, each derived from a BLAKE2b digest of the token — so vectors are
    stable across processes and runs (no PYTHONHASHSEED dependence), equal
    texts give equal vectors, and texts sharing tokens score higher against
    each other than disjoint ones. That last property is what makes it a
    usable fixture for retrieval tests, not just a shape placeholder.
    """

    def __init__(
        self,
        model_name: str = "fake-embedder",
        *,
        dim: int = FAKE_DIM,
        max_seq_tokens: int = FAKE_MAX_SEQ_TOKENS,
    ) -> None:
        self._model_name = model_name
        self._dim = int(dim)
        self._max_seq_tokens = int(max_seq_tokens)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_seq_tokens(self) -> int:
        return self._max_seq_tokens

    def preload(self) -> None:
        """No-op: there is nothing to load."""

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=self._dim).digest()
        vector = (np.frombuffer(digest, dtype=np.uint8).astype(np.float64) - 127.5) / 127.5
        return vector / np.linalg.norm(vector)

    def _vector(self, text: str) -> np.ndarray:
        tokens = _tokenize(text) or ["\x00empty"]
        vector = np.mean([self._token_vector(t) for t in tokens], axis=0)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # only reachable if token vectors cancel exactly
            return self._token_vector(text or "\x00empty").astype(_DTYPE)
        return (vector / norm).astype(_DTYPE)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=_DTYPE)
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def __repr__(self) -> str:
        return f"FakeEmbedder(model_name={self._model_name!r}, dim={self._dim})"


class FakeReranker:
    """Deterministic offline stand-in for :class:`Reranker`.

    Score is the fraction of query tokens present in the text, in ``[0, 1]``
    — same scale and same monotonicity as the real reranker, so threshold
    logic can be tested without a model.
    """

    def __init__(self, model_name: str = "fake-reranker") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_seq_tokens(self) -> int:
        return FAKE_MAX_SEQ_TOKENS

    def preload(self) -> None:
        """No-op: there is nothing to load."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return [0.0] * len(texts)
        return [
            len(query_tokens & set(_tokenize(text))) / len(query_tokens) for text in texts
        ]

    def __repr__(self) -> str:
        return f"FakeReranker(model_name={self._model_name!r})"
