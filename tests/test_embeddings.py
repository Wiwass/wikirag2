"""Tests for wikirag.llm.embeddings — offline by default.

The default run never imports sentence-transformers and never touches the
network: the wrappers are exercised against a stub module injected into
``sys.modules``, and the fakes need nothing at all. The tests against the
real models (which may download gigabytes on first run) are opt-in::

    WIKIRAG_MODEL_TESTS=1 venv/bin/pytest tests/test_embeddings.py
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from wikirag.llm import embeddings as emb

MODEL_TESTS = pytest.mark.skipif(
    os.environ.get("WIKIRAG_MODEL_TESTS") != "1",
    reason="set WIKIRAG_MODEL_TESTS=1 to run tests that load (and may download) real models",
)

STUB_DIM = 4


class StubSentenceTransformer:
    """Minimal stand-in for the sentence-transformers class, no torch."""

    constructed: list[tuple[str, str | None]] = []

    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        type(self).constructed.append((model_name_or_path, device))
        self.model_name = model_name_or_path
        self.prompts: dict[str, str] = {}
        self.max_seq_length = 8192
        self.encoded: list[tuple[list[str], str | None]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return STUB_DIM

    def encode(self, texts, prompt=None, batch_size=32, normalize_embeddings=False,
               convert_to_numpy=True, show_progress_bar=False):
        self.encoded.append((list(texts), prompt))
        # Deliberately NOT unit vectors: the wrapper owes normalized output
        # whatever the model returns.
        return np.array([[len(t) + 1.0, 2.0, 0.0, 0.0] for t in texts], dtype=np.float64)


class StubCrossEncoder:
    constructed: list[tuple[str, str | None]] = []

    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        type(self).constructed.append((model_name_or_path, device))
        self.max_length = 8192
        self.predicted: list[list[tuple[str, str]]] = []

    def predict(self, inputs, batch_size=32, activation_fn=None,
                convert_to_numpy=True, show_progress_bar=False):
        self.predicted.append(list(inputs))
        # Raw logits: positive when the passage contains the query.
        logits = np.array(
            [8.0 if query in text else -8.0 for query, text in inputs], dtype=np.float64
        )
        return activation_fn(logits) if activation_fn is not None else logits


@pytest.fixture
def stub_st(monkeypatch):
    """Install a fake sentence_transformers module and reset shared state."""
    StubSentenceTransformer.constructed = []
    StubCrossEncoder.constructed = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubSentenceTransformer
    module.CrossEncoder = StubCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    emb.reset_shared()
    yield module
    emb.reset_shared()


# ------------------------------------------------------------------- the fakes


def test_fake_embedder_shape_dtype_and_normalization():
    fake = emb.FakeEmbedder()
    matrix = fake.encode_passages(["alpha beta", "gamma", ""])
    assert matrix.shape == (3, fake.dim) == (3, 32)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)

    vector = fake.encode_query("alpha beta")
    assert vector.shape == (fake.dim,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


def test_fake_embedder_reports_a_window():
    assert emb.FakeEmbedder().max_seq_tokens == 512
    assert emb.FakeEmbedder(dim=8, max_seq_tokens=64).dim == 8


def test_fake_embedder_is_deterministic_and_collides_on_equal_text():
    a, b = emb.FakeEmbedder(), emb.FakeEmbedder()
    first = a.encode_passages(["the quick brown fox"])[0]
    second = b.encode_passages(["the quick brown fox"])[0]
    assert np.array_equal(first, second)
    # Query and passage encoders share the space: identical text, identical vector.
    assert np.array_equal(first, a.encode_query("the quick brown fox"))
    # Stable across runs/processes (no PYTHONHASHSEED dependence): a value
    # recomputed here must equal the one recomputed from scratch by any run.
    assert np.array_equal(first, emb.FakeEmbedder().encode_query("The Quick Brown Fox"))


def test_fake_embedder_empty_input():
    matrix = emb.FakeEmbedder().encode_passages([])
    assert matrix.shape == (0, 32)
    assert matrix.dtype == np.float32
    # The empty string still gets a usable (finite, unit) vector.
    assert np.isclose(np.linalg.norm(emb.FakeEmbedder().encode_query("")), 1.0, atol=1e-6)


def test_fake_embedder_similar_beats_disjoint():
    fake = emb.FakeEmbedder()
    query = fake.encode_query("dragon magic ritual")
    similar, disjoint = fake.encode_passages(
        ["dragon magic ritual circle", "quarterly logistics invoice spreadsheet"]
    )
    assert float(query @ similar) > float(query @ disjoint)
    # Partial overlap sits between exact match and disjoint.
    partial = fake.encode_query("dragon magic")
    assert float(query @ query) > float(query @ partial) > float(query @ disjoint)


def test_fake_embedder_vectors_survive_the_vector_index(tmp_path):
    from wikirag.storage.vectors import VectorIndex

    fake = emb.FakeEmbedder()
    index = VectorIndex.open(tmp_path / "vectors", fake.model_name, fake.dim)
    texts = {"c1": "dragon magic ritual", "c2": "quarterly logistics invoice"}
    for owner_id, (text, vector) in zip(
        texts, zip(texts.values(), fake.encode_passages(list(texts.values())))
    ):
        index.upsert(("chunk", owner_id), "h", vector)
    hits = index.search(fake.encode_query("magic dragon"), k=2)
    assert hits[0][0].id == "c1"
    assert hits[0][1] > hits[1][1]


def test_fake_reranker_scores():
    reranker = emb.FakeReranker()
    scores = reranker.score("dragon magic", ["dragon magic ritual", "invoice", "dragon"])
    assert scores == [1.0, 0.0, 0.5]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert reranker.score("dragon", []) == []
    assert reranker.score("", ["dragon"]) == [0.0]
    # Deterministic and order-preserving.
    assert reranker.score("dragon magic", ["invoice", "dragon magic ritual"]) == [0.0, 1.0]


def test_fakes_satisfy_the_protocols():
    assert isinstance(emb.FakeEmbedder(), emb.EmbedderLike)
    assert isinstance(emb.FakeReranker(), emb.RerankerLike)


def test_real_classes_expose_the_same_surface():
    # Checked on the classes, not instances: touching Embedder().dim on an
    # instance would load the real model.
    for name in ("model_name", "dim", "max_seq_tokens", "encode_passages", "encode_query"):
        assert hasattr(emb.Embedder, name) and hasattr(emb.FakeEmbedder, name)
    for name in ("model_name", "max_seq_tokens", "score"):
        assert hasattr(emb.Reranker, name) and hasattr(emb.FakeReranker, name)


# ------------------------------------------------------------------- laziness


def test_get_embedder_loads_nothing(stub_st):
    embedder = emb.get_embedder()
    assert embedder.model_name == emb.DEFAULT_EMBEDDING_MODEL
    assert not embedder.loaded
    assert StubSentenceTransformer.constructed == []


def test_get_reranker_loads_nothing(stub_st):
    reranker = emb.get_reranker()
    assert reranker.model_name == emb.DEFAULT_RERANKER_MODEL
    assert not reranker.loaded
    assert StubCrossEncoder.constructed == []


def test_constructing_an_embedder_loads_nothing(stub_st):
    emb.Embedder("some/model")
    emb.Reranker("some/model")
    assert StubSentenceTransformer.constructed == []
    assert StubCrossEncoder.constructed == []


def test_shared_instances_are_reused_per_model(stub_st):
    assert emb.get_embedder() is emb.get_embedder()
    assert emb.get_embedder() is emb.get_embedder(emb.DEFAULT_EMBEDDING_MODEL)
    assert emb.get_embedder("other/model") is not emb.get_embedder()
    assert emb.get_reranker() is emb.get_reranker()
    assert emb.get_reranker("other/model") is not emb.get_reranker()


def test_model_is_loaded_once_and_shared(stub_st):
    embedder = emb.get_embedder()
    embedder.encode_query("hello")
    embedder.encode_passages(["a", "b"])
    assert emb.get_embedder().loaded
    assert StubSentenceTransformer.constructed == [(emb.DEFAULT_EMBEDDING_MODEL, None)]


def test_preload_warms_both(stub_st):
    emb.preload()
    assert emb.get_embedder().loaded and emb.get_reranker().loaded
    assert StubSentenceTransformer.constructed == [(emb.DEFAULT_EMBEDDING_MODEL, None)]
    assert StubCrossEncoder.constructed == [(emb.DEFAULT_RERANKER_MODEL, None)]


def test_reset_shared_drops_instances(stub_st):
    first = emb.get_embedder()
    emb.reset_shared()
    assert emb.get_embedder() is not first


# ------------------------------------------------------- wrappers, stub-backed


def test_embedder_output_is_normalized_float32(stub_st):
    embedder = emb.Embedder("stub/model")
    matrix = embedder.encode_passages(["a", "bb", "ccc"])
    assert matrix.shape == (3, STUB_DIM)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    assert embedder.encode_query("a").shape == (STUB_DIM,)


def test_embedder_reads_dim_and_window_from_the_model(stub_st):
    embedder = emb.Embedder("stub/model")
    assert embedder.dim == STUB_DIM
    assert embedder.max_seq_tokens == 8192
    assert embedder.loaded


def test_embedder_empty_batch(stub_st):
    matrix = emb.Embedder("stub/model").encode_passages([])
    assert matrix.shape == (0, STUB_DIM)
    assert matrix.dtype == np.float32


def test_embedder_passes_no_prompt_when_the_model_declares_none(stub_st):
    embedder = emb.Embedder("stub/model")
    embedder.encode_query("q")
    embedder.encode_passages(["p"])
    assert embedder._model.encoded == [(["q"], None), (["p"], None)]


def test_embedder_uses_the_prompts_the_model_ships(stub_st, monkeypatch):
    class PromptedStub(StubSentenceTransformer):
        def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
            super().__init__(model_name_or_path, device)
            self.prompts = {"query": "task: search | query: ", "document": "title: none | "}

    monkeypatch.setattr(stub_st, "SentenceTransformer", PromptedStub)
    embedder = emb.Embedder("stub/prompted")
    embedder.encode_query("q")
    embedder.encode_passages(["p"])
    assert embedder._model.encoded == [
        (["q"], "task: search | query: "),
        (["p"], "title: none | "),
    ]


def test_embedder_falls_back_to_the_e5_prefix_convention(stub_st):
    embedder = emb.Embedder("intfloat/multilingual-e5-large")
    embedder.encode_query("q")
    embedder.encode_passages(["p"])
    assert embedder._model.encoded == [(["q"], "query: "), (["p"], "passage: ")]


def test_reranker_maps_logits_through_a_sigmoid(stub_st):
    reranker = emb.Reranker("stub/model")
    scores = reranker.score("dragon", ["a dragon here", "nothing relevant"])
    assert len(scores) == 2
    assert all(0.0 < s < 1.0 for s in scores)
    assert scores[0] > 0.9 > 0.1 > scores[1]
    assert scores[0] == pytest.approx(1 / (1 + np.exp(-8.0)))
    assert reranker.score("dragon", []) == []
    assert reranker.max_seq_tokens == 8192


def test_sigmoid_is_monotone_and_overflow_free():
    values = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    scores = emb._sigmoid(values)
    assert np.all(np.isfinite(scores))
    assert np.all(np.diff(scores) > 0)
    assert scores[2] == pytest.approx(0.5)


# ------------------------------------------------- real models (opt-in, slow)


@pytest.fixture(scope="module")
def real_embedder():
    embedder = emb.Embedder(emb.DEFAULT_EMBEDDING_MODEL)
    embedder.preload()
    return embedder


@pytest.fixture(scope="module")
def real_reranker():
    reranker = emb.Reranker(emb.DEFAULT_RERANKER_MODEL)
    reranker.preload()
    return reranker


@MODEL_TESTS
def test_real_embedder_dim_and_window(real_embedder):
    assert real_embedder.dim > 0
    # The chunker sizes 400-600-token chunks against this; anything under ~1k
    # would invalidate the chunking plan (D14, docs/decisions.md).
    assert real_embedder.max_seq_tokens >= 1024


@MODEL_TESTS
def test_real_embedder_output_contract(real_embedder):
    matrix = real_embedder.encode_passages(["hello world", "goodbye world"])
    assert matrix.shape == (2, real_embedder.dim)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)
    vector = real_embedder.encode_query("hello world")
    assert vector.shape == (real_embedder.dim,)
    assert vector.dtype == np.float32


@MODEL_TESTS
def test_real_embedder_ranks_relevant_passage_first(real_embedder):
    passages = [
        "The dragon's breath weapon deals fire damage in a cone.",
        "Quarterly revenue rose after the logistics contract was renewed.",
    ]
    matrix = real_embedder.encode_passages(passages)
    query = real_embedder.encode_query("How does a dragon's fire breath work?")
    assert float(query @ matrix[0]) > float(query @ matrix[1])


@MODEL_TESTS
def test_real_embedder_is_cross_lingual(real_embedder):
    """The reason for the model choice: Italian query, English corpus."""
    passages = [
        "The dragon's breath weapon deals fire damage in a cone.",
        "Quarterly revenue rose after the logistics contract was renewed.",
    ]
    matrix = real_embedder.encode_passages(passages)
    query = real_embedder.encode_query("Come funziona il soffio di fuoco del drago?")
    assert float(query @ matrix[0]) > float(query @ matrix[1])


@MODEL_TESTS
def test_real_reranker_orders_and_stays_in_range(real_reranker):
    scores = real_reranker.score(
        "How does a dragon's fire breath work?",
        [
            "The dragon's breath weapon deals fire damage in a cone.",
            "Quarterly revenue rose after the logistics contract was renewed.",
        ],
    )
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]


@MODEL_TESTS
def test_real_reranker_scores_are_absolute_not_pool_relative(real_reranker):
    """Same pair, different pool, same score — what the no-answer gate needs."""
    relevant = "The dragon's breath weapon deals fire damage in a cone."
    query = "How does a dragon's fire breath work?"
    alone = real_reranker.score(query, [relevant])[0]
    in_pool = real_reranker.score(query, ["unrelated filler text", relevant])[1]
    assert alone == pytest.approx(in_pool, abs=1e-4)


@MODEL_TESTS
def test_real_reranker_window(real_reranker):
    assert real_reranker.max_seq_tokens >= 1024
