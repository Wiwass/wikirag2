"""Tests for wikirag.storage.vectors — offline, numpy only, fast.

The scale benchmark that answers the npz-vs-sqlite-vec open question in
docs/storage.md is skipped by default; run it with::

    WIKIRAG_BENCH=1 venv/bin/pytest tests/test_vectors.py -k benchmark -s
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from wikirag.storage.vectors import (
    INDEX_FILENAME,
    Owner,
    VectorIndex,
    VectorIndexError,
)

MODEL = "test-model"
DIM = 4

# Handcrafted, mutually distinguishable directions in a 4-d space.
E1 = [1.0, 0.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0, 0.0]
E3 = [0.0, 0.0, 1.0, 0.0]
E12 = [1.0, 1.0, 0.0, 0.0]


def make_index(tmp_path, model: str = MODEL, dim: int = DIM) -> VectorIndex:
    return VectorIndex.open(tmp_path / "vectors", model, dim)


def populated(tmp_path) -> VectorIndex:
    index = make_index(tmp_path)
    index.upsert(("chunk", "c1"), "h1", E1)
    index.upsert(("chunk", "c2"), "h2", E2)
    index.upsert(("note", "n1"), "h3", E3)
    return index


# ------------------------------------------------------------------ basics


def test_empty_index_searches_empty(tmp_path):
    index = make_index(tmp_path)
    assert len(index) == 0
    assert index.search(E1, k=5) == []
    assert index.owners() == []


def test_upsert_and_nearest_neighbor_order(tmp_path):
    index = populated(tmp_path)
    assert len(index) == 3
    results = index.search(E1, k=3)
    assert [owner for owner, _ in results] == [
        Owner("chunk", "c1"),
        Owner("chunk", "c2"),
        Owner("note", "n1"),
    ]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.0)


def test_scores_are_cosine_and_query_scale_invariant(tmp_path):
    index = make_index(tmp_path)
    index.upsert(("chunk", "diag"), "h", E12)
    (owner, score), = index.search(E1, k=1)
    assert owner == ("chunk", "diag")
    assert score == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    scaled = index.search([100.0, 0.0, 0.0, 0.0], k=1)
    assert scaled[0][1] == pytest.approx(score, abs=1e-6)


def test_upsert_replaces_vector_and_hash_without_growing(tmp_path):
    index = populated(tmp_path)
    index.upsert(("chunk", "c1"), "h1-new", E3)
    assert len(index) == 3
    assert index.hash_of(("chunk", "c1")) == "h1-new"
    assert [owner for owner, _ in index.search(E3, k=2)] == [
        Owner("chunk", "c1"),
        Owner("note", "n1"),
    ]
    # c1 no longer points along E1, so nothing scores above zero there.
    assert all(score == pytest.approx(0.0) for _, score in index.search(E1, k=3))


def test_k_larger_than_index_and_non_positive_k(tmp_path):
    index = populated(tmp_path)
    assert len(index.search(E1, k=99)) == 3
    assert index.search(E1, k=0) == []
    assert index.search(E1, k=-1) == []


def test_ties_broken_deterministically_by_owner(tmp_path):
    index = make_index(tmp_path)
    for owner_id in ["c3", "c1", "c2"]:
        index.upsert(("chunk", owner_id), "h", E1)
    assert [o.id for o, _ in index.search(E1, k=3)] == ["c1", "c2", "c3"]


def test_owner_tuple_and_namedtuple_are_interchangeable(tmp_path):
    index = populated(tmp_path)
    assert ("chunk", "c1") in index
    assert Owner("chunk", "c1") in index
    assert ("chunk", "nope") not in index
    assert index.vector_of(Owner("chunk", "c1")) is not None


def test_vector_of_is_normalized_copy(tmp_path):
    index = make_index(tmp_path)
    index.upsert(("chunk", "c1"), "h", [3.0, 4.0, 0.0, 0.0])
    vector = index.vector_of(("chunk", "c1"))
    assert vector is not None
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    vector[:] = 0.0
    assert np.linalg.norm(index.vector_of(("chunk", "c1"))) == pytest.approx(1.0, abs=1e-6)
    assert index.vector_of(("chunk", "missing")) is None


# ------------------------------------------------------------------ filtering


def test_filter_kind(tmp_path):
    index = populated(tmp_path)
    notes = index.search(E1, k=5, filter_kind="note")
    assert [owner for owner, _ in notes] == [Owner("note", "n1")]
    chunks = index.search(E3, k=5, filter_kind="chunk")
    assert {owner.kind for owner, _ in chunks} == {"chunk"}
    assert len(chunks) == 2


def test_filter_kind_with_no_members(tmp_path):
    index = make_index(tmp_path)
    index.upsert(("chunk", "c1"), "h", E1)
    assert index.search(E1, k=5, filter_kind="note") == []


def test_unknown_kinds_are_rejected(tmp_path):
    index = populated(tmp_path)
    with pytest.raises(ValueError):
        index.search(E1, k=1, filter_kind="keyword")
    with pytest.raises(ValueError):
        index.upsert(("keyword", "k1"), "h", E1)
    with pytest.raises(ValueError):
        index.owners(kind="keyword")


def test_owners_listing(tmp_path):
    index = populated(tmp_path)
    assert index.owners() == [
        Owner("chunk", "c1"),
        Owner("chunk", "c2"),
        Owner("note", "n1"),
    ]
    assert index.owners(kind="note") == [Owner("note", "n1")]


# -------------------------------------------------------------------- delete


def test_delete_removes_owner_and_reports_presence(tmp_path):
    index = populated(tmp_path)
    assert index.delete(("chunk", "c1")) is True
    assert index.delete(("chunk", "c1")) is False
    assert len(index) == 2
    assert ("chunk", "c1") not in index
    assert index.hash_of(("chunk", "c1")) is None
    assert [owner for owner, _ in index.search(E1, k=5)] == [
        Owner("chunk", "c2"),
        Owner("note", "n1"),
    ]


def test_delete_keeps_remaining_vectors_intact(tmp_path):
    index = make_index(tmp_path)
    index.upsert(("chunk", "c1"), "h1", E1)
    index.upsert(("chunk", "c2"), "h2", E2)
    index.upsert(("note", "n1"), "h3", E3)
    # Deleting a middle row exercises the swap-with-last path.
    index.delete(("chunk", "c2"))
    assert index.search(E3, k=1)[0][0] == Owner("note", "n1")
    assert index.search(E1, k=1)[0][0] == Owner("chunk", "c1")
    assert index.hash_of(("note", "n1")) == "h3"


def test_delete_then_reinsert(tmp_path):
    index = populated(tmp_path)
    index.delete(("chunk", "c2"))
    index.upsert(("chunk", "c2"), "h2b", E2)
    assert len(index) == 3
    assert index.hash_of(("chunk", "c2")) == "h2b"
    assert index.search(E2, k=1)[0][0] == Owner("chunk", "c2")


# --------------------------------------------------------------- persistence


def test_persistence_across_reopen(tmp_path):
    index = populated(tmp_path)
    index.delete(("chunk", "c2"))
    index.save()

    reopened = make_index(tmp_path)
    assert len(reopened) == 2
    assert reopened.owners() == [Owner("chunk", "c1"), Owner("note", "n1")]
    assert reopened.hash_of(("note", "n1")) == "h3"
    assert reopened.search(E1, k=1)[0][0] == Owner("chunk", "c1")
    assert reopened.search(E1, k=1)[0][1] == pytest.approx(1.0)
    # Reopening a saved index has nothing to write.
    assert reopened.dirty is False


def test_unsaved_mutations_are_not_persisted(tmp_path):
    index = populated(tmp_path)
    index.save()
    index.upsert(("chunk", "c9"), "h9", E2)
    assert index.dirty is True
    assert ("chunk", "c9") not in make_index(tmp_path)


def test_context_manager_saves_on_clean_exit(tmp_path):
    with make_index(tmp_path) as index:
        index.upsert(("chunk", "c1"), "h1", E1)
    assert ("chunk", "c1") in make_index(tmp_path)


def test_context_manager_does_not_save_on_error(tmp_path):
    with pytest.raises(RuntimeError):
        with make_index(tmp_path) as index:
            index.upsert(("chunk", "c1"), "h1", E1)
            raise RuntimeError("boom")
    assert len(make_index(tmp_path)) == 0


def test_save_creates_directory_and_leaves_no_temp_files(tmp_path):
    index = populated(tmp_path)
    index.save()
    entries = sorted(p.name for p in (tmp_path / "vectors").iterdir())
    assert entries == [INDEX_FILENAME]


def test_save_empty_index_roundtrips(tmp_path):
    index = make_index(tmp_path)
    index.save()
    assert len(make_index(tmp_path)) == 0


def test_save_overwrites_previous_version(tmp_path):
    index = populated(tmp_path)
    index.save()
    index.delete(("chunk", "c1"))
    index.upsert(("note", "n2"), "h4", E2)
    index.save()
    reopened = make_index(tmp_path)
    assert reopened.owners() == [Owner("chunk", "c2"), Owner("note", "n1"), Owner("note", "n2")]


def test_corrupt_index_file_raises(tmp_path):
    index = populated(tmp_path)
    index.save()
    index.path.write_bytes(b"not an npz file")
    with pytest.raises(VectorIndexError):
        make_index(tmp_path)


# ------------------------------------------------------- staleness / mismatch


def test_stale_owners_detects_missing_and_changed_hashes(tmp_path):
    index = populated(tmp_path)
    current = {
        ("chunk", "c1"): "h1",  # unchanged
        ("chunk", "c2"): "h2-changed",  # re-edited text
        ("note", "n1"): "h3",  # unchanged
        ("note", "n2"): "h4",  # never embedded
    }
    assert index.stale_owners(current) == [Owner("chunk", "c2"), Owner("note", "n2")]


def test_stale_owners_ignores_owners_absent_from_current(tmp_path):
    index = populated(tmp_path)
    assert index.stale_owners({("chunk", "c1"): "h1"}) == []


def test_stale_owners_after_model_change(tmp_path):
    index = populated(tmp_path)
    index.save()
    current = {("chunk", "c1"): "h1", ("chunk", "c2"): "h2", ("note", "n1"): "h3"}
    assert make_index(tmp_path).stale_owners(current) == []

    other_model = make_index(tmp_path, model="other-model")
    assert len(other_model) == 0
    assert other_model.stale_owners(current) == [
        Owner("chunk", "c1"),
        Owner("chunk", "c2"),
        Owner("note", "n1"),
    ]


def test_dimension_change_discards_the_old_space(tmp_path):
    index = populated(tmp_path)
    index.save()
    wider = make_index(tmp_path, dim=8)
    assert len(wider) == 0
    assert wider.dim == 8
    # The old file is only replaced on the next save.
    wider.upsert(("chunk", "c1"), "h1", [1.0] + [0.0] * 7)
    wider.save()
    assert make_index(tmp_path, dim=8).owners() == [Owner("chunk", "c1")]


def test_vector_dimension_mismatch_is_rejected(tmp_path):
    index = make_index(tmp_path)
    with pytest.raises(ValueError):
        index.upsert(("chunk", "c1"), "h", [1.0, 0.0])
    with pytest.raises(ValueError):
        index.upsert(("chunk", "c1"), "h", np.zeros((1, DIM)))
    with pytest.raises(ValueError):
        index.search([1.0, 0.0])


def test_zero_and_non_finite_vectors_are_rejected(tmp_path):
    index = make_index(tmp_path)
    with pytest.raises(ValueError):
        index.upsert(("chunk", "c1"), "h", [0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        index.upsert(("chunk", "c1"), "h", [float("nan"), 1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        index.search([0.0, 0.0, 0.0, 0.0])


def test_invalid_dim_rejected(tmp_path):
    with pytest.raises(ValueError):
        VectorIndex(tmp_path / "vectors", MODEL, 0)


# ------------------------------------------------------------------ at scale


def test_many_vectors_recover_a_planted_neighbor(tmp_path):
    rng = np.random.default_rng(0)
    index = make_index(tmp_path, dim=32)
    target = rng.normal(size=32)
    for i in range(500):
        index.upsert(("chunk", f"c{i}"), f"h{i}", rng.normal(size=32))
    index.upsert(("note", "planted"), "hp", target)
    query = target + rng.normal(size=32) * 0.01
    assert index.search(query, k=1)[0][0] == Owner("note", "planted")
    index.save()
    assert make_index(tmp_path, dim=32).search(query, k=1)[0][0] == Owner("note", "planted")


@pytest.mark.skipif(
    os.environ.get("WIKIRAG_BENCH") != "1",
    reason="scale benchmark; set WIKIRAG_BENCH=1 to run",
)
@pytest.mark.parametrize("n", [10_000, 50_000, 200_000])
def test_benchmark_brute_force_search(tmp_path, n: int) -> None:
    """Measure brute-force search latency at KB scale (dim 1024)."""
    dim = 1024
    rng = np.random.default_rng(1)
    index = make_index(tmp_path, dim=dim)
    index._ensure_capacity(n)  # pre-size so the benchmark times search, not growth
    build_start = time.perf_counter()
    block = 5_000
    for start in range(0, n, block):
        vectors = rng.normal(size=(min(block, n - start), dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for offset, vector in enumerate(vectors):
            index.upsert(("chunk", f"c{start + offset}"), "h", vector)
    build_s = time.perf_counter() - build_start

    query = rng.normal(size=dim).astype(np.float32)
    index.search(query, k=20)  # warm up BLAS threads
    timings = []
    for _ in range(10):
        started = time.perf_counter()
        index.search(query, k=20)
        timings.append((time.perf_counter() - started) * 1000)
    filtered = time.perf_counter()
    index.search(query, k=20, filter_kind="chunk")
    filtered_ms = (time.perf_counter() - filtered) * 1000

    save_start = time.perf_counter()
    index.save()
    save_s = time.perf_counter() - save_start
    size_mb = index.path.stat().st_size / 1e6
    load_start = time.perf_counter()
    make_index(tmp_path, dim=dim)
    load_s = time.perf_counter() - load_start

    print(
        f"\nn={n:>7} dim={dim} | search median {np.median(timings):6.1f} ms "
        f"(p90 {np.percentile(timings, 90):6.1f} ms, filtered {filtered_ms:6.1f} ms) "
        f"| build {build_s:5.1f} s | save {save_s:4.2f} s | load {load_s:4.2f} s "
        f"| file {size_mb:6.1f} MB"
    )
