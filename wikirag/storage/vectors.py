"""Vector index: embedding storage and nearest-neighbor search.

One embedding space per KB (a single model; chunks and notes both live in
it, distinguished by owner_kind). Content-hash tracking so re-embedding
happens only when text or model changes (the one v1 mechanism kept as-is);
orphan cleanup is driven by the store's foreign keys, not sync passes.

Backend: a single ``index.npz`` file per KB holding the L2-normalized
matrix plus its own metadata (owner ids, owner kinds, content hashes,
model name, dim). Search is numpy brute force (dot product of normalized
vectors == cosine). The index is self-contained: it never reads the SQLite
store, so it can be rebuilt, copied or deleted on its own.

Benchmarked at 10k/50k/200k vectors x dim 1024 (docs/decisions.md, D13):
brute force stays in the low tens of milliseconds at personal-KB scale, so
``sqlite-vec`` is deferred. Everything below the public API of
``VectorIndex`` is private, so an ANN or sqlite-vec backend can replace the
matrix and the file format without touching callers.

Persistence
-----------
Mutations (``upsert``/``delete``) are in-memory; ``save()`` writes the whole
index to a temporary file in the same directory, fsyncs it and ``os.replace``s
it over ``index.npz``. A crash therefore either leaves the previous complete
index or the new complete one, never a half-written file — at the price of
losing unsaved mutations, which are recoverable by re-embedding (the store,
not this index, is the source of truth). ``VectorIndex`` is also a context
manager: it saves on clean exit when there are unsaved changes.

API
---
- ``Owner(kind, id)``: NamedTuple, so plain ``(kind, id)`` tuples work too.
  ``kind`` is ``"chunk"`` or ``"note"``; ``id`` is a string (stringify
  integer note ids at the call site).
- ``VectorIndex.open(directory, model, dim) -> VectorIndex``
- ``upsert(owner, text_hash, vector)``, ``delete(owner) -> bool``
- ``search(query_vector, k=10, filter_kind=None) -> list[(Owner, score)]``
- ``stale_owners(current_hashes) -> list[Owner]`` for incremental re-embeds
- ``hash_of(owner)``, ``vector_of(owner)``, ``owners(kind=None)``,
  ``save()``, ``len(index)``, ``owner in index``
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import TracebackType
from typing import Iterator, Mapping, NamedTuple, Sequence

import numpy as np

INDEX_FILENAME = "index.npz"
FORMAT_VERSION = 1

#: Owner kinds mirror ``embeddings.owner_kind`` in docs/storage.md. They are
#: validated on write and on filter so that a typo fails loudly instead of
#: silently returning nothing.
OWNER_KINDS: tuple[str, ...] = ("chunk", "note")
_KIND_CODE = {kind: code for code, kind in enumerate(OWNER_KINDS)}

_INITIAL_CAPACITY = 64
_DTYPE = np.float32


class Owner(NamedTuple):
    """Identity of an embedded item: ``("chunk"|"note", id)``."""

    kind: str
    id: str


OwnerLike = Owner | tuple[str, str]


class VectorIndexError(RuntimeError):
    """Raised when an index file cannot be read or is not a valid index."""


class VectorIndex:
    """Embeddings of one KB, with content hashes, in one directory.

    Construct with :meth:`open`. Vectors are stored L2-normalized, so
    ``search`` scores are cosine similarities in ``[-1, 1]``.
    """

    def __init__(self, directory: Path | str, model: str, dim: int) -> None:
        """Create an empty in-memory index. Use :meth:`open` to load from disk."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._directory = Path(directory)
        self._model = model
        self._dim = int(dim)
        self._matrix: np.ndarray = np.zeros((_INITIAL_CAPACITY, self._dim), dtype=_DTYPE)
        self._kind_codes: np.ndarray = np.zeros(_INITIAL_CAPACITY, dtype=np.int8)
        self._owners: list[Owner] = []
        self._hashes: list[str] = []
        self._row: dict[Owner, int] = {}
        self._count = 0
        self._dirty = False

    # ---------------------------------------------------------------- open/save

    @classmethod
    def open(cls, directory: Path | str, model: str, dim: int) -> "VectorIndex":
        """Load the index in ``directory``, or start an empty one.

        A stored index built with a different model or dimension is discarded:
        vectors from another embedding space are not comparable, and they are
        derived data that :meth:`stale_owners` will then report as needing a
        re-embed (it reports everything, since the index is empty). The old
        file stays on disk until the next :meth:`save`.
        """
        index = cls(directory, model, dim)
        path = index.path
        if not path.exists():
            return index
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"].item()))
                if meta.get("model") != model or meta.get("dim") != index._dim:
                    return index
                vectors = np.ascontiguousarray(data["vectors"], dtype=_DTYPE)
                kinds = [str(k) for k in data["kinds"].tolist()]
                ids = [str(i) for i in data["ids"].tolist()]
                hashes = [str(h) for h in data["hashes"].tolist()]
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise VectorIndexError(f"cannot read vector index at {path}: {exc}") from exc

        count = len(ids)
        if vectors.shape != (count, index._dim) or len(kinds) != count or len(hashes) != count:
            raise VectorIndexError(f"inconsistent vector index at {path}")

        index._matrix = vectors
        index._kind_codes = np.array([_KIND_CODE.get(k, -1) for k in kinds], dtype=np.int8)
        index._owners = [Owner(k, i) for k, i in zip(kinds, ids)]
        index._hashes = hashes
        index._row = {owner: row for row, owner in enumerate(index._owners)}
        index._count = count
        if len(index._row) != count:
            raise VectorIndexError(f"duplicate owners in vector index at {path}")
        return index

    def save(self) -> None:
        """Write the index atomically (temp file + fsync + replace)."""
        self._directory.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "model": self._model,
                "dim": self._dim,
                "count": self._count,
            }
        )
        tmp = self._directory / f"{INDEX_FILENAME}.tmp-{os.getpid()}"
        # np.savez is handed an open file object: passing a path would append a
        # second ".npz" to the temporary name.
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                meta=np.array(meta),
                vectors=np.ascontiguousarray(self._matrix[: self._count]),
                kinds=np.array([o.kind for o in self._owners], dtype=np.str_),
                ids=np.array([o.id for o in self._owners], dtype=np.str_),
                hashes=np.array(self._hashes, dtype=np.str_),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        directory_fd = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._dirty = False

    def __enter__(self) -> "VectorIndex":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None and self._dirty:
            self.save()

    # ------------------------------------------------------------------ mutate

    def upsert(self, owner: OwnerLike, text_hash: str, vector: Sequence[float] | np.ndarray) -> None:
        """Insert or replace the vector and content hash of ``owner``."""
        key = self._as_owner(owner)
        normalized = self._normalize(vector)
        row = self._row.get(key)
        if row is None:
            row = self._count
            self._ensure_capacity(row + 1)
            self._row[key] = row
            self._owners.append(key)
            self._hashes.append(text_hash)
            self._kind_codes[row] = _KIND_CODE[key.kind]
            self._count += 1
        else:
            self._hashes[row] = text_hash
        self._matrix[row] = normalized
        self._dirty = True

    def delete(self, owner: OwnerLike) -> bool:
        """Remove ``owner``; return whether it was present."""
        key = self._as_owner(owner)
        row = self._row.pop(key, None)
        if row is None:
            return False
        last = self._count - 1
        if row != last:
            # Swap the last row into the hole: O(1) delete, row order is not
            # part of the contract (search sorts by score, then by owner).
            self._matrix[row] = self._matrix[last]
            self._kind_codes[row] = self._kind_codes[last]
            self._owners[row] = self._owners[last]
            self._hashes[row] = self._hashes[last]
            self._row[self._owners[row]] = row
        self._owners.pop()
        self._hashes.pop()
        self._count = last
        self._dirty = True
        return True

    # -------------------------------------------------------------------- read

    def search(
        self,
        query_vector: Sequence[float] | np.ndarray,
        k: int = 10,
        filter_kind: str | None = None,
    ) -> list[tuple[Owner, float]]:
        """Return the ``k`` nearest owners as ``(owner, cosine)``, best first.

        Ties are broken by owner (kind, id) so results are deterministic.
        """
        if filter_kind is not None and filter_kind not in _KIND_CODE:
            raise ValueError(f"unknown owner kind {filter_kind!r}, expected one of {OWNER_KINDS}")
        # The query is validated even when the answer is trivially empty: a
        # malformed query vector is a caller bug, not an empty result.
        query = self._normalize(query_vector)
        if k <= 0 or self._count == 0:
            return []
        scores = self._matrix[: self._count] @ query
        if filter_kind is not None:
            mask = self._kind_codes[: self._count] == _KIND_CODE[filter_kind]
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)
        wanted = min(k, self._count)
        if wanted < self._count:
            candidates = np.argpartition(-scores, wanted - 1)[:wanted]
        else:
            candidates = np.arange(self._count)
        rows = [int(row) for row in candidates if np.isfinite(scores[row])]
        rows.sort(key=lambda row: (-float(scores[row]), self._owners[row]))
        return [(self._owners[row], float(scores[row])) for row in rows]

    def stale_owners(self, current_hashes: Mapping[OwnerLike, str]) -> list[Owner]:
        """Owners in ``current_hashes`` that need (re-)embedding.

        Stale means: absent from the index, or stored under a different content
        hash. A model or dimension change empties the index at :meth:`open`, so
        it makes every owner stale. Owners present here but not in
        ``current_hashes`` are not reported: deletions are driven by the store.
        """
        stale: list[Owner] = []
        for owner, text_hash in current_hashes.items():
            key = self._as_owner(owner)
            row = self._row.get(key)
            if row is None or self._hashes[row] != text_hash:
                stale.append(key)
        stale.sort()
        return stale

    def hash_of(self, owner: OwnerLike) -> str | None:
        """Content hash stored for ``owner``, or None if absent."""
        row = self._row.get(self._as_owner(owner))
        return None if row is None else self._hashes[row]

    def vector_of(self, owner: OwnerLike) -> np.ndarray | None:
        """Normalized vector of ``owner`` (a copy), or None if absent.

        Handy for embedding-neighborhood association (D10): feed it back to
        :meth:`search` to find what sits next to an item.
        """
        row = self._row.get(self._as_owner(owner))
        return None if row is None else self._matrix[row].copy()

    def owners(self, kind: str | None = None) -> list[Owner]:
        """All owners in the index, sorted; optionally restricted to ``kind``."""
        if kind is not None and kind not in _KIND_CODE:
            raise ValueError(f"unknown owner kind {kind!r}, expected one of {OWNER_KINDS}")
        return sorted(o for o in self._owners if kind is None or o.kind == kind)

    # ------------------------------------------------------------- properties

    @property
    def path(self) -> Path:
        """Path of the index file."""
        return self._directory / INDEX_FILENAME

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dirty(self) -> bool:
        """Whether there are mutations not yet written by :meth:`save`."""
        return self._dirty

    def __len__(self) -> int:
        return self._count

    def __contains__(self, owner: object) -> bool:
        if not isinstance(owner, tuple) or len(owner) != 2:
            return False
        return Owner(*owner) in self._row

    def __iter__(self) -> Iterator[Owner]:
        return iter(list(self._owners))

    def __repr__(self) -> str:
        return (
            f"VectorIndex(directory={str(self._directory)!r}, model={self._model!r}, "
            f"dim={self._dim}, count={self._count})"
        )

    # ---------------------------------------------------------------- internal

    @staticmethod
    def _as_owner(owner: OwnerLike) -> Owner:
        kind, owner_id = owner
        if kind not in _KIND_CODE:
            raise ValueError(f"unknown owner kind {kind!r}, expected one of {OWNER_KINDS}")
        return Owner(kind, str(owner_id))

    def _normalize(self, vector: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=_DTYPE)
        if array.shape != (self._dim,):
            raise ValueError(f"expected a vector of shape ({self._dim},), got {array.shape}")
        norm = float(np.linalg.norm(array))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("vector must be finite and non-zero")
        return array / _DTYPE(norm)

    def _ensure_capacity(self, needed: int) -> None:
        capacity = self._matrix.shape[0]
        if needed <= capacity:
            return
        new_capacity = max(needed, capacity * 2)
        matrix = np.zeros((new_capacity, self._dim), dtype=_DTYPE)
        matrix[: self._count] = self._matrix[: self._count]
        self._matrix = matrix
        codes = np.zeros(new_capacity, dtype=np.int8)
        codes[: self._count] = self._kind_codes[: self._count]
        self._kind_codes = codes
