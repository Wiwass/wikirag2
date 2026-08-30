"""Vector index: embedding storage and nearest-neighbor search.

One embedding space per KB (a single model; chunks and notes both live in
it, distinguished by owner_kind). Content-hash tracking so re-embedding
happens only when text or model changes (the one v1 mechanism kept as-is);
orphan cleanup is driven by the store's foreign keys, not sync passes.

Open decision (docs/decisions.md): npz matrices in RAM vs sqlite-vec.
This module hides that choice behind a small interface so the
micro-benchmark at implementation time can swap backends freely.

Planned API
-----------
- class VectorIndex: upsert(owner, text_hash, vector), delete(owner),
  search(query_vector, k, filter_kind=None) -> [(owner, score)],
  stale_owners(current_hashes) for incremental re-embeds.
"""
