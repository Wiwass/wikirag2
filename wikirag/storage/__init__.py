"""Storage layer: the single source of truth for a knowledge base.

One SQLite database per KB (schema draft in docs/storage.md) plus a vector
index. The markdown vault is a regenerable export, never primary storage.
No pipeline logic lives here; other components read/write only through this
layer's APIs so that every stage output commits atomically.

Modules:
    store    SQLite schema, migrations, transactional accessors, job state
    vectors  embedding matrices + nearest-neighbor search
    vault    markdown export (Obsidian-compatible)
"""
