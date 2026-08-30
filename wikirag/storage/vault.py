"""Vault export: render the human-facing markdown view from the store.

The vault is a *product*, not a store: one file per keyword note, per
document summary, plus overview.md; wikilinks rendered as [[name]] for
Obsidian. Deleting vault/ loses nothing; export is idempotent and fast
enough to re-run after every ingest.

Notes carry their provenance (note_sources), so export can optionally
render footnote-style source references per note.

Two-way sync (user edits flowing back into the store) is explicitly out of
scope for now — docs/decisions.md.

Planned API
-----------
- export_vault(kb, out_dir=None, incremental=True)
"""
