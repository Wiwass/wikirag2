"""Configuration: KB registry and the two sliders.

Responsibilities
----------------
- Locate the knowledge-bases root and enumerate/create/open KBs by name.
- Define the `KnowledgeBase` handle: name, paths, open store, config. It is
  constructed in exactly one place (here) and passed to every operation —
  no function signatures accumulating loose paths and dicts (v1 lesson).
- Per-KB config lives in the store's `kb_config` table, not in module
  constants: synthesis level (S0-S3), embedding model name, role->model map.
- Global config: providers/credentials location, defaults for new KBs.

Sliders
-------
- SynthesisLevel: S0 index-only, S1 summaries, S2 wiki, S3 atlas.
- RetrievalDepth: R0 lookup, R1 ranked (default), R2 multi-hop, R3 agentic.
Both are small enums used across the package; retrieval code must branch on
what exists in the store, never on the configured S level (docs/architecture.md).

Planned API
-----------
- class SynthesisLevel(Enum); class RetrievalDepth(Enum)
- class KnowledgeBase: .name, .root, .store, .config, .close()
- list_kbs() / create_kb(name) / open_kb(name) -> KnowledgeBase
"""
