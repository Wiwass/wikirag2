"""S3: the atlas — overview map, hub pages, link curation.

Purely for the human product and for global/sensemaking queries; expected
to show little R0-R1 benchmark gain, and that is fine and stated up front
(docs/synthesis.md).

- Overview: map-reduce over S1 summaries (v1 Phase F, kept: grouped map
  calls cached as jobs, reduce combines with top-centrality keyword notes).
  Prepended as a primer in chat.
- Hub pages: curated entry points for high-centrality keywords
  (centrality = incoming note_links, now a SQL query).
- Link curation: merge near-duplicate keywords, prune weak links.

Planned API
-----------
- build_overview(kb, client) -> note_id
- curate_links(kb, client) -> CurationReport
"""
