"""S1: document/section summaries.

One summary note per document (per major section for large books),
synthesized from that document's chunks by reference. Summaries are
indexed content (FTS + embedding) serving mid-level queries, and they are
the raw material for cheap atlas building (v1's Phase F insight, kept).

Planned API
-----------
- summarize_document(kb, doc_id, client) -> note_id(s)
"""
