"""Native cross-reference extraction. Pattern-based, no LLM (D10).

Prose has a native graph the way code has imports: explicit textual
cross-references ("see Chapter 5", "cfr. ArM5 p.83", "as described in
The Mysteries"). This stage extracts them from chunks into the xrefs
table with high precision and zero API cost — the only graph materialized
eagerly at ingest.

Resolution is best-effort: a raw reference is stored always; to_doc_id /
to_section_id are filled when the target matches a known document title,
section heading, or page mapping. Unresolved xrefs stay useful for the
agent (R3) as literal follow-up cues.

Pattern set starts small (see/cfr/chapter/page + known doc abbreviations,
per-KB extendable via kb_config) and grows from observed corpus forms.

Planned API
-----------
- extract_xrefs(kb, doc_id) -> [XRef]  (idempotent per document)
"""
