"""Local models: embedding encoder and cross-encoder reranker.

One lazily-loaded shared instance of each per process (v1's
SharedEmbeddingModel pattern, kept), preloadable at chat startup so the
first query doesn't pay the load.

Constraints (v1 lesson): chunking targets ~400-600 tokens precisely so
that content fits the encoder's window with margin — the encoder's max
sequence length is a hard input to the chunker, exposed here as a
property, never assumed.

Model choice is open (docs/decisions.md): long-context, ideally
multilingual (Italian queries over English corpora). Candidates: bge-m3,
gte-modernbert-base, EmbeddingGemma; reranker: bge-reranker class.

Planned API
-----------
- class Embedder: encode_passages(texts), encode_query(text),
  max_seq_tokens, model_name; normalize + prefix conventions inside.
- class Reranker: score(query, texts) -> [float]  (absolute scores:
  they also drive the no-answer threshold, docs/retrieval.md)
"""
