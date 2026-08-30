"""R3: agentic navigation loop.

An agent (the answering model, or a cheaper driver model) iterates with
tools until it can answer or concludes the KB lacks the information.
Hard caps on tool calls and total tokens.

Tools (thin wrappers over retrieval primitives and store read APIs —
the agent never bypasses them):
- search(query, k)            -> fused+reranked hits (id, snippet, score)
- read_chunk(id) / read_note(keyword) / neighbors(keyword)
- read_section(doc, section)  -> local context around a hit

This module requires nothing beyond stable IDs and read APIs, which the
storage design guarantees — deliberately implementable last (roadmap
step 5) without touching anything below it.

Planned API
-----------
- answer_agentic(kb, query, client, caps: AgentCaps) -> AgentAnswer
  (answer text + tool-call trace for logging/eval)
"""
