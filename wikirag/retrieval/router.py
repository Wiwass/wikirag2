"""Query router: decide depth and decompose complex questions.

Ports v1's query classifier (it measured well: 28/30 on the gold set) into
the R-slider world:
- classify simple vs complex; complex questions get sub-questions
  (concrete, self-contained, overview-aware) as additional search anchors;
- output feeds depth selection when the user didn't set --depth
  explicitly: simple -> R1, complex -> R2, with config overrides;
- retrieval_guidance survives as a note to the answering model;
- structured output via json_schema; on failure, degrade to simple/R1
  (never block the answer path).

Planned API
-----------
- classify(client, query, history, overview_text) -> Classification
"""
