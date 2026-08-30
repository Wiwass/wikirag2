# LLM layer

Provider-agnostic by design: cheap ingest is a stated goal, and the freedom to mix providers (or local models) per stage is how it gets pursued.

## Client protocol

A minimal interface, roughly:

```python
class LLMClient(Protocol):
    def generate(self, *, system: str, user: str | list[Part],
                 json_schema: dict | None = None,
                 max_output_tokens: int | None = None) -> LLMResult

@dataclass
class LLMResult:
    text: str
    truncated: bool
    usage: Usage          # prompt/output/total tokens, cached tokens
    cost_estimate: float  # from a per-model price table
```

- **Structured output** goes through `json_schema` (provider-native constrained decoding where available). Combined with the no-verbatim-echo rule, v1's JSON repair heuristics should become dead weight; keep a thin `parse_json` fallback and log every time it actually fires.
- **Roles, not models, in config**: stages reference roles (`enrich`, `synth`, `answer`, `judge`, `classify`); `kb_config`/global config maps roles → provider+model. Swapping the enrich role to a local model is a config change.
- **Files/PDF upload** stays behind a provider capability flag; only Gemini needs it initially.

## First implementation: Gemini

Port from v1, which had real production hardening worth keeping:

- retry with backoff, 429 quota detection, broken-key detection (401/403),
- key pool with linear rotation and optional carousel mode,
- truncation detection via finish reason.

Strip from the v1 class what does not belong here: logging setup, path resolution, output cleanup (moves to a small text-utils module or the calling stage).

## Cost metering

Every call records `(kb, stage, role, model, tokens, cost, timestamp)` to a `llm_calls` table in the KB store. This is infrastructure for the ingestion research program: method comparisons in [ingestion.md](ingestion.md) cite these numbers, and `wikirag costs <kb>` summarizes what a KB actually cost to build.

## Local models

Two distinct kinds, kept separate in config:

- **Embedding + reranker** (sentence-transformers): always local, one shared lazily-loaded instance per process, preloaded at chat startup (v1 pattern, kept).
- **Generative local models** (optional, research): an `LLMClient` implementation over Ollama/llama.cpp for cheap enrich-stage work. Same protocol, same metering (cost 0, wall time still recorded).
