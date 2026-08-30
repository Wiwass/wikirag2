"""Gemini implementation of LLMClient.

Ports the production-hardened parts of v1's GeminiPipeline and nothing else:
- retry with backoff; 429/RESOURCE_EXHAUSTED quota detection;
- broken-key detection (401/403 UNAUTHENTICATED / PERMISSION_DENIED);
- key pool with linear rotation and optional carousel mode
  (wait-and-restart when the whole pool is exhausted);
- truncation detection via finish reason; usage extraction.

Deliberately NOT here (v1 mixed these in): logging setup, path resolution,
markdown output cleanup — those belong to callers or small text utils.

PDF/file upload support sits behind the provider capability flag; only
needed if an ingest method chooses native-PDF calls for short documents.
"""
