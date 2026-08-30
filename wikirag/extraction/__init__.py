"""Extraction: PDF -> ordered, structured, clean text. No LLM calls, ever.

Modules:
    pdf   TOC-aware section extraction with font-size heuristic fallback,
          repeated-line boilerplate removal (ports the good ideas from v1's
          pdf_extractor, drops its dead code)
"""
