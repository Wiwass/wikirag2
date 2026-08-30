"""Command-line interface. Thin: parses arguments, opens the KB, delegates.

Planned commands
----------------
  wikirag kb list | create <name> | info <name>
  wikirag ingest <kb> [paths...]          # runs pending jobs; resumable
  wikirag set-level <kb> <S0..S3>         # raise/lower synthesis level
  wikirag export-vault <kb>               # render markdown vault from store
  wikirag forget <kb> <doc>
  wikirag query <kb> "..." [--depth R0..R3]
  wikirag chat <kb> [--depth ...]
  wikirag costs <kb>                      # summarize llm_calls metering
  wikirag eval <kb> [--grid ...]          # benchmark harness entry point

No business logic here (v1 lesson: cli.py accumulated orchestration that
belonged to the pipeline). Argument parsing via argparse or typer (decide at
implementation time; typer only if it earns its dependency).
"""
