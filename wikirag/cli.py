"""Command-line interface. Thin: parses arguments, opens the KB, delegates.

Commands
--------
  wikirag kb list | create <name> | info <name>
  wikirag ingest <kb> [paths...]          # runs pending jobs; resumable
  wikirag set-level <kb> <S0..S3>         # raise/lower synthesis level
  wikirag forget <kb> <doc_id>
  wikirag query <kb> "..." [--depth R0..R3] [--budget N] [--fake]
  wikirag export-vault <kb> [--clean]      # render vault/ from the store
  wikirag costs <kb> [--by stage,role,model]
  wikirag eval <kb> [--depths R0,R1] [--budgets 2500,10000] [--out DIR]
                    [--dataset PATH] [--compare BASELINE] [--fake]

Later waves add the rest of the planned surface (chat, consolidate);
they arrive with the modules they drive, not before.

No business logic here (v1 lesson: cli.py accumulated orchestration that
belonged to the pipeline). Every command opens the KB in exactly one place
(`wikirag.config.open_kb`) and hands the handle to a pipeline function:
ingestion's runner, `answer.answer`, the store, `llm.meter.summarize`.
Argument parsing is argparse — typer would not have earned a dependency.

The one thing the CLI does own is provider construction: it maps the
`answer` role to a model (kb_config `answer_model`) and wraps the provider
in MeteredClient, so every call from a query lands in llm_calls. `--fake`
swaps in FakeLLMClient for offline smoke tests and CI.

Exit codes: 0 success, 1 the operation ran but something failed (ingest
errors, unknown document, provider failure), 2 the request was wrong
(unknown KB, bad name or level, argparse usage errors).

Public API
----------
- main(argv=None) -> int
- build_parser() -> argparse.ArgumentParser
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from wikirag.answer import (
    DEFAULT_BUDGET_TOKENS,
    Answer,
    answer,
)
from wikirag.config import (
    KnowledgeBase,
    RetrievalDepth,
    SynthesisLevel,
    create_kb,
    kb_root,
    list_kbs,
    open_kb,
)
from wikirag.llm.client import AllKeysExhausted, FakeLLMClient, LLMClient, LLMResult
from wikirag.llm.meter import GROUP_FIELDS, MeteredClient, summarize

PROG = "wikirag"

#: kb_config key holding the model for the `answer` role (docs/llm.md:
#: stages reference roles, config maps roles to models).
ANSWER_MODEL_KEY = "answer_model"
DEFAULT_ANSWER_MODEL = "gemini-3.5-flash-lite"
ANSWER_ROLE = "answer"
QUERY_STAGE = "query"

FAKE_ANSWER_TEXT = (
    "(--fake: no model was called. The retrieval pipeline ran and its "
    "sources are listed below.)"
)


# --------------------------------------------------------------------- helpers


def _answer_client(kb: KnowledgeBase, *, fake: bool = False) -> LLMClient:
    """The metered client for the `answer` role of this KB."""
    inner: LLMClient
    if fake:
        inner = FakeLLMClient(LLMResult(text=FAKE_ANSWER_TEXT))
    else:
        # Imported here so that no command which never generates has to load
        # google-genai (or fail because it is missing).
        from wikirag.llm.gemini import GeminiClient

        model = kb.store.config_get(ANSWER_MODEL_KEY, DEFAULT_ANSWER_MODEL)
        assert model is not None
        inner = GeminiClient(model)
    return MeteredClient(inner, store=kb.store, stage=QUERY_STAGE, role=ANSWER_ROLE)


#: kb_config key holding the model for the `judge` role. The judge is pinned
#: per dataset version (docs/evaluation.md): changing it invalidates every
#: cross-run comparison, so it is configured per KB, never inferred.
JUDGE_MODEL_KEY = "judge_model"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_ROLE = "judge"
EVAL_STAGE = "eval"

#: A schema-valid zero judgment, so `eval --fake` exercises the whole grid
#: (answer -> judge -> report) without a provider.
FAKE_JUDGE_JSON = (
    '{"correctness": 0, "completeness": 0, "groundedness": 0,'
    ' "rationale": "(--fake: no judge model was called.)"}'
)


def _judge_client(kb: KnowledgeBase, *, fake: bool = False) -> LLMClient:
    """The metered client for the `judge` role of this KB."""
    inner: LLMClient
    if fake:
        inner = FakeLLMClient(LLMResult(text=FAKE_JUDGE_JSON), model="fake-judge")
    else:
        from wikirag.llm.gemini import GeminiClient

        model = kb.store.config_get(JUDGE_MODEL_KEY, DEFAULT_JUDGE_MODEL)
        assert model is not None
        inner = GeminiClient(model)
    return MeteredClient(inner, store=kb.store, stage=EVAL_STAGE, role=JUDGE_ROLE)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _source_line(entry: Any) -> str:
    """One `Sources:` line for a context entry (assembly's provenance)."""
    if entry.owner_kind == "note":
        return f"  [{entry.ref}] Note: {entry.doc_title or entry.owner_id}"
    title = entry.doc_title or entry.doc_id or entry.owner_id
    if entry.section_title:
        return f"  [{entry.ref}] {title} — {entry.section_title}"
    return f"  [{entry.ref}] {title}"


def _print_answer(reply: Answer, cost: float) -> None:
    print(reply.text)
    if reply.provenance:
        print("\nSources:")
        for entry in reply.provenance:
            print(_source_line(entry))
    if not reply.no_answer:
        print(
            f"\n[{reply.depth.value} | context {reply.context_tokens} tok | "
            f"prompt {reply.usage.prompt_tokens} tok | "
            f"output {reply.usage.output_tokens} tok | ${cost:.4f}]"
        )


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Left-aligned fixed-width table; numbers are pre-formatted as strings."""
    cells = [[str(row[c]) for c in columns] for row in rows]
    widths = [
        max(len(str(col)), *(len(cell[i]) for cell in cells)) if cells else len(str(col))
        for i, col in enumerate(columns)
    ]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(cell[i].ljust(w) for i, w in enumerate(widths)) for cell in cells]
    return "\n".join(lines)


# -------------------------------------------------------------------- commands


def _cmd_kb_list(args: argparse.Namespace) -> int:
    names = list_kbs()
    if not names:
        print(f"no knowledge bases in {kb_root()}")
        return 0
    for name in names:
        print(name)
    return 0


def _cmd_kb_create(args: argparse.Namespace) -> int:
    with create_kb(args.name) as kb:
        print(f"created knowledge base {kb.name!r} at {kb.root}")
    return 0


def _cmd_kb_info(args: argparse.Namespace) -> int:
    with open_kb(args.name) as kb:
        documents = kb.store.list_documents()
        chunks = sum(len(kb.store.get_document_chunks(d["id"])) for d in documents)
        ingested = sum(1 for d in documents if d["status"] == "ingested")
        print(f"knowledge base: {kb.name}")
        print(f"path:           {kb.root}")
        print(f"documents:      {len(documents)} ({ingested} ingested)")
        print(f"chunks:         {chunks}")
        print(f"synthesis:      {kb.synthesis_level.value}")
        print("config:")
        for key, value in sorted(kb.config.items()):
            print(f"  {key} = {value}")
    return 0


def _stage_inputs(kb: KnowledgeBase, paths: Sequence[str]) -> int:
    """Copy the given PDFs into the inbox; returns how many were rejected."""
    rejected = 0
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            _warn(f"{path}: no such file, skipped")
            rejected += 1
            continue
        if path.suffix.lower() != ".pdf":
            _warn(f"{path.name}: not a PDF, skipped")
            rejected += 1
            continue
        shutil.copy2(path, kb.raw_dir / path.name)
    return rejected


def _cmd_ingest(args: argparse.Namespace) -> int:
    from wikirag.ingestion import index_notes, register_default_stages
    from wikirag.ingestion.runner import discover_inbox, run_pending

    with open_kb(args.kb) as kb:
        rejected = _stage_inputs(kb, args.paths)
        register_default_stages()
        added = discover_inbox(kb)
        print(f"inbox: {len(added)} new document(s)")
        report = run_pending(kb)
        # Synthesis stages write notes; notes are content, so they need the
        # dense channel too (the store's triggers already handle FTS).
        indexed = index_notes(kb)
        if indexed:
            print(f"indexed {indexed} note(s)")
        for stage in sorted(set(report.done) | set(report.failed) | set(report.skipped)):
            print(
                f"  {stage:<10} done {report.done[stage]}  "
                f"failed {report.failed[stage]}  skipped {report.skipped[stage]}"
            )
        if report.docs_completed:
            print(f"completed: {', '.join(report.docs_completed)}")
        for doc_id, stage, error in report.errors:
            print(f"error: {doc_id} [{stage}] {error}", file=sys.stderr)
        if report.stopped_early:
            print("stopped early (quota exhausted); rerun to resume", file=sys.stderr)
        return 1 if (report.errors or rejected) else 0


def _cmd_set_level(args: argparse.Namespace) -> int:
    from wikirag.ingestion.runner import enqueue_missing

    with open_kb(args.kb) as kb:
        kb.synthesis_level = SynthesisLevel.parse(args.level)
        created = enqueue_missing(kb)
        print(f"{kb.name}: synthesis level {kb.synthesis_level.value}, {created} job(s) added")
    return 0


def _cmd_export_vault(args: argparse.Namespace) -> int:
    """Render the markdown vault from the store (a regenerable view)."""
    from wikirag.storage.vault import export_vault

    with open_kb(args.kb) as kb:
        report = export_vault(kb, clean=args.clean)
        print(f"vault: {report.vault_dir}")
        print(
            f"  written {len(report.written)}  updated {len(report.updated)}  "
            f"unchanged {len(report.unchanged)}  removed {len(report.removed)}"
        )
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    with open_kb(args.kb) as kb:
        if not kb.store.forget_document(args.doc_id):
            print(f"error: no document {args.doc_id!r} in {kb.name}", file=sys.stderr)
            return 1
        print(f"forgot {args.doc_id}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    with open_kb(args.kb) as kb:
        client = _answer_client(kb, fake=args.fake)
        before = len(kb.store.get_llm_calls())
        reply = answer(
            kb,
            args.query,
            depth=RetrievalDepth.parse(args.depth),
            client=client,
            budget_tokens=args.budget,
        )
        cost = sum(call["cost"] for call in kb.store.get_llm_calls()[before:])
        _print_answer(reply, cost)
    return 0


def _cmd_costs(args: argparse.Namespace) -> int:
    fields = [f.strip() for f in args.by.split(",") if f.strip()]
    with open_kb(args.kb) as kb:
        rows = summarize(kb.store, group_by=fields)
        if not rows:
            print(f"{kb.name}: no LLM calls recorded")
            return 0
        formatted = [
            {
                **{f: row[f] for f in fields},
                "calls": row["calls"],
                "tokens_in": row["tokens_in"],
                "tokens_out": row["tokens_out"],
                "cost": f"${row['cost']:.4f}",
                "wall_s": f"{row['wall_ms'] / 1000:.1f}",
            }
            for row in rows
        ]
        print(_table(formatted, [*fields, "calls", "tokens_in", "tokens_out", "cost", "wall_s"]))
        total = sum(row["cost"] for row in rows)
        print(f"total: ${total:.4f} over {sum(row['calls'] for row in rows)} call(s)")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from datetime import datetime

    from wikirag.eval.harness import compare, load_dataset, run_grid, write_report

    dataset = Path(args.dataset) if args.dataset else Path("benchmarks") / args.kb / "dataset.json"
    questions = load_dataset(dataset)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = Path(args.out) if args.out else dataset.parent / "runs" / f"grid_{stamp}"

    with open_kb(args.kb) as kb:
        report = run_grid(
            kb,
            questions,
            client=_answer_client(kb, fake=args.fake),
            judge_client=_judge_client(kb, fake=args.fake),
            depths=[d.strip() for d in args.depths.split(",") if d.strip()],
            budgets=[int(b) for b in args.budgets.split(",") if b.strip()],
            out_dir=out_dir,
            resume=not args.no_resume,
            dataset_path=dataset,
            corpus=args.corpus,
        )

    print(f"{report.kb} [{report.synthesis_level}] -> {report.out_dir}")
    print(
        _table(
            [
                {
                    "depth": cell["depth"],
                    "budget": cell["budget_tokens"],
                    "status": cell["status"],
                    "judged": f"{cell['judged']}/{cell['n']}",
                    "score": "—" if cell["mean_total"] is None else f"{cell['mean_total']:.1f}",
                    "sem": "—" if cell["sem_total"] is None else f"±{cell['sem_total']:.1f}",
                    "cost": "—" if cell["cost"] is None else f"${cell['cost']:.4f}",
                }
                for cell in report.rows()
            ],
            ["depth", "budget", "status", "judged", "score", "sem", "cost"],
        )
    )
    print(f"report: {write_report(report.out_dir)}")

    if args.compare:
        for cell in report.completed():
            print(f"\nvs baseline — {cell.artifact}")
            rows = compare(args.compare, report.out_dir / cell.artifact)
            print(
                _table(
                    [
                        {
                            "subset": row["subset"],
                            "baseline": f"{row['baseline_mean']:.1f}"
                            if row["baseline_mean"] is not None
                            else "—",
                            "run": f"{row['run_mean']:.1f}"
                            if row["run_mean"] is not None
                            else "—",
                            "delta": f"{row['delta']:+.1f}" if row["delta"] is not None else "—",
                            "delta_sem": f"±{row['delta_sem']:.1f}"
                            if row["delta_sem"] is not None
                            else "—",
                            "verdict": row["verdict"],
                        }
                        for row in rows
                    ],
                    ["subset", "baseline", "run", "delta", "delta_sem", "verdict"],
                )
            )

    if report.aborted:
        print(f"stopped early ({report.aborted}); rerun to resume", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(dest="command", required=True)

    kb_parser = subcommands.add_parser("kb", help="knowledge base lifecycle")
    kb_sub = kb_parser.add_subparsers(dest="kb_command", required=True)
    kb_sub.add_parser("list", help="list knowledge bases").set_defaults(func=_cmd_kb_list)
    create = kb_sub.add_parser("create", help="create a knowledge base")
    create.add_argument("name")
    create.set_defaults(func=_cmd_kb_create)
    info = kb_sub.add_parser("info", help="show documents, chunks, level, config")
    info.add_argument("name")
    info.set_defaults(func=_cmd_kb_info)

    ingest = subcommands.add_parser("ingest", help="add PDFs and run pending jobs")
    ingest.add_argument("kb")
    ingest.add_argument("paths", nargs="*", help="PDFs to copy into the inbox")
    ingest.set_defaults(func=_cmd_ingest)

    set_level = subcommands.add_parser("set-level", help="set the synthesis level")
    set_level.add_argument("kb")
    set_level.add_argument("level", help="S0, S1, S2 or S3")
    set_level.set_defaults(func=_cmd_set_level)

    forget = subcommands.add_parser("forget", help="delete a document and its derivatives")
    forget.add_argument("kb")
    forget.add_argument("doc_id")
    forget.set_defaults(func=_cmd_forget)

    export_vault_cmd = subcommands.add_parser(
        "export-vault", help="render the markdown vault from the store"
    )
    export_vault_cmd.add_argument("kb")
    export_vault_cmd.add_argument(
        "--clean",
        action="store_true",
        help="also remove vault files with nothing behind them in the store",
    )
    export_vault_cmd.set_defaults(func=_cmd_export_vault)

    query = subcommands.add_parser("query", help="ask one question")
    query.add_argument("kb")
    query.add_argument("query")
    query.add_argument(
        "--depth",
        default=RetrievalDepth.R1.value,
        help="retrieval depth R0..R3 (default: R1)",
    )
    query.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET_TOKENS,
        help=f"prompt+answer token envelope (default: {DEFAULT_BUDGET_TOKENS})",
    )
    query.add_argument(
        "--fake",
        action="store_true",
        help="answer with the offline fake client (smoke tests, CI)",
    )
    query.set_defaults(func=_cmd_query)

    costs = subcommands.add_parser("costs", help="summarize recorded LLM calls")
    costs.add_argument("kb")
    costs.add_argument(
        "--by",
        default=",".join(GROUP_FIELDS),
        help=f"comma-separated grouping fields from {','.join(GROUP_FIELDS)}",
    )
    costs.set_defaults(func=_cmd_costs)

    # Imported here, and not at module scope, so that only `eval` pays for the
    # harness (and so the two default axes have exactly one definition).
    from wikirag.eval.harness import DEFAULT_BUDGETS as EVAL_DEFAULT_BUDGETS
    from wikirag.eval.harness import DEFAULT_DEPTHS as EVAL_DEFAULT_DEPTHS

    evaluate = subcommands.add_parser("eval", help="run the depth x budget benchmark grid")
    evaluate.add_argument("kb")
    evaluate.add_argument(
        "--depths",
        default=",".join(d.value for d in EVAL_DEFAULT_DEPTHS),
        help="comma-separated retrieval depths (default: %(default)s)",
    )
    evaluate.add_argument(
        "--budgets",
        default=",".join(str(b) for b in EVAL_DEFAULT_BUDGETS),
        help="comma-separated token budgets (default: %(default)s)",
    )
    evaluate.add_argument("--out", help="run directory (default: <dataset dir>/runs/grid_<ts>)")
    evaluate.add_argument("--dataset", help="gold dataset (default: benchmarks/<kb>/dataset.json)")
    evaluate.add_argument("--compare", help="frozen baseline run dir or artifact to diff against")
    evaluate.add_argument(
        "--corpus", default="", help="one-line corpus description for the manifest"
    )
    evaluate.add_argument(
        "--no-resume", action="store_true", help="re-run every cell, even completed ones"
    )
    evaluate.add_argument(
        "--fake", action="store_true", help="run the grid with the offline fake clients"
    )
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (AllKeysExhausted, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
