# Benchmarks

One directory per knowledge base: `benchmarks/<kb>/dataset.json` is its gold set, `benchmarks/<kb>/runs/<run_id>/` holds what `wikirag eval` recorded for one grid.

## What is in the repository

The reference corpus is the Ars Magica library (`arsmagicaV2`), a commercial RPG whose books cannot be redistributed. Anything that quotes them stays local and is ignored by git:

| File | In the repository |
|---|---|
| `arsmagicaV2/dataset.json`: the 30 gold questions, answers and key points | no |
| `arsmagicaV2/runs/*/answers_*.json`: every recorded answer and judgment | no |
| `arsmagicaV2/runs/*/manifest.json`: models, budgets, dataset hash, commit | yes |
| `arsmagicaV2/runs/*/summary.md`, `retrieval_report.json`: aggregate scores and retrieval hit rates | yes |

The tests that read the local files skip when they are missing.

## Building a gold set for another corpus

`wikirag eval <kb>` reads `benchmarks/<kb>/dataset.json` (or `--dataset PATH`):

```json
{
    "kb": "books",
    "version": 1,
    "questions": [
        {
            "id": "q01",
            "type": "simple",
            "question": "What does the author say about X?",
            "gold_answer": "The reference answer the judge grades against.",
            "key_points": ["a fact the answer must contain", "another one"],
            "source_docs": ["Title of the cited document"],
            "citation": "pages 64, 110"
        }
    ]
}
```

`type` is `simple` (direct lookup) or `complex` (multi-hop); keep the two subsets about the same size, since every score is reported per subset. `id`, `type`, `question` and `gold_answer` are required. `source_docs` and the pages in `citation` feed the retrieval hit rates (`doc_hit`, `page_hit`). See [docs/evaluation.md](../docs/evaluation.md) for how the judge scores answers and how much a difference has to be before it counts.
