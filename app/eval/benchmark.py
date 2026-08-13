"""Run baseline vs advanced RAG on the eval JSONL and print a metric table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from app.core.rag_chain import run_rag
from app.core.vectorstore import close_client
from app.eval.dataset import (
    DEFAULT_DATASET_PATH,
    load_eval_dataset,
    upload_langsmith_dataset,
)
from app.eval.evaluators import (
    aggregate_scores,
    comparison_table,
    format_comparison_table,
    score_prediction,
)
from app.eval.tracing import configure_tracing, flush_traces

DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "last_benchmark.json"
)
PIPELINES = ("baseline", "advanced")


def run_example(example: dict, pipeline: str, filters: dict | None = None) -> dict:
    result = run_rag(example["question"], filters=filters, pipeline=pipeline)
    scored = score_prediction(example, result)
    scored["pipeline"] = pipeline
    return scored


def run_benchmark(
    examples: list[dict],
    pipelines: Iterable[str] = PIPELINES,
    *,
    filters: dict | None = None,
) -> dict:
    pipeline_names = tuple(pipelines)
    per_pipeline: dict[str, dict] = {}
    for name in pipeline_names:
        rows = [run_example(example, name, filters=filters) for example in examples]
        per_pipeline[name] = {
            "rows": rows,
            "summary": aggregate_scores(rows),
        }

    summaries = {name: payload["summary"] for name, payload in per_pipeline.items()}
    table = []
    if "baseline" in summaries and "advanced" in summaries:
        table = comparison_table(summaries)

    return {
        "n": len(examples),
        "pipelines": list(pipeline_names),
        "per_pipeline": per_pipeline,
        "comparison": table,
    }


def save_benchmark(result: dict, path: str | Path | None = None) -> Path:
    target = Path(path) if path else DEFAULT_RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return target


def load_benchmark(path: str | Path | None = None) -> dict:
    target = Path(path) if path else DEFAULT_RESULTS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def _print_progress(example: dict, pipeline: str, scored: dict) -> None:
    print(
        f"[{pipeline}] {example.get('id')}: "
        f"hit={scored['hit_rate']:.0f} mrr={scored['mrr']:.2f} "
        f"rel={scored['relevance']:.2f} {scored['latency_ms']}ms"
    )


def run_benchmark_verbose(
    examples: list[dict],
    pipelines: Iterable[str] = PIPELINES,
) -> dict:
    pipeline_names = tuple(pipelines)
    per_pipeline: dict[str, dict] = {}
    for name in pipeline_names:
        rows = []
        for example in examples:
            scored = run_example(example, name)
            _print_progress(example, name, scored)
            rows.append(scored)
        per_pipeline[name] = {
            "rows": rows,
            "summary": aggregate_scores(rows),
        }
    summaries = {name: payload["summary"] for name, payload in per_pipeline.items()}
    table = []
    if "baseline" in summaries and "advanced" in summaries:
        table = comparison_table(summaries)
    return {
        "n": len(examples),
        "pipelines": list(pipeline_names),
        "per_pipeline": per_pipeline,
        "comparison": table,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark vector-only baseline vs hybrid+rerank+prune",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to rag_eval.jsonl",
    )
    parser.add_argument(
        "--pipeline",
        choices=("baseline", "advanced", "both"),
        default="both",
    )
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N questions")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the JSONL to LangSmith as dataset rag-eval",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing LangSmith dataset examples when uploading",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS_PATH),
        help="Where to write last_benchmark.json",
    )
    args = parser.parse_args(argv)

    configure_tracing()
    try:
        examples = load_eval_dataset(args.dataset)
        if args.limit and args.limit > 0:
            examples = examples[: args.limit]

        if args.upload:
            uploaded = upload_langsmith_dataset(
                examples,
                overwrite=args.overwrite,
            )
            print(uploaded["message"])

        pipelines = PIPELINES if args.pipeline == "both" else (args.pipeline,)
        print(f"Scoring {len(examples)} questions on: {', '.join(pipelines)}")
        result = run_benchmark_verbose(examples, pipelines)
        saved = save_benchmark(result, args.out)
        if result["comparison"]:
            print()
            print(format_comparison_table(result["comparison"]))
        else:
            name = result["pipelines"][0]
            print(json.dumps(result["per_pipeline"][name]["summary"], indent=2))
        print(f"\nWrote {saved}")
        return 0
    finally:
        flush_traces()
        close_client()


if __name__ == "__main__":
    raise SystemExit(main())
