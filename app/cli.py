import argparse
import json
import sys

from app.core.ingest import ingest_pdfs
from app.core.rag_chain import run_rag
from app.core.vectorstore import close_client
from app.eval.tracing import configure_tracing, flush_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser(
        "ingest",
        help="Load PDFs, chunk, embed, upsert to Qdrant, rebuild BM25",
    )
    ingest_parser.add_argument("--pdf", default=None, help="Optional path to a single PDF")

    ask_parser = sub.add_parser(
        "ask",
        help="Run the RAG graph (retrieve → rewrite/retry if needed → generate)",
    )
    ask_parser.add_argument("question", help="Question to ask the RAG graph")
    ask_parser.add_argument(
        "--filters",
        default=None,
        help='Optional JSON filters, e.g. {"section": "Retrieval", "page_gte": 10}',
    )
    ask_parser.add_argument(
        "--pipeline",
        choices=("advanced", "baseline"),
        default="advanced",
        help="advanced = full graph; baseline = vector → generate",
    )

    eval_parser = sub.add_parser(
        "eval",
        help="Score baseline vs advanced on data/eval/rag_eval.jsonl",
    )
    eval_parser.add_argument("--dataset", default=None)
    eval_parser.add_argument(
        "--pipeline",
        choices=("baseline", "advanced", "both"),
        default="both",
    )
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--upload", action="store_true")
    eval_parser.add_argument("--overwrite", action="store_true")
    eval_parser.add_argument("--out", default=None)

    args = parser.parse_args()
    if args.command == "eval":
        from app.eval.benchmark import DEFAULT_RESULTS_PATH, main as eval_main

        argv = ["--pipeline", args.pipeline]
        if args.dataset:
            argv.extend(["--dataset", args.dataset])
        if args.limit:
            argv.extend(["--limit", str(args.limit)])
        if args.upload:
            argv.append("--upload")
        if args.overwrite:
            argv.append("--overwrite")
        argv.extend(["--out", args.out or str(DEFAULT_RESULTS_PATH)])
        sys.exit(eval_main(argv))

    configure_tracing()
    try:
        if args.command == "ingest":
            results = ingest_pdfs(args.pdf)
            print(json.dumps(results, indent=2))
            return

        filters = json.loads(args.filters) if args.filters else None
        output = run_rag(args.question, filters=filters, pipeline=args.pipeline)
        print(output["answer"])
        print("\nSources:")
        print(json.dumps(output["sources"], indent=2))
        extra = {
            "latency_ms": output.get("latency_ms", 0),
            "pipeline": output.get("pipeline", args.pipeline),
        }
        if output.get("rewritten_query"):
            extra["rewritten_query"] = output["rewritten_query"]
            extra["retry_count"] = output.get("retry_count", 0)
        print("\nMeta:")
        print(json.dumps(extra, indent=2))
    finally:
        flush_traces()
        close_client()


if __name__ == "__main__":
    main()
