import argparse
import json

from app.core.ingest import ingest_pdfs
from app.core.rag_chain import run_rag
from app.core.vectorstore import close_client


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Load PDFs, chunk, embed, upsert to Qdrant")
    ingest_parser.add_argument("--pdf", default=None, help="Optional path to a single PDF")

    ask_parser = sub.add_parser("ask", help="Run retrieve → generate and print the answer")
    ask_parser.add_argument("question", help="Question to ask the RAG graph")
    ask_parser.add_argument(
        "--filters",
        default=None,
        help='Optional JSON filters, e.g. {"section": "Retrieval", "page_gte": 10}',
    )

    args = parser.parse_args()
    try:
        if args.command == "ingest":
            results = ingest_pdfs(args.pdf)
            print(json.dumps(results, indent=2))
            return

        filters = json.loads(args.filters) if args.filters else None
        output = run_rag(args.question, filters=filters)
        print(output["answer"])
        print("\nSources:")
        print(json.dumps(output["sources"], indent=2))
    finally:
        close_client()


if __name__ == "__main__":
    main()
