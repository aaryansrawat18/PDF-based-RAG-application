"""Load the local RAG eval JSONL and optionally upload it to LangSmith."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DATASET_NAME = "rag-eval"
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "rag_eval.jsonl"
)

_REQUIRED = ("question", "reference_answer", "expected_pages")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_DATASET_PATH
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _parse_pages(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [int(raw)]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace(";", ",").split(",")]
        pages: list[int] = []
        for part in parts:
            if not part:
                continue
            pages.append(int(part))
        return pages
    if isinstance(raw, list):
        pages = []
        for item in raw:
            pages.extend(_parse_pages(item))
        return pages
    raise ValueError(f"expected_pages must be a list of ints, got {raw!r}")


def parse_example(row: dict, *, line_no: int | None = None) -> dict:
    """Normalize one JSONL object into a scored-example dict."""
    missing = [key for key in _REQUIRED if key not in row]
    if missing:
        where = f" on line {line_no}" if line_no is not None else ""
        raise ValueError(f"eval example{where} missing fields: {missing}")
    question = str(row["question"] or "").strip()
    if not question:
        where = f" on line {line_no}" if line_no is not None else ""
        raise ValueError(f"eval example{where} has an empty question")
    pages = _parse_pages(row.get("expected_pages"))
    if not pages:
        where = f" on line {line_no}" if line_no is not None else ""
        raise ValueError(f"eval example{where} has no expected_pages")
    example_id = str(row.get("id") or "").strip()
    if not example_id:
        example_id = f"q{line_no:02d}" if line_no is not None else question[:40]
    return {
        "id": example_id,
        "question": question,
        "reference_answer": str(row.get("reference_answer") or "").strip(),
        "expected_pages": pages,
    }


def load_eval_dataset(path: str | Path | None = None) -> list[dict]:
    """Load `question`, `reference_answer`, `expected_pages` rows from JSONL."""
    dataset_path = resolve_dataset_path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Eval dataset not found: {dataset_path}")

    examples: list[dict] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"eval example on line {line_no} must be a JSON object")
            examples.append(parse_example(row, line_no=line_no))
    if not examples:
        raise ValueError(f"Eval dataset is empty: {dataset_path}")
    return examples


def examples_to_langsmith(examples: list[dict]) -> list[dict]:
    """Shape local examples for LangSmith create_examples()."""
    payload = []
    for example in examples:
        payload.append(
            {
                "inputs": {"question": example["question"]},
                "outputs": {
                    "reference_answer": example["reference_answer"],
                    "expected_pages": example["expected_pages"],
                },
                "metadata": {"id": example.get("id")},
            }
        )
    return payload


def upload_langsmith_dataset(
    examples: list[dict] | None = None,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    path: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Create or replace a LangSmith dataset from the local JSONL.

    Returns dataset id/name/count. No-ops are not used: missing API key raises.
    """
    from langsmith import Client

    rows = examples if examples is not None else load_eval_dataset(path)
    if not rows:
        raise ValueError("No examples to upload")

    client = Client()
    existed = False
    try:
        existed = bool(client.has_dataset(dataset_name=dataset_name))
    except Exception:
        existed = False

    if existed and overwrite:
        dataset = client.read_dataset(dataset_name=dataset_name)
        existing_ids = [example.id for example in client.list_examples(dataset_id=dataset.id)]
        if existing_ids:
            client.delete_examples(example_ids=existing_ids)
    elif existed:
        dataset = client.read_dataset(dataset_name=dataset_name)
        return {
            "dataset_id": str(dataset.id),
            "dataset_name": dataset.name,
            "count": len(rows),
            "created": False,
            "message": (
                f"Dataset {dataset_name!r} already exists. "
                "Pass overwrite=True to replace examples."
            ),
        }
    else:
        dataset = client.create_dataset(
            dataset_name,
            description=(
                "PDF RAG eval: question, reference_answer, expected_pages. "
                "Used to compare vector-only baseline vs hybrid+rerank+prune."
            ),
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=examples_to_langsmith(rows),
    )
    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset.name,
        "count": len(rows),
        "created": not existed or overwrite,
        "message": f"Uploaded {len(rows)} examples to {dataset_name!r}.",
    }
