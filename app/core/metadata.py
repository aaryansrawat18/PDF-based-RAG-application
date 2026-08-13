from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

_EXACT_KEYWORD_FIELDS = ("section", "document", "chunk_id", "content_type")


def _as_dict(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    if filters is None:
        return {}
    if hasattr(filters, "model_dump"):
        return dict(filters.model_dump(exclude_none=True))
    return {key: value for key, value in dict(filters).items() if value is not None}


def filters_to_qdrant(filters: Mapping[str, Any] | None) -> Filter | None:
    """Turn AskRequest.filters into a Qdrant Filter (AND of all set fields)."""
    data = _as_dict(filters)
    if not data:
        return None

    must: list[FieldCondition] = []
    for field in _EXACT_KEYWORD_FIELDS:
        value = data.get(field)
        if value is None or value == "":
            continue
        must.append(FieldCondition(key=field, match=MatchValue(value=str(value))))

    if data.get("page") is not None:
        must.append(FieldCondition(key="page", match=MatchValue(value=int(data["page"]))))

    page_gte = data.get("page_gte")
    page_lte = data.get("page_lte")
    if page_gte is not None or page_lte is not None:
        must.append(
            FieldCondition(
                key="page",
                range=Range(
                    gte=int(page_gte) if page_gte is not None else None,
                    lte=int(page_lte) if page_lte is not None else None,
                ),
            )
        )

    return Filter(must=must) if must else None
