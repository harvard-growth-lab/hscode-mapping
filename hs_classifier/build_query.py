"""Build a single classifier query from one raw row."""

from dataclasses import dataclass


@dataclass
class QueryInput:
    query: str
    context: str = ""


def _clean(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def build_query(
    row: dict,
    text_field: str = "product_description",
    context_fields: tuple[str, ...] = ("container_description", "item_unit"),
) -> QueryInput:
    """Build one query string and optional context from a raw row dict."""
    query = _clean(row.get(text_field, ""))

    context_parts = []
    for field in context_fields:
        value = _clean(row.get(field, ""))
        if value:
            context_parts.append(value)

    return QueryInput(query=query, context=" | ".join(context_parts))
