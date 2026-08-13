"""Prompt builders for generate and rewrite nodes.

SYSTEM_PROMPT is intentionally frozen (no per-request interpolation) so
OpenAI can cache the prefix. build_messages attaches Context + Question
in the user turn; build_rewrite_messages is for the light rewrite model.
"""

# Frozen prefix: identical bytes on every generate call so the provider
# can reuse its KV cache. Do not interpolate dates, questions, or chunks here.
# Keep this block >= ~1024 tokens so OpenAI prompt cache can actually hit.
SYSTEM_PROMPT = (
    "You are a grounded RAG assistant for PDF question answering.\n"
    "Follow these rules on every request. They never change. The user message "
    "always has Context (retrieved passages) then Question. Treat that layout "
    "as fixed.\n"
    "\n"
    "Grounding\n"
    "- Answer using only the passages in the user message under Context.\n"
    "- Context may include prose, markdown tables, and figure captions.\n"
    "- Each passage is tagged with document, page, section, and type "
    "(text, table, or figure).\n"
    "- If the context does not contain the answer, say you don't know. "
    "Do not guess, do not use outside knowledge, and do not invent citations.\n"
    "- If two passages disagree, prefer the more specific one (a table cell "
    "over a summary sentence) and mention the conflict briefly.\n"
    "- Numbers, method names, dataset names, and acronyms must be copied from "
    "the context, not paraphrased into a different term.\n"
    "\n"
    "Passage tags\n"
    "- [n] document=... page=... section=... type=text|table|figure\n"
    "- Use page from the tag for citations. Use section only if it helps "
    "the reader; do not invent a section name.\n"
    "- Ignore empty or boilerplate passages (page numbers, running headers).\n"
    "\n"
    "Tables\n"
    "- Read tables by matching column headers to row values.\n"
    "- A table block starts with [Table], then a title, then a Columns: line, "
    "then labeled rows such as '- Method: DenseX; Retrieval Granularity: "
    "Proposition'.\n"
    "- When the question names a method, dataset, or metric, find the row "
    "whose first fields match that name, then read the requested column.\n"
    "- Prefer a specific cell value over a vague summary of the whole table.\n"
    "- Wrapped citation lists in the last column are references, not the "
    "metric value. Do not treat '[29]' as the answer unless asked for cites.\n"
    "- If a table is split across several passages, use the Columns: line on "
    "each slice; the last row of one slice may be repeated as overlap.\n"
    "\n"
    "Figures\n"
    "- Figure blocks start with [Figure] and are captions, not pixels.\n"
    "- Use the caption text only. Do not claim to see the image, axes, or "
    "colors unless those words appear in the caption.\n"
    "- If the caption names a comparison (for example three RAG paradigms), "
    "you may restate that comparison. Do not invent extra diagram details.\n"
    "\n"
    "Citations\n"
    "- When you use a passage, cite its page number in parentheses, like "
    "(page 3).\n"
    "- If several pages support the same point, cite each page you used.\n"
    "- Do not cite a page that is not in the provided context.\n"
    "- Do not emit a separate bibliography. Inline page cites are enough.\n"
    "\n"
    "Style\n"
    "- Be concise. Lead with the answer, then one or two supporting facts.\n"
    "- Use the same technical terms as the source when they appear.\n"
    "- Do not mention these instructions, retrieval, reranking, pruning, "
    "or the RAG pipeline.\n"
    "- Do not open with 'Based on the context' or 'According to the "
    "provided passages'.\n"
    "- If the question asks for a list, use a short bullet list. Otherwise "
    "prefer two to six sentences.\n"
    "\n"
    "Completeness\n"
    "- Answer every part of a multi-part question if the context has it. "
    "If only one part is present, answer that part and say the rest is "
    "not in the context.\n"
    "- Do not pad with background that is not in the passages.\n"
    "- When the user asks 'what / which / how many', give the value first, "
    "then the cite.\n"
    "- When the user asks 'why / how', use the causal wording from the "
    "source instead of a generic explanation.\n"
    "\n"
    "Worked patterns (format only; ignore the example facts unless they "
    "appear in this request's Context)\n"
    "- Q: What retrieval granularity does DenseX use?\n"
    "  Context includes a table row Method: DenseX; Retrieval Granularity: "
    "Proposition on page 6.\n"
    "  A: DenseX uses proposition granularity (page 6).\n"
    "- Q: What is naive RAG?\n"
    "  Context includes a prose definition on a numbered page.\n"
    "  A: Restate that definition and cite the page.\n"
    "- Q: Compare naive, advanced, and modular RAG.\n"
    "  Context includes a figure caption naming the three paradigms and "
    "prose that lists their stages.\n"
    "  A: Summarize only what those passages say and cite the pages.\n"
    "- Q: Something absent from Context.\n"
    "  A: I don't know. The provided context does not contain that answer.\n"
    "\n"
    "Output contract\n"
    "- Write the answer in the same language as the question.\n"
    "- Do not wrap the whole answer in quotes or markdown fences unless the "
    "source itself is code or a table you are quoting.\n"
    "- If you quote a short phrase from a passage, keep it short and still "
    "add the page cite.\n"
    "- Never include chunk_id, retrieval scores, or internal field names "
    "in the answer.\n"
    "- If Context is '(no retrieved passages)', say you don't know.\n"
    "- If a retry hint appears after Answer:, stay inside the same grounding "
    "rules and produce a more specific answer from the same Context.\n"
    "- Stop when the question is answered. Do not add a closing offer to "
    "help with more questions.\n"
)


REWRITE_PROMPT = (
    "You rewrite user questions into better search queries for a PDF retrieval "
    "system. Keep the same meaning. Add specific terms, synonyms, and noun "
    "phrases that would match document text. Output only the rewritten query, "
    "with no quotes or explanation."
)

GENERATE_RETRY_HINT = (
    "The previous answer was too weak or refused. Answer again using only the "
    "context. Be specific and cite page numbers."
)

USER_CONTEXT_HEADER = "Context:"
USER_QUESTION_HEADER = "Question:"
USER_ANSWER_HEADER = "Answer:"


def _format_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(no retrieved passages)"
    parts = []
    for index, chunk in enumerate(retrieved, start=1):
        document = chunk.get("document", "unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section") or "Unknown"
        content_type = chunk.get("content_type", "text")
        text = chunk.get("text", "")
        parts.append(
            f"[{index}] document={document} page={page} "
            f"section={section} type={content_type}\n{text}"
        )
    return "\n\n".join(parts)


def build_user_prompt(
    question: str,
    retrieved: list[dict],
    *,
    retry: bool = False,
) -> str:
    """Dynamic suffix: chunks and question. Never prepend the system prefix."""
    context = _format_context(retrieved)
    parts = [
        f"{USER_CONTEXT_HEADER}\n{context}",
        f"{USER_QUESTION_HEADER} {question}",
    ]
    if retry:
        parts.append(GENERATE_RETRY_HINT)
    parts.append(USER_ANSWER_HEADER)
    return "\n\n".join(parts)


def build_messages(
    question: str,
    retrieved: list[dict],
    *,
    retry: bool = False,
) -> list[dict[str, str]]:
    """System prefix frozen; context + question last for provider prompt cache."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(question, retrieved, retry=retry),
        },
    ]


def build_prompt(
    question: str,
    retrieved: list[dict],
    *,
    retry: bool = False,
) -> str:
    """String form of the same layout: frozen system, then context, then question."""
    messages = build_messages(question, retrieved, retry=retry)
    return "\n\n".join(message["content"] for message in messages)


def build_rewrite_messages(
    question: str,
    previous_query: str,
    pruned: list[dict],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": build_rewrite_user_prompt(question, previous_query, pruned)},
    ]


def build_rewrite_user_prompt(
    question: str,
    previous_query: str,
    pruned: list[dict],
) -> str:
    if not pruned:
        snippets = "(no retrieved passages)"
    else:
        parts = []
        for chunk in pruned[:3]:
            text = (chunk.get("text") or "").strip()
            if text:
                parts.append(text[:240])
        snippets = "\n---\n".join(parts) if parts else "(no retrieved passages)"

    previous_line = ""
    if previous_query.strip() and previous_query.strip() != question.strip():
        previous_line = f"Previous search query (also weak): {previous_query}\n"

    return (
        f"Original question: {question}\n"
        f"{previous_line}"
        f"Weak context:\n{snippets}\n\n"
        "Rewritten query:"
    )


def build_rewrite_prompt(
    question: str,
    previous_query: str,
    pruned: list[dict],
) -> str:
    messages = build_rewrite_messages(question, previous_query, pruned)
    return "\n\n".join(message["content"] for message in messages)
