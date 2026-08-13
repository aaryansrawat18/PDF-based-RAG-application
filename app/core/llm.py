"""Chat completions for generate + query rewrite.

Uses the frozen system prompt from app.core.prompts and optional OpenAI
prompt-cache headers so repeated asks reuse the long system prefix.
"""

from functools import lru_cache

from openai import OpenAI

from app.config import settings


def generate(
    prompt: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    model: str | None = None,
    use_prompt_cache: bool = True,
) -> str:
    resolved = _resolve_messages(prompt, messages)
    return _generate_openai(
        resolved,
        model=model,
        use_prompt_cache=use_prompt_cache,
    )


def _resolve_messages(
    prompt: str | None,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if messages:
        return messages
    if prompt and prompt.strip():
        return [{"role": "user", "content": prompt}]
    raise ValueError("generate() needs a prompt string or messages.")


def openai_cache_kwargs(*, use_prompt_cache: bool = True) -> dict:
    """Provider prompt-cache flags for chat.completions.create."""
    if not use_prompt_cache:
        return {}
    payload: dict = {}
    key = (settings.prompt_cache_key or "").strip()
    if key:
        payload["prompt_cache_key"] = key
    retention = (settings.prompt_cache_retention or "").strip()
    if retention:
        payload["prompt_cache_retention"] = retention
    return payload


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and set it.")
    from app.eval.tracing import wrap_openai_client

    return wrap_openai_client(OpenAI(api_key=settings.openai_api_key))


def _generate_openai(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    use_prompt_cache: bool = True,
) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and set it.")
    kwargs: dict = {
        "model": model or settings.openai_model,
        "messages": messages,
    }
    kwargs.update(openai_cache_kwargs(use_prompt_cache=use_prompt_cache))
    response = _openai_client().chat.completions.create(**kwargs)
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return text.strip()
