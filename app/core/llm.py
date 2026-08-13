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
    provider = settings.llm_provider.strip().lower()
    if provider == "openai":
        return _generate_openai(
            resolved,
            model=model,
            use_prompt_cache=use_prompt_cache,
        )
    if provider == "gemini":
        return _generate_gemini(resolved, model=model)
    raise ValueError(f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. Use openai or gemini.")


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


def _generate_gemini(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set it.")
    from google import genai

    prompt = "\n\n".join(item["content"] for item in messages)
    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=model or settings.gemini_model,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()
