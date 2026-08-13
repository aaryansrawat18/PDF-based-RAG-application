from app.config import settings


def generate(prompt: str) -> str:
    provider = settings.llm_provider.strip().lower()
    if provider == "openai":
        return _generate_openai(prompt)
    if provider == "gemini":
        return _generate_gemini(prompt)
    raise ValueError(f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. Use gemini or openai.")


def _generate_gemini(prompt: str) -> str:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set it.")
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def _generate_openai(prompt: str) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and set it.")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return text.strip()
