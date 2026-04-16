from __future__ import annotations

from translator.config import AppConfig
from translator.providers.base import TranslationProvider
from translator.providers.manual_provider import ManualTranslationProvider
from translator.providers.mock import MockTranslationProvider


def build_provider(
    provider_name: str,
    model: str,
    config: AppConfig | None = None,
) -> TranslationProvider:
    if provider_name == "mock":
        return MockTranslationProvider()
    if provider_name == "manual":
        return ManualTranslationProvider()
    if provider_name == "ollama":
        from translator.providers.ollama_provider import OllamaTranslationProvider

        provider_settings = config.provider_settings("ollama") if config is not None else {}
        return OllamaTranslationProvider(
            model=model,
            base_url=provider_settings.get("base_url"),
            prefer_gpu=config.prefer_gpu if config is not None else True,
            precision=config.precision if config is not None else "auto",
        )
    if provider_name == "openai":
        from translator.providers.openai_provider import OpenAITranslationProvider

        return OpenAITranslationProvider(model=model)
    raise ValueError(f"Unsupported provider: {provider_name}")
