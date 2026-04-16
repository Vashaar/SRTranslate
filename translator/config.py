from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from translator.models import LanguageConfig


DEFAULT_LANGUAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "ar": {"label": "Arabic", "rtl": True, "profile": "balanced", "aliases": ["arabic"]},
    "bn": {"label": "Bengali", "rtl": False, "profile": "balanced", "aliases": ["bengali"]},
    "de": {"label": "German", "rtl": False, "profile": "balanced", "aliases": ["german"]},
    "es": {"label": "Spanish", "rtl": False, "profile": "balanced", "aliases": ["spanish"]},
    "fa": {"label": "Persian", "rtl": True, "profile": "balanced", "aliases": ["persian", "farsi"]},
    "fr": {"label": "French", "rtl": False, "profile": "balanced", "aliases": ["french"]},
    "hi": {"label": "Hindi", "rtl": False, "profile": "balanced", "aliases": ["hindi"]},
    "id": {"label": "Indonesian", "rtl": False, "profile": "balanced", "aliases": ["indonesian", "indonesia"]},
    "it": {"label": "Italian", "rtl": False, "profile": "balanced", "aliases": ["italian"]},
    "ja": {"label": "Japanese", "rtl": False, "profile": "balanced", "aliases": ["japanese"]},
    "ko": {"label": "Korean", "rtl": False, "profile": "balanced", "aliases": ["korean"]},
    "ms": {"label": "Malay", "rtl": False, "profile": "balanced", "aliases": ["malay"]},
    "nl": {"label": "Dutch", "rtl": False, "profile": "balanced", "aliases": ["dutch"]},
    "pt": {"label": "Portuguese", "rtl": False, "profile": "balanced", "aliases": ["portuguese"]},
    "ru": {"label": "Russian", "rtl": False, "profile": "balanced", "aliases": ["russian", "russia"]},
    "sw": {"label": "Swahili", "rtl": False, "profile": "balanced", "aliases": ["swahili"]},
    "tr": {"label": "Turkish", "rtl": False, "profile": "balanced", "aliases": ["turkish"]},
    "ur": {"label": "Urdu", "rtl": True, "profile": "balanced", "aliases": ["urdu"]},
}


def _normalize_language_code(value: str) -> str:
    return str(value or "").strip().lower()


@dataclass(slots=True)
class AppConfig:
    raw: dict[str, Any]

    @property
    def source_language(self) -> str:
        return str(self.raw.get("source_language", "en"))

    @property
    def provider(self) -> str:
        return str(self.raw.get("provider", "lmstudio"))

    @property
    def target_language(self) -> str:
        return self.resolve_language_code(str(self.raw.get("target_language", "es")))

    @property
    def model(self) -> str:
        configured = self.raw.get("model")
        if configured:
            return str(configured)
        provider_specific = self.provider_settings(self.provider).get("model")
        if provider_specific:
            return str(provider_specific)
        return str("qwen2.5-7b-instruct")

    @property
    def style_profile(self) -> str:
        return str(self.raw.get("style_profile", "balanced"))

    @property
    def deen_mode(self) -> bool:
        return bool(self.raw.get("deen_mode", False))

    @property
    def output_dir(self) -> Path:
        output = self.raw.get("output", {}).get("output_dir", "outputs")
        return Path(output)

    @property
    def low_confidence_threshold(self) -> float:
        return float(
            self.raw.get("translation", {}).get("low_confidence_threshold", 0.7)
        )

    @property
    def retry_low_confidence(self) -> bool:
        return bool(self.raw.get("translation", {}).get("retry_low_confidence", True))

    @property
    def max_repair_attempts(self) -> int:
        return int(self.raw.get("translation", {}).get("max_repair_attempts", 1))

    @property
    def line_rebalancing_enabled(self) -> bool:
        return bool(
            self.raw.get("line_rebalancing", {}).get("enabled", False)
        )

    @property
    def max_chars_per_line(self) -> int:
        return int(
            self.raw.get("line_rebalancing", {}).get("max_chars_per_line", 42)
        )

    @property
    def max_lines_per_subtitle(self) -> int:
        return int(
            self.raw.get("line_rebalancing", {}).get("max_lines_per_subtitle", 2)
        )

    @property
    def translation_batch_size(self) -> int:
        value = int(self.raw.get("translation", {}).get("batch_size", 12))
        return max(5, min(15, value))

    @property
    def alignment_search_radius(self) -> int:
        value = int(self.raw.get("alignment", {}).get("search_radius", 5))
        return max(1, value)

    @property
    def prefer_gpu(self) -> bool:
        return bool(self.raw.get("runtime", {}).get("prefer_gpu", True))

    @property
    def precision(self) -> str:
        return str(self.raw.get("runtime", {}).get("precision", "auto")).lower()

    def resolve_language_code(self, lang: str) -> str:
        normalized = _normalize_language_code(lang)
        if normalized in DEFAULT_LANGUAGE_CONFIGS:
            return normalized
        for code, settings in DEFAULT_LANGUAGE_CONFIGS.items():
            aliases = {_normalize_language_code(alias) for alias in settings.get("aliases", [])}
            if normalized == code or normalized in aliases:
                return code
        for code, settings in self.raw.get("language_settings", {}).items():
            normalized_code = _normalize_language_code(code)
            aliases = {_normalize_language_code(alias) for alias in settings.get("aliases", [])}
            if normalized == normalized_code or normalized in aliases:
                return normalized_code
        return normalized

    def language_settings(self, lang: str) -> dict[str, Any]:
        resolved = self.resolve_language_code(lang)
        return dict(self.raw.get("language_settings", {}).get(resolved, {}))

    def language_config(self, lang: str) -> LanguageConfig:
        resolved = self.resolve_language_code(lang)
        defaults = dict(DEFAULT_LANGUAGE_CONFIGS.get(resolved, {}))
        overrides = self.language_settings(resolved)
        merged = {**defaults, **overrides}
        aliases = []
        for value in defaults.get("aliases", []) + overrides.get("aliases", []):
            normalized = _normalize_language_code(value)
            if normalized and normalized not in aliases:
                aliases.append(normalized)
        return LanguageConfig(
            code=resolved,
            label=str(merged.get("label", resolved.upper())),
            rtl=bool(merged.get("rtl", False)),
            profile=str(merged.get("profile", self.style_profile)),
            aliases=aliases,
            normalize_grammar=bool(merged.get("normalize_grammar", True)),
        )

    def supported_languages(self) -> list[LanguageConfig]:
        configured_codes = {
            self.resolve_language_code(code)
            for code in self.raw.get("language_settings", {}).keys()
        }
        language_codes = sorted(set(DEFAULT_LANGUAGE_CONFIGS) | configured_codes)
        languages = [self.language_config(code) for code in language_codes]
        return sorted(languages, key=lambda language: language.label.lower())

    def provider_settings(self, provider_name: str) -> dict[str, Any]:
        nested = dict(self.raw.get("providers", {}).get(provider_name, {}))
        top_level = self.raw.get(provider_name, {})
        if isinstance(top_level, dict):
            return {**nested, **top_level}
        return nested


def load_config(path: str | Path | None) -> AppConfig:
    config_path = Path(path or "config.yaml")
    if not config_path.exists():
        return AppConfig(raw={})
    with config_path.open("r", encoding="utf-8") as handle:
        return AppConfig(raw=yaml.safe_load(handle) or {})
