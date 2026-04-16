from __future__ import annotations

from pathlib import Path

import yaml


def normalize_protected_terms(raw_terms: object) -> tuple[list[str], dict[str, list[str]]]:
    if isinstance(raw_terms, dict):
        equivalents: dict[str, list[str]] = {}
        preferred_terms: list[str] = []
        for canonical_term, variants in raw_terms.items():
            canonical_key = str(canonical_term).strip().lower()
            if isinstance(variants, list):
                normalized_variants = [str(value).strip() for value in variants if str(value).strip()]
            else:
                normalized_variants = [str(variants).strip()] if str(variants).strip() else []
            if not normalized_variants:
                continue
            equivalents[canonical_key] = normalized_variants
            preferred_terms.append(normalized_variants[0])
        return preferred_terms, equivalents

    if isinstance(raw_terms, list):
        terms = [str(value).strip() for value in raw_terms if str(value).strip()]
    else:
        terms = []
    equivalents = {term.strip().lower(): [term] for term in terms}
    return terms, equivalents


def normalize_forced_translations(raw_terms: object) -> dict[str, dict[str, dict[str, str]]]:
    if not isinstance(raw_terms, dict):
        return {}

    normalized: dict[str, dict[str, dict[str, str]]] = {}
    for canonical_term, language_map in raw_terms.items():
        canonical_key = str(canonical_term).strip().lower()
        if not canonical_key or not isinstance(language_map, dict):
            continue
        normalized_languages: dict[str, dict[str, str]] = {}
        for language_code, translated_value in language_map.items():
            normalized_code = str(language_code).strip().lower()
            if not normalized_code:
                continue
            forms: dict[str, str] = {}
            if isinstance(translated_value, dict):
                singular = str(translated_value.get("singular", "")).strip()
                plural = str(translated_value.get("plural", "")).strip()
                if singular:
                    forms["singular"] = singular
                if plural:
                    forms["plural"] = plural
            else:
                singular = str(translated_value).strip()
                if singular:
                    forms["singular"] = singular
            if forms:
                normalized_languages[normalized_code] = forms
        if normalized_languages:
            normalized[canonical_key] = normalized_languages
    return normalized


def load_glossary(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {
            "terms": {},
            "do_not_translate": [],
            "protected_terms": [],
            "protected_term_equivalents": {},
            "forced_translations": {},
        }
    glossary_path = Path(path)
    if not glossary_path.exists():
        raise FileNotFoundError(f"Glossary file not found: {glossary_path}")
    with glossary_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    protected_terms, protected_term_equivalents = normalize_protected_terms(data.get("protected_terms", []))
    return {
        "terms": dict(data.get("terms", {})),
        "do_not_translate": list(data.get("do_not_translate", [])),
        "protected_terms": protected_terms,
        "protected_term_equivalents": protected_term_equivalents,
        "forced_translations": normalize_forced_translations(data.get("forced_translations", {})),
    }
