from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from urllib import error, request

from translator.models import BatchTranslationRequest, TranslationResult
from translator.providers.base import TranslationProvider
from translator.providers.structured import parse_batch_translation_payload

logger = logging.getLogger(__name__)
LMSTUDIO_SAFE_BATCH_TOKEN_LIMIT = 2600
LMSTUDIO_TEST_BATCH = [
    {"index": 0, "text": "Hello, how are you?"},
    {"index": 1, "text": "This is a test subtitle."},
    {"index": 2, "text": "We are verifying LM Studio integration."},
]
SPANISH_HEURISTIC_MARKERS = {
    "hola",
    "como",
    "estas",
    "esto",
    "subtitulo",
    "verificando",
    "integracion",
    "prueba",
    "somos",
    "estamos",
    "que",
}
SPANISH_HEURISTIC_CHARS = ("\u00e1", "\u00e9", "\u00ed", "\u00f3", "\u00fa", "\u00f1", "\u00bf", "\u00a1")
EXPLANATION_MARKERS = (
    "meaning",
    "i.e.",
    "that is",
    "in other words",
    "es decir",
    "esto significa",
    "يعني",
)


def _looks_like_target_language(texts: list[str], target_language: str) -> bool:
    normalized_target = str(target_language).strip().lower()
    combined_text = " ".join(texts).lower()
    if normalized_target != "spanish":
        return True
    if any(marker in combined_text for marker in SPANISH_HEURISTIC_MARKERS):
        return True
    return any(character in combined_text for character in SPANISH_HEURISTIC_CHARS)


def _estimate_text_tokens(text: str) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _estimate_batch_item_tokens(item) -> int:
    return (
        _estimate_text_tokens(item.source_subtitle_text)
        + _estimate_text_tokens(item.previous_subtitle_text)
        + _estimate_text_tokens(item.next_subtitle_text)
        + _estimate_text_tokens(item.script_context)
        + 24
    )


def _estimate_request_overhead_tokens(request_payload: BatchTranslationRequest) -> int:
    return (
        _estimate_text_tokens(request_payload.source_language)
        + _estimate_text_tokens(request_payload.target_language)
        + _estimate_text_tokens(request_payload.target_language_name)
        + _estimate_text_tokens(request_payload.style_profile)
        + sum(_estimate_text_tokens(term) for term in request_payload.do_not_translate)
        + sum(_estimate_text_tokens(term) for term in request_payload.protected_terms)
        + sum(
            _estimate_text_tokens(source) + _estimate_text_tokens(target)
            for source, target in request_payload.glossary_terms.items()
        )
        + 220
    )


def _estimate_batch_tokens(request_payload: BatchTranslationRequest) -> int:
    return _estimate_request_overhead_tokens(request_payload) + sum(
        _estimate_batch_item_tokens(item) for item in request_payload.items
    )


def _build_token_aware_batches(
    request_payload: BatchTranslationRequest,
    token_limit: int,
) -> list[tuple[BatchTranslationRequest, int]]:
    if not request_payload.items:
        return []

    overhead_tokens = _estimate_request_overhead_tokens(request_payload)
    batches: list[tuple[BatchTranslationRequest, int]] = []
    current_items = []
    current_tokens = overhead_tokens

    def flush() -> None:
        nonlocal current_items, current_tokens
        if not current_items:
            return
        chunk_request = BatchTranslationRequest(
            items=list(current_items),
            source_language=request_payload.source_language,
            target_language=request_payload.target_language,
            style_profile=request_payload.style_profile,
            glossary_terms=request_payload.glossary_terms,
            do_not_translate=request_payload.do_not_translate,
            protected_terms=request_payload.protected_terms,
            protected_term_equivalents=request_payload.protected_term_equivalents,
            forced_translations=request_payload.forced_translations,
            deen_mode=request_payload.deen_mode,
            target_language_name=request_payload.target_language_name,
            rtl=request_payload.rtl,
        )
        batches.append((chunk_request, current_tokens))
        current_items = []
        current_tokens = overhead_tokens

    for item in request_payload.items:
        item_tokens = _estimate_batch_item_tokens(item)
        proposed_tokens = current_tokens + item_tokens
        if current_items and proposed_tokens > token_limit:
            flush()
            proposed_tokens = current_tokens + item_tokens

        current_items.append(item)
        current_tokens = proposed_tokens

        if len(current_items) == 1 and current_tokens > token_limit:
            flush()

    flush()
    return batches


def _extract_chat_content(raw: dict[str, object], *, provider_label: str) -> str:
    try:
        return str(raw["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{provider_label} returned an unexpected response payload.") from exc


def _post_lmstudio_chat(
    *,
    base_url: str,
    payload: dict[str, object],
    timeout: int,
    debug_label: str,
) -> tuple[dict[str, object], float]:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request_started_at = time.perf_counter()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s request: %s", debug_label, json.dumps(payload, ensure_ascii=False, indent=2))
    with request.urlopen(req, timeout=timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - request_started_at
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s response: %s", debug_label, json.dumps(raw, ensure_ascii=False, indent=2))
    return raw, latency


def _build_lmstudio_test_payload(model: str, target_language: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional subtitle translator. Maintain meaning, tone, and brevity.",
            },
            {
                "role": "user",
                "content": (
                    f"Translate each entry into {target_language}. Return JSON with identical indices.\n"
                    f"Output MUST be entirely in {target_language}.\n"
                    "Do not mix languages.\n\n"
                    "Return STRICT JSON only in this format:\n"
                    '{\n  "translations": [\n'
                    '    {"index": 0, "text": "..."},\n'
                    '    {"index": 1, "text": "..."},\n'
                    '    {"index": 2, "text": "..."}\n'
                    "  ]\n}\n\n"
                    f"Batch:\n{json.dumps(LMSTUDIO_TEST_BATCH, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        "temperature": 0.2,
    }


def _fallback_lmstudio_test_translations() -> list[dict[str, object]]:
    return [{"index": item["index"], "text": item["text"]} for item in LMSTUDIO_TEST_BATCH]


def run_lmstudio_inference_test(
    *,
    base_url: str = "http://localhost:1234/v1",
    model: str = "qwen2.5-7b-instruct",
    target_language: str = "Spanish",
    timeout: int = 60,
) -> dict[str, object]:
    payload = _build_lmstudio_test_payload(model, target_language)
    last_error: str | None = None

    for attempt in range(1, 3):
        try:
            raw, latency = _post_lmstudio_chat(
                base_url=base_url,
                payload=payload,
                timeout=timeout,
                debug_label="LM Studio inference test",
            )
        except error.URLError as exc:
            logger.warning("LM Studio inference test attempt %s failed: %s", attempt, exc)
            last_error = str(exc)
            continue

        logger.info("LM Studio inference test latency: %.2fs", latency)
        try:
            content = _extract_chat_content(raw, provider_label="LM Studio")
        except RuntimeError as exc:
            last_error = str(exc)
            logger.warning("LM Studio inference test attempt %s returned an unexpected payload.", attempt)
            continue

        parsed = parse_batch_translation_payload(
            content,
            expected_indices=[int(item["index"]) for item in LMSTUDIO_TEST_BATCH],
        )
        output_items = [
            {"index": item["index"], "text": text}
            for item, text in zip(LMSTUDIO_TEST_BATCH, parsed.texts, strict=True)
        ]
        translated_texts = [str(entry["text"]) for entry in output_items]
        indices_match = [entry["index"] for entry in output_items] == [
            int(item["index"]) for item in LMSTUDIO_TEST_BATCH
        ]
        language_match = _looks_like_target_language(translated_texts, target_language)
        is_valid = (
            len(output_items) == len(LMSTUDIO_TEST_BATCH)
            and indices_match
            and language_match
            and not parsed.missing_indices
            and not parsed.extra_indices
            and not parsed.duplicate_indices
            and not parsed.invalid_entries
        )
        if is_valid:
            parsed_json_result = {"translations": output_items}
            return {
                "ok": True,
                "device": "GPU (LM Studio)",
                "model": model,
                "base_url": base_url.rstrip("/"),
                "latency_seconds": latency,
                "attempts": attempt,
                "target_language": target_language,
                "validation_passed": True,
                "retry_needed": attempt > 1,
                "fallback_used": False,
                "raw_response_content": content,
                "parsed_json_result": parsed_json_result,
                "translations": output_items,
                "structured_output": parsed.metadata(),
            }

        last_error = "invalid structured output"
        if not language_match:
            last_error = f"output did not appear to be entirely in {target_language}"
        logger.warning(
            "LM Studio inference test attempt %s returned invalid output: %s",
            attempt,
            parsed.metadata(),
        )

    fallback = _fallback_lmstudio_test_translations()
    logger.warning(
        "LM Studio inference test failed validation after retry; falling back to original text."
    )
    return {
        "ok": False,
        "device": "GPU (LM Studio)",
        "model": model,
        "base_url": base_url.rstrip("/"),
        "latency_seconds": None,
        "attempts": 2,
        "target_language": target_language,
        "validation_passed": False,
        "retry_needed": True,
        "fallback_used": True,
        "raw_response_content": None,
        "parsed_json_result": {"translations": fallback},
        "translations": fallback,
        "error": last_error,
    }


def _build_lmstudio_batch_payload(model: str, request_payload: BatchTranslationRequest) -> dict[str, object]:
    target_language = request_payload.target_language_name or request_payload.target_language
    glossary_text = "\n".join(
        f"- {source} => {target}"
        for source, target in request_payload.glossary_terms.items()
    )
    batch_payload = [
        {
            "index": item.index,
            "previous_subtitle_text": item.previous_subtitle_text,
            "subtitle_text": item.source_subtitle_text,
            "next_subtitle_text": item.next_subtitle_text,
            "aligned_script_excerpt": item.script_context,
        }
        for item in request_payload.items
    ]
    user_prompt = (
        f"Translate each entry into {target_language}. Return JSON with identical indices.\n"
        f"Output MUST be entirely in {target_language}.\n"
        "Do not mix languages.\n\n"
        f"Source language: {request_payload.source_language}\n"
        f"Target language code: {request_payload.target_language}\n"
        f"Style profile: {request_payload.style_profile}\n"
        f"RTL language: {request_payload.rtl}\n"
        f"Do not translate: {', '.join(request_payload.do_not_translate) or 'None'}\n"
        f"Protected terms: {', '.join(request_payload.protected_terms) or 'None'}\n"
        "Glossary rules:\n"
        f"{glossary_text or '- None'}\n\n"
        "Do NOT translate protected religious terms. Keep them as-is.\n"
        "Preserve exact spelling where possible for protected terms and standard transliterations.\n"
        "Each input block must return exactly one output with the same index.\n"
        "Preserve original order.\n"
        "Prefer the aligned script excerpt where it clearly resolves subtitle transcription mistakes.\n"
        "Keep names and protected terms conservative and consistent.\n"
        "Return STRICT JSON only in this format:\n"
        '{\n  "translations": [\n'
        '    {"index": 0, "text": "..."},\n'
        '    {"index": 1, "text": "..."}\n'
        "  ]\n}\n\n"
        f"Batch:\n{json.dumps(batch_payload, ensure_ascii=False, indent=2)}"
    )
    system_prompt = "You are a professional subtitle translator. Maintain meaning, tone, and brevity."
    if request_payload.deen_mode:
        system_prompt += (
            " Translate faithfully and conservatively."
            " Do not reinterpret, summarize, or modernize meanings."
            " Preserve original intent and terminology."
            " Maintain respectful tone for religious figures."
            " Preserve honorifics (AS, RA, SAW or equivalents)."
        )
        if request_payload.protected_terms:
            protected_term_lines = "\n".join(f" - {term}" for term in request_payload.protected_terms)
            system_prompt += (
                " The following terms MUST NOT be translated under any circumstance:\n"
                f"{protected_term_lines}\n"
                " These terms must be preserved exactly or using standard transliteration."
                " Do NOT replace them with translated equivalents in the target language."
            )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }


def _build_stricter_deen_payload(model: str, request_payload: BatchTranslationRequest) -> dict[str, object]:
    payload = _build_lmstudio_batch_payload(model, request_payload)
    stricter_user = str(payload["messages"][1]["content"])
    stricter_user += "\n\nDo NOT add or remove meaning. Literal faithfulness required."
    payload["messages"][1]["content"] = stricter_user
    return payload


def _normalize_term_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    lowered = without_marks.lower().replace("’", "").replace("'", "").replace("`", "")
    return re.sub(r"[^a-z0-9]+", "", lowered)


def _tokenize_text_for_match(text: str) -> list[tuple[str, int, int, str]]:
    tokens: list[tuple[str, int, int, str]] = []
    for match in re.finditer(r"[^\W_]+(?:[’'`-][^\W_]+)*", str(text or ""), flags=re.UNICODE):
        raw_segment = match.group(0)
        normalized = _normalize_term_for_match(raw_segment)
        if normalized:
            tokens.append((normalized, match.start(), match.end(), raw_segment))
    return tokens


def _phrase_tokens(text: str) -> list[str]:
    return [normalized for normalized, _start, _end, _raw in _tokenize_text_for_match(text)]


def _find_phrase_matches(text: str, phrase: str) -> list[dict[str, object]]:
    phrase_tokens = _phrase_tokens(phrase)
    if not phrase_tokens:
        return []

    text_tokens = _tokenize_text_for_match(text)
    matches: list[dict[str, object]] = []
    window_size = len(phrase_tokens)
    for start_index in range(0, len(text_tokens) - window_size + 1):
        candidate_tokens = [
            normalized
            for normalized, _start, _end, _raw in text_tokens[start_index : start_index + window_size]
        ]
        if candidate_tokens == phrase_tokens:
            start = text_tokens[start_index][1]
            end = text_tokens[start_index + window_size - 1][2]
            matches.append({"start": start, "end": end, "segment": text[start:end]})
    return matches


def _replace_spans(text: str, spans: list[tuple[int, int]], replacement: str) -> str:
    updated = text
    for start, end in reversed(spans):
        updated = f"{updated[:start]}{replacement}{updated[end:]}"
    return updated


def _pluralize_phrase(phrase: str) -> str:
    parts = str(phrase or "").split()
    if not parts:
        return str(phrase or "")
    if parts[-1].lower().endswith("s"):
        return " ".join(parts)
    parts[-1] = f"{parts[-1]}s"
    return " ".join(parts)


def _is_capitalized(segment: str) -> bool:
    for character in str(segment or "").strip():
        if character.isalpha():
            return character.isupper()
    return False


def _apply_source_casing(source_segment: str, replacement: str) -> str:
    if not replacement:
        return replacement
    alpha_index = next((index for index, character in enumerate(replacement) if character.isalpha()), None)
    if alpha_index is None:
        return replacement
    transformed = list(replacement)
    transformed[alpha_index] = (
        transformed[alpha_index].upper() if _is_capitalized(source_segment) else transformed[alpha_index].lower()
    )
    return "".join(transformed)


def _select_source_term_forms(
    canonical_term: str,
    language_map: dict[str, dict[str, str]],
    source_language: str,
) -> tuple[str, str]:
    source_forms = dict(language_map.get(source_language, {}))
    singular = str(source_forms.get("singular") or canonical_term.replace("_", " ").strip()).strip()
    plural = str(source_forms.get("plural") or _pluralize_phrase(singular)).strip()
    return singular, plural


def _select_target_term(
    language_map: dict[str, dict[str, str]],
    target_language: str,
    *,
    is_plural: bool,
) -> str:
    target_forms = dict(language_map.get(target_language, {}))
    desired_form = "plural" if is_plural else "singular"
    return str(target_forms.get(desired_form) or target_forms.get("singular", "")).strip()


def _candidate_replacement_terms(
    canonical_term: str,
    language_map: dict[str, dict[str, str]],
    source_singular: str,
    source_plural: str,
    *,
    is_plural: bool,
    target_term: str,
) -> list[str]:
    desired_form = "plural" if is_plural else "singular"
    raw_candidates = [canonical_term.replace("_", " "), source_singular, source_plural]
    for forms in language_map.values():
        raw_candidates.extend(
            [
                forms.get(desired_form, ""),
                forms.get("singular", ""),
                forms.get("plural", ""),
            ]
        )

    candidate_terms: list[str] = []
    for candidate in raw_candidates:
        cleaned_candidate = str(candidate or "").strip()
        if not cleaned_candidate or cleaned_candidate == target_term or cleaned_candidate in candidate_terms:
            continue
        candidate_terms.append(cleaned_candidate)
    return candidate_terms


def _apply_forced_translations(
    request_payload: BatchTranslationRequest,
    translated_texts: list[str],
) -> list[str]:
    if not request_payload.forced_translations:
        return translated_texts

    protected_variants = {
        _normalize_term_for_match(variant)
        for variants in request_payload.protected_term_equivalents.values()
        for variant in variants
    }
    protected_variants.update(
        _normalize_term_for_match(term)
        for term in request_payload.protected_terms
        if str(term).strip()
    )

    source_language = str(request_payload.source_language).strip().lower()
    target_language = str(request_payload.target_language).strip().lower()
    updated_texts: list[str] = []

    for item, translated_text in zip(request_payload.items, translated_texts, strict=True):
        updated_text = translated_text
        source_text = item.source_subtitle_text
        for canonical_term, language_map in request_payload.forced_translations.items():
            source_singular, source_plural = _select_source_term_forms(canonical_term, language_map, source_language)
            plural_matches = _find_phrase_matches(source_text, source_plural)
            singular_matches = _find_phrase_matches(source_text, source_singular)
            if not plural_matches and not singular_matches:
                continue

            is_plural = bool(plural_matches)
            source_match = plural_matches[0] if plural_matches else singular_matches[0]
            target_term = _select_target_term(language_map, target_language, is_plural=is_plural)
            if not target_term:
                continue

            normalized_values = {
                _normalize_term_for_match(canonical_term),
                _normalize_term_for_match(source_singular),
                _normalize_term_for_match(source_plural),
                _normalize_term_for_match(target_term),
            }
            if any(value in protected_variants for value in normalized_values if value):
                continue

            logger.info(
                "Forced translation source match for '%s' in block %s: plural=%s target='%s'",
                canonical_term,
                item.index,
                is_plural,
                target_term,
            )

            candidate_terms = _candidate_replacement_terms(
                canonical_term,
                language_map,
                source_singular,
                source_plural,
                is_plural=is_plural,
                target_term=target_term,
            )
            replacement_value = _apply_source_casing(str(source_match["segment"]), target_term)
            for candidate_term in candidate_terms:
                candidate_matches = _find_phrase_matches(updated_text, candidate_term)
                if not candidate_matches:
                    continue
                updated_text = _replace_spans(
                    updated_text,
                    [(int(match["start"]), int(match["end"])) for match in candidate_matches],
                    replacement_value,
                )
                logger.info(
                    "Applied forced translation '%s' -> '%s' for block %s (%s, plural=%s)",
                    candidate_term,
                    replacement_value,
                    item.index,
                    target_language,
                    is_plural,
                )
        updated_texts.append(updated_text)
    return updated_texts


def _deen_validation_issues(
    request_payload: BatchTranslationRequest,
    translated_texts: list[str],
) -> list[str]:
    issues: list[str] = []
    normalized_equivalents = {
        canonical_term: {
            _normalize_term_for_match(variant)
            for variant in variants
        }
        for canonical_term, variants in request_payload.protected_term_equivalents.items()
    }
    for item, translated_text in zip(request_payload.items, translated_texts, strict=True):
        source_text = str(item.source_subtitle_text)
        source_words = max(1, len(source_text.split()))
        translated_words = len(str(translated_text).split())
        if source_words >= 6 and translated_words < max(3, int(source_words * 0.55)):
            issues.append(f"possible summarization in block {item.index}")
        if (
            "(" in translated_text
            or "[" in translated_text
            or any(marker in translated_text.lower() for marker in EXPLANATION_MARKERS)
        ) and "(" not in source_text and "[" not in source_text:
            issues.append(f"possible added explanation in block {item.index}")
        normalized_translation = _normalize_term_for_match(translated_text)
        normalized_source = _normalize_term_for_match(source_text)
        for canonical_term, allowed_variants in normalized_equivalents.items():
            if any(variant in normalized_source for variant in allowed_variants):
                if not any(variant in normalized_translation for variant in allowed_variants):
                    issues.append(
                        f"protected terminology was not preserved for '{canonical_term}' in block {item.index}"
                    )
    return issues


class LMStudioTranslationProvider(TranslationProvider):
    """OpenAI-compatible local provider backed by LM Studio."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        configured = base_url or "http://localhost:1234/v1"
        self.base_url = configured.rstrip("/")
        self.model = model
        self.device = "GPU (LM Studio)"
        self.precision = "fp16"

    def _translate_chunk(
        self,
        request_payload: BatchTranslationRequest,
        *,
        estimated_tokens: int | None = None,
    ) -> list[TranslationResult]:
        target_language = request_payload.target_language_name or request_payload.target_language
        last_error: str | None = None
        batch_size = len(request_payload.items)
        token_estimate = estimated_tokens if estimated_tokens is not None else _estimate_batch_tokens(request_payload)
        if request_payload.deen_mode:
            logger.info(
                "LM Studio deen_mode active for batch size=%s est_tokens=%s",
                batch_size,
                token_estimate,
            )
        stricter_deen_retry = False

        for attempt in range(1, 3):
            payload = (
                _build_stricter_deen_payload(self.model, request_payload)
                if stricter_deen_retry
                else _build_lmstudio_batch_payload(self.model, request_payload)
            )
            try:
                raw, latency = _post_lmstudio_chat(
                    base_url=self.base_url,
                    payload=payload,
                    timeout=180,
                    debug_label=f"LM Studio batch size {batch_size} est_tokens {token_estimate}",
                )
            except error.URLError as exc:
                last_error = str(exc)
                logger.warning(
                    "LM Studio batch request failed for size=%s est_tokens=%s attempt=%s/2 retry=%s: %s",
                    batch_size,
                    token_estimate,
                    attempt,
                    attempt > 1,
                    exc,
                )
                continue

            logger.info(
                "LM Studio batch size=%s est_tokens=%s latency=%.2fs retry=%s",
                batch_size,
                token_estimate,
                latency,
                attempt > 1,
            )
            try:
                content = _extract_chat_content(raw, provider_label="LM Studio")
            except RuntimeError as exc:
                last_error = str(exc)
                logger.warning(
                    "LM Studio batch size=%s est_tokens=%s attempt=%s returned an unexpected payload.",
                    batch_size,
                    token_estimate,
                    attempt,
                )
                continue

            parsed = parse_batch_translation_payload(
                content,
                expected_indices=[item.index for item in request_payload.items],
            )
            translated_texts = [str(text) for text in parsed.texts]
            count_match = (
                len(parsed.texts) == len(request_payload.items)
                and not parsed.missing_indices
                and not parsed.extra_indices
                and not parsed.duplicate_indices
                and not parsed.invalid_entries
            )
            language_match = _looks_like_target_language(translated_texts, target_language)
            deen_issues = (
                _deen_validation_issues(request_payload, translated_texts)
                if request_payload.deen_mode and count_match and language_match
                else []
            )
            if count_match and language_match and not deen_issues:
                post_processed_texts = _apply_forced_translations(request_payload, translated_texts)
                shared_metadata = {
                    "provider": "lmstudio",
                    "model": self.model,
                    "base_url": self.base_url,
                    "device": self.device,
                    "precision": self.precision,
                    "batch_size": batch_size,
                    "estimated_tokens": token_estimate,
                    "latency_seconds": latency,
                    "attempts": attempt,
                    "retry_used": attempt > 1,
                    "deen_mode": request_payload.deen_mode,
                    "structured_output": parsed.metadata(),
                }
                return [
                    TranslationResult(
                        translated_text=text,
                        confidence=0.75,
                        notes=[],
                        provider_metadata=shared_metadata,
                    )
                    for text in post_processed_texts
                ]

            last_error = "output count mismatch"
            if not language_match:
                last_error = f"output did not appear to be entirely in {target_language}"
            if deen_issues:
                last_error = "; ".join(deen_issues)
                if not stricter_deen_retry and attempt < 2:
                    stricter_deen_retry = True
                    logger.warning(
                        "LM Studio deen_mode stricter retry triggered for batch size=%s est_tokens=%s: %s",
                        batch_size,
                        token_estimate,
                        last_error,
                    )
            logger.warning(
                "LM Studio batch size=%s est_tokens=%s attempt=%s invalid output retry=%s metadata=%s",
                batch_size,
                token_estimate,
                attempt,
                attempt < 2,
                parsed.metadata(),
            )

        logger.warning(
            "LM Studio batch size=%s est_tokens=%s falling back to source text after retry: %s",
            batch_size,
            token_estimate,
            last_error or "unknown error",
        )
        return [
            TranslationResult(
                translated_text=item.source_subtitle_text,
                confidence=0.0,
                notes=["LM Studio batch translation failed validation after retry; source text was preserved."],
                provider_metadata={
                    "provider": "fallback",
                    "upstream_provider": "lmstudio",
                    "model": self.model,
                    "base_url": self.base_url,
                    "device": self.device,
                    "precision": self.precision,
                    "batch_size": batch_size,
                    "estimated_tokens": token_estimate,
                    "attempts": 2,
                    "retry_used": True,
                    "deen_mode": request_payload.deen_mode,
                    "error": last_error,
                },
            )
            for item in request_payload.items
        ]

    def translate_batch(self, request_payload: BatchTranslationRequest) -> list[TranslationResult]:
        aggregated: list[TranslationResult] = []
        token_aware_batches = _build_token_aware_batches(
            request_payload,
            token_limit=LMSTUDIO_SAFE_BATCH_TOKEN_LIMIT,
        )
        for chunk_request, estimated_tokens in token_aware_batches:
            logger.info(
                "LM Studio selected batch size=%s est_tokens=%s token_limit=%s",
                len(chunk_request.items),
                estimated_tokens,
                LMSTUDIO_SAFE_BATCH_TOKEN_LIMIT,
            )
            aggregated.extend(
                self._translate_chunk(
                    chunk_request,
                    estimated_tokens=estimated_tokens,
                )
            )
        return aggregated
