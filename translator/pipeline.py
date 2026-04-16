from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable

from parsers.semantic_alignment import align_subtitles_to_script
from parsers.script_parser import parse_script
from parsers.srt_parser import build_srt_blocks, parse_srt
from translator.config import AppConfig
from translator.factory import build_provider
from translator.glossary import load_glossary, normalize_forced_translations, normalize_protected_terms
from translator.memory import TranslationMemory
from translator.models import (
    AlignmentResult,
    BatchTranslationItem,
    BatchTranslationRequest,
    LanguageConfig,
    LanguageArtifacts,
    SubtitleBlock,
    TranslationResult,
    VerificationIssue,
    VerificationReport,
)
from translator.reporting import ensure_output_dir, write_srt
from translator.text import clean_translated_text, is_rtl_language, normalize_text, rebalance_subtitle_lines
from translator.providers.manual_provider import ManualTranslationProvider
from translator.providers.lmstudio_provider import run_lmstudio_inference_test
from verifier.validation import validate_and_repair_translation

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str], None]
DebugMappingCallback = Callable[[str, int, str, str], None]
RuntimeInfoCallback = Callable[[str, str], None]
BatchMetricsCallback = Callable[[str, int, int, int, float], None]
PerformanceSummaryCallback = Callable[[float, float, int], None]
LANGUAGE_LABELS = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "tr": "Turkish",
    "ur": "Urdu",
}


def run_lmstudio_provider_test(config: AppConfig | None = None) -> dict[str, object]:
    provider_settings = config.provider_settings("lmstudio") if config is not None else {}
    base_url = str(provider_settings.get("base_url", "http://127.0.0.1:1234/v1"))
    model = str(provider_settings.get("model", "qwen2.5-7b-instruct"))
    return run_lmstudio_inference_test(base_url=base_url, model=model)


def translate_project(
    srt_path: str,
    script_path: str,
    langs: list[str],
    config: AppConfig,
    glossary_path: str | None = None,
    profile: str | None = None,
    review_mode: bool = False,
    progress_callback: ProgressCallback | None = None,
    subtitle_limit: int | None = None,
    debug_mapping_callback: DebugMappingCallback | None = None,
    debug_performance: bool = False,
    runtime_info_callback: RuntimeInfoCallback | None = None,
    batch_metrics_callback: BatchMetricsCallback | None = None,
    performance_summary_callback: PerformanceSummaryCallback | None = None,
) -> dict[str, Path]:
    artifacts = translate_project_with_artifacts(
        srt_path=srt_path,
        script_path=script_path,
        langs=langs,
        config=config,
        glossary_path=glossary_path,
        profile=profile,
        review_mode=review_mode,
        progress_callback=progress_callback,
        subtitle_limit=subtitle_limit,
        debug_mapping_callback=debug_mapping_callback,
        debug_performance=debug_performance,
        runtime_info_callback=runtime_info_callback,
        batch_metrics_callback=batch_metrics_callback,
        performance_summary_callback=performance_summary_callback,
    )
    return {lang: result.srt_path for lang, result in artifacts.items()}


def translate_project_with_artifacts(
    srt_path: str,
    script_path: str,
    langs: list[str],
    config: AppConfig,
    glossary_path: str | None = None,
    profile: str | None = None,
    review_mode: bool = False,
    progress_callback: ProgressCallback | None = None,
    subtitle_limit: int | None = None,
    debug_mapping_callback: DebugMappingCallback | None = None,
    debug_performance: bool = False,
    runtime_info_callback: RuntimeInfoCallback | None = None,
    batch_metrics_callback: BatchMetricsCallback | None = None,
    performance_summary_callback: PerformanceSummaryCallback | None = None,
) -> dict[str, LanguageArtifacts]:
    started_at = time.perf_counter()
    source_blocks = parse_srt(srt_path)
    original_block_count = len(source_blocks)
    if subtitle_limit is not None:
        if subtitle_limit <= 0:
            raise ValueError("subtitle_limit must be greater than 0 when provided.")
        source_blocks = source_blocks[:subtitle_limit]
    batch_ranges = _window_ranges(len(source_blocks), config.translation_batch_size)
    total_steps = max(1, 3 + len(langs) * (len(batch_ranges) + 3))
    current_step = 0

    def advance(message: str) -> None:
        nonlocal current_step
        current_step += 1
        if progress_callback is not None:
            progress_callback(current_step, total_steps, message)

    if subtitle_limit is not None and subtitle_limit < original_block_count:
        advance(
            f"Loaded {len(source_blocks)} of {original_block_count} subtitle blocks (limited run)"
        )
    else:
        advance(f"Loaded {len(source_blocks)} subtitle blocks")
    script = parse_script(script_path)
    advance(f"Parsed script from {Path(script_path).name}")
    alignments = align_subtitles_to_script(source_blocks, script)
    advance("Aligned subtitles to script context")
    glossary = load_glossary(glossary_path or config.raw.get("glossary", {}).get("default_path"))
    config_glossary = config.raw.get("glossary", {})
    config_protected_terms, config_protected_equivalents = normalize_protected_terms(
        config_glossary.get("protected_terms", {})
    )
    glossary["protected_term_equivalents"] = {
        **dict(glossary.get("protected_term_equivalents", {})),
        **config_protected_equivalents,
    }
    glossary["protected_terms"] = list(
        dict.fromkeys(list(glossary.get("protected_terms", [])) + config_protected_terms)
    )
    glossary["do_not_translate"] = list(dict.fromkeys(
        list(glossary.get("do_not_translate", []))
        + [str(value).strip() for value in config_glossary.get("do_not_translate", []) if str(value).strip()]
    ))
    glossary["forced_translations"] = {
        **dict(glossary.get("forced_translations", {})),
        **normalize_forced_translations(config_glossary.get("forced_translations", {})),
    }
    provider = _build_provider_with_fallback(config)
    device = str(getattr(provider, "device", "CPU"))
    precision = str(getattr(provider, "precision", "fp32"))
    logger.info("Running on %s", device)
    logger.info("Translation runtime precision: %s", precision)
    if runtime_info_callback is not None:
        runtime_info_callback(device, precision)

    output_dir = config.output_dir
    ensure_output_dir(output_dir)
    srt_stem = Path(srt_path).stem

    batch_timings: list[float] = []
    results: dict[str, LanguageArtifacts] = {}
    for lang in langs:
        logger.info("Translating language %s", lang)
        language_start_step = current_step
        language_config = config.language_config(lang)

        def language_progress(current: int, _total: int, message: str, lang: str = lang) -> None:
            if progress_callback is not None:
                progress_callback(language_start_step + current, total_steps, f"{lang.upper()}: {message}")

        translated_blocks, translations, fallback_block_indices = _translate_language(
            source_blocks=source_blocks,
            alignments=alignments,
            lang=lang,
            provider=provider,
            config=config,
            glossary=glossary,
            profile=profile or language_config.profile or config.style_profile,
            language_config=language_config,
            batch_ranges=batch_ranges,
            progress_callback=language_progress,
            batch_timings=batch_timings,
            debug_mapping_callback=debug_mapping_callback,
            batch_metrics_callback=batch_metrics_callback,
        )
        current_step += len(batch_ranges)
        validation = validate_and_repair_translation(
            language=lang,
            source_blocks=source_blocks,
            translated_blocks=translated_blocks,
            alignments=alignments,
            translations=translations,
            allowed_source_leftovers=list(
                config.raw.get("translation", {}).get("allow_source_language_leftovers", [])
            )
            + list(glossary.get("do_not_translate", [])),
            glossary_terms=dict(glossary.get("terms", {})),
            protected_terms=list(glossary.get("protected_terms", []))
            + list(glossary.get("do_not_translate", []))
            + list(config.raw.get("glossary", {}).get("protected_terms", [])),
            rtl=language_config.rtl,
            max_chars_per_line=config.max_chars_per_line,
            max_lines_per_subtitle=config.max_lines_per_subtitle,
        )
        advance(f"{lang.upper()}: validation complete")
        corrected_blocks = validation.corrected_blocks
        report = _augment_report_with_fallbacks(validation.report, fallback_block_indices)
        file_stem = _build_output_stem(srt_stem, lang)
        output_srt = output_dir / f"{file_stem}.srt"
        write_srt(output_srt, corrected_blocks)
        advance(f"{lang.upper()}: wrote SRT output")

        results[lang] = LanguageArtifacts(
            language=lang,
            srt_path=output_srt,
            report=report,
        )
        advance(f"{lang.upper()}: ready")
    if debug_performance:
        total_runtime = time.perf_counter() - started_at
        average_batch_time = sum(batch_timings) / len(batch_timings) if batch_timings else 0.0
        logger.info("Total runtime: %.2fs", total_runtime)
        logger.info("Average time per batch: %.2fs", average_batch_time)
        if performance_summary_callback is not None:
            performance_summary_callback(
                total_runtime,
                average_batch_time,
                len(source_blocks) * len(langs),
            )
    elif performance_summary_callback is not None:
        performance_summary_callback(
            time.perf_counter() - started_at,
            sum(batch_timings) / len(batch_timings) if batch_timings else 0.0,
            len(source_blocks) * len(langs),
        )
    return results


def _translate_language(
    source_blocks: list[SubtitleBlock],
    alignments: list[AlignmentResult],
    lang: str,
    provider,
    config: AppConfig,
    glossary: dict[str, object],
    profile: str,
    language_config: LanguageConfig,
    batch_ranges: list[tuple[int, int]] | None = None,
    progress_callback: ProgressCallback | None = None,
    batch_timings: list[float] | None = None,
    debug_mapping_callback: DebugMappingCallback | None = None,
    batch_metrics_callback: BatchMetricsCallback | None = None,
) -> tuple[list[SubtitleBlock], list[TranslationResult], list[int]]:
    translated_line_sets: list[list[str]] = []
    translation_results: list[TranslationResult] = []
    fallback_block_indices: list[int] = []
    translation_memory = TranslationMemory()
    glossary_terms = dict(glossary.get("terms", {}))
    do_not_translate = list(glossary.get("do_not_translate", []))
    protected_terms = list(glossary.get("protected_terms", [])) + do_not_translate
    protected_term_equivalents = dict(glossary.get("protected_term_equivalents", {}))
    forced_translations = dict(glossary.get("forced_translations", {}))
    rtl = language_config.rtl
    batch_windows = batch_ranges or _window_ranges(len(source_blocks), config.translation_batch_size)
    for window_number, (start, end) in enumerate(batch_windows, start=1):
        batch_started_at = time.perf_counter()
        batch_results = _translate_batch_window(
            source_blocks=source_blocks,
            alignments=alignments,
            start=start,
            end=end,
            provider=provider,
            config=config,
            lang=lang,
            profile=profile,
            glossary_terms=glossary_terms,
            do_not_translate=do_not_translate,
            protected_terms=protected_terms,
            protected_term_equivalents=protected_term_equivalents,
            forced_translations=forced_translations,
            rtl=rtl,
            target_language_name=language_config.label,
            translation_memory=translation_memory,
        )
        for block, result in zip(source_blocks[start:end], batch_results, strict=True):
            if _is_fallback_translation(result):
                fallback_block_indices.append(block.index)
            cleaned_text = clean_translated_text(
                result.translated_text,
                source_text=block.text,
                language=language_config.code,
                normalize_grammar_enabled=language_config.normalize_grammar,
            )
            if cleaned_text != result.translated_text:
                result = TranslationResult(
                    translated_text=cleaned_text,
                    confidence=result.confidence,
                    notes=[*result.notes, "Post-processing cleanup normalized the translation output."],
                    provider_metadata=result.provider_metadata,
                )
            if config.retry_low_confidence and result.confidence < config.low_confidence_threshold:
                result.notes.append("Low-confidence translation; review recommended.")
            translated_lines = (
                rebalance_subtitle_lines(
                    result.translated_text,
                    config.max_chars_per_line,
                    config.max_lines_per_subtitle,
                )
                if config.line_rebalancing_enabled
                else [line for line in result.translated_text.splitlines() if line.strip()] or [result.translated_text]
            )
            if debug_mapping_callback is not None:
                debug_mapping_callback(
                    language_config.code,
                    block.index,
                    block.text,
                    "\n".join(translated_lines),
                )
            translated_line_sets.append(translated_lines)
            translation_results.append(result)
        batch_elapsed = time.perf_counter() - batch_started_at
        if batch_timings is not None:
            batch_timings.append(batch_elapsed)
        if batch_metrics_callback is not None:
            batch_metrics_callback(
                language_config.code,
                window_number,
                len(batch_windows),
                end - start,
                batch_elapsed,
            )
        logger.info(
            "%s batch %s/%s processed %s subtitles in %.2fs",
            language_config.code.upper(),
            window_number,
            len(batch_windows),
            end - start,
            batch_elapsed,
        )
        if progress_callback is not None:
            progress_callback(
                window_number,
                len(batch_windows),
                f"translated batch {window_number}/{len(batch_windows)} ({end - start} subtitles)",
            )
    translated_blocks = build_srt_blocks(source_blocks, translated_line_sets)
    return translated_blocks, translation_results, fallback_block_indices


def _build_provider_with_fallback(config: AppConfig):
    try:
        return build_provider(config.provider, config.model, config=config)
    except Exception as exc:
        if config.provider == "manual":
            raise
        logger.warning(
            "Provider %s could not be initialized (%s). Falling back to manual provider so output files can still be generated.",
            config.provider,
            exc,
        )
        return ManualTranslationProvider()


def _window_ranges(total_blocks: int, batch_size: int) -> list[tuple[int, int]]:
    if total_blocks <= 0:
        return []
    sizes: list[int] = []
    remaining = total_blocks
    while remaining > 0:
        size = min(batch_size, remaining)
        if remaining > batch_size and remaining - size < 5:
            size = min(15, remaining)
        sizes.append(size)
        remaining -= size

    ranges: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _translate_batch_window(
    source_blocks: list[SubtitleBlock],
    alignments: list[AlignmentResult],
    start: int,
    end: int,
    provider,
    config: AppConfig,
    lang: str,
    profile: str,
    glossary_terms: dict[str, str],
    do_not_translate: list[str],
    protected_terms: list[str],
    protected_term_equivalents: dict[str, list[str]],
    forced_translations: dict[str, dict[str, dict[str, str]]],
    rtl: bool,
    target_language_name: str,
    translation_memory: TranslationMemory,
) -> list[TranslationResult]:
    window_blocks = source_blocks[start:end]
    window_alignments = alignments[start:end]
    resolved_results: list[TranslationResult | None] = [None] * len(window_blocks)
    pending_items: list[BatchTranslationItem] = []
    pending_positions: list[int] = []

    for relative_position, (block, alignment) in enumerate(zip(window_blocks, window_alignments, strict=True)):
        memory_result = translation_memory.lookup(block.text)
        if memory_result is not None:
            resolved_results[relative_position] = memory_result
            continue
        absolute_position = start + relative_position
        pending_positions.append(relative_position)
        pending_items.append(
            _build_batch_item(
                block=block,
                alignment=alignment,
                source_blocks=source_blocks,
                absolute_position=absolute_position,
            )
        )

    if pending_items:
        batch_request = BatchTranslationRequest(
            items=pending_items,
            source_language=config.source_language,
            target_language=lang,
            target_language_name=target_language_name,
            style_profile=profile,
            glossary_terms=glossary_terms,
            do_not_translate=do_not_translate,
            protected_terms=protected_terms,
            protected_term_equivalents=protected_term_equivalents,
            forced_translations=forced_translations,
            deen_mode=config.deen_mode,
            rtl=rtl,
        )
        batch_results = _attempt_batch_translation(
            provider=provider,
            batch_request=batch_request,
            max_attempts=max(1, config.max_repair_attempts + 1),
        )
        if batch_results is None:
            batch_results = [
                _translate_single_item_with_retry(
                    provider=provider,
                    batch_request=batch_request,
                    item=item,
                    max_attempts=max(1, config.max_repair_attempts + 1),
                )
                for item in pending_items
            ]
        for relative_position, item, result in zip(pending_positions, pending_items, batch_results, strict=True):
            if not _has_usable_translation(result):
                result = _translate_single_item_with_retry(
                    provider=provider,
                    batch_request=batch_request,
                    item=item,
                    max_attempts=max(1, config.max_repair_attempts + 1),
                )
            resolved_results[relative_position] = result
            translation_memory.remember(item.source_subtitle_text, result)

    finalized_results: list[TranslationResult] = []
    for block, result in zip(window_blocks, resolved_results, strict=True):
        if result is None:
            result = _fallback_translation_result(
                block.text,
                "Translation window returned no result; source text was preserved.",
                block_index=block.index,
            )
            translation_memory.remember(block.text, result)
        finalized_results.append(result)
    return finalized_results


def _build_batch_item(
    block: SubtitleBlock,
    alignment: AlignmentResult,
    source_blocks: list[SubtitleBlock],
    absolute_position: int,
) -> BatchTranslationItem:
    previous_text = source_blocks[absolute_position - 1].text if absolute_position > 0 else ""
    next_text = source_blocks[absolute_position + 1].text if absolute_position + 1 < len(source_blocks) else ""
    return BatchTranslationItem(
        index=block.index,
        source_subtitle_text=block.text,
        script_context=alignment.script_excerpt,
        previous_subtitle_text=previous_text,
        next_subtitle_text=next_text,
        metadata={
            "subtitle_index": block.index,
            "subtitle_start": block.start,
            "subtitle_end": block.end,
            "script_similarity": alignment.similarity,
            "used_script_as_truth": alignment.used_script_as_truth,
        },
    )


def _attempt_batch_translation(
    provider,
    batch_request: BatchTranslationRequest,
    max_attempts: int,
) -> list[TranslationResult] | None:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            results = provider.translate_batch(batch_request)
            _validate_batch_results(batch_request, results)
            return results
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Batch translation attempt %s/%s failed for indices %s (%s).",
                attempt,
                max_attempts,
                [item.index for item in batch_request.items],
                exc,
            )
    if last_error is not None:
        logger.warning(
            "Batch translation failed after %s attempts; retrying individual subtitle blocks.",
            max_attempts,
        )
    return None


def _translate_single_item_with_retry(
    provider,
    batch_request: BatchTranslationRequest,
    item: BatchTranslationItem,
    max_attempts: int,
) -> TranslationResult:
    single_request = BatchTranslationRequest(
        items=[item],
        source_language=batch_request.source_language,
        target_language=batch_request.target_language,
        style_profile=batch_request.style_profile,
        glossary_terms=batch_request.glossary_terms,
        do_not_translate=batch_request.do_not_translate,
        protected_terms=batch_request.protected_terms,
        protected_term_equivalents=batch_request.protected_term_equivalents,
        forced_translations=batch_request.forced_translations,
        deen_mode=batch_request.deen_mode,
        target_language_name=batch_request.target_language_name,
        rtl=batch_request.rtl,
    )
    for attempt in range(1, max_attempts + 1):
        try:
            results = provider.translate_batch(single_request)
            _validate_batch_results(single_request, results)
            if _has_usable_translation(results[0]):
                return results[0]
        except Exception as exc:
            logger.warning(
                "Single-block translation attempt %s/%s failed for block %s (%s).",
                attempt,
                max_attempts,
                item.index,
                exc,
            )
    return _fallback_translation_result(
        item.source_subtitle_text,
        "Automatic translation failed for this block after retries; source text was preserved.",
        block_index=item.index,
    )


def _validate_batch_results(
    batch_request: BatchTranslationRequest,
    results: list[TranslationResult],
) -> None:
    if len(results) != len(batch_request.items):
        raise ValueError("Batch result count did not match the request size.")
    for item, result in zip(batch_request.items, results, strict=True):
        if not isinstance(result.translated_text, str):
            raise ValueError(f"Block {item.index} returned a non-string translation.")


def _has_usable_translation(result: TranslationResult) -> bool:
    return bool(normalize_text(result.translated_text))


def _is_fallback_translation(result: TranslationResult) -> bool:
    return result.provider_metadata.get("provider") == "fallback"


def _augment_report_with_fallbacks(
    report: VerificationReport,
    fallback_block_indices: list[int],
) -> VerificationReport:
    summary = dict(report.summary)
    summary["fallback_count"] = len(fallback_block_indices)
    if fallback_block_indices:
        summary["fallback_block_indices"] = list(fallback_block_indices)

    issues = list(report.issues)
    for block_index in fallback_block_indices:
        issues.append(
            VerificationIssue(
                "medium",
                "fallback_source_text",
                block_index,
                "Automatic translation failed for this subtitle block, so the source text was kept.",
            )
        )

    return VerificationReport(
        language=report.language,
        passed=report.passed,
        issues=issues,
        summary=summary,
    )


def _fallback_translation_result(source_text: str, note: str, *, block_index: int | None = None) -> TranslationResult:
    if block_index is not None:
        logger.warning("Falling back to source text for block %s: %s", block_index, note)
    return TranslationResult(
        translated_text=source_text,
        confidence=0.0,
        notes=[note],
        provider_metadata={"provider": "fallback"},
    )


def _build_output_stem(source_stem: str, lang: str) -> str:
    language_label = LANGUAGE_LABELS.get(lang, lang.upper())
    return _sanitize_output_name(language_label)


def _sanitize_output_name(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "-", value).strip()
    return cleaned or "Translation"
