from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LanguageConfig:
    code: str
    label: str
    rtl: bool = False
    profile: str = "balanced"
    aliases: list[str] = field(default_factory=list)
    normalize_grammar: bool = True


@dataclass(slots=True)
class SubtitleBlock:
    index: int
    start: str
    end: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass(slots=True)
class ScriptDocument:
    path: Path
    raw_text: str
    normalized_text: str
    segments: list[str]


@dataclass(slots=True)
class AlignmentResult:
    block_index: int
    subtitle_text: str
    script_excerpt: str
    similarity: float
    used_script_as_truth: bool


@dataclass(slots=True)
class TranslationRequest:
    source_subtitle_text: str
    script_context: str
    source_language: str
    target_language: str
    style_profile: str
    glossary_terms: dict[str, str]
    do_not_translate: list[str]
    protected_terms: list[str]
    protected_term_equivalents: dict[str, list[str]] = field(default_factory=dict)
    forced_translations: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    deen_mode: bool = False
    target_language_name: str = ""
    rtl: bool = False
    previous_subtitle_text: str = ""
    next_subtitle_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationResult:
    translated_text: str
    confidence: float
    notes: list[str] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchTranslationItem:
    index: int
    source_subtitle_text: str
    script_context: str
    previous_subtitle_text: str = ""
    next_subtitle_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchTranslationRequest:
    items: list[BatchTranslationItem]
    source_language: str
    target_language: str
    style_profile: str
    glossary_terms: dict[str, str]
    do_not_translate: list[str]
    protected_terms: list[str]
    protected_term_equivalents: dict[str, list[str]] = field(default_factory=dict)
    forced_translations: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    deen_mode: bool = False
    target_language_name: str = ""
    rtl: bool = False


@dataclass(slots=True)
class VerificationIssue:
    severity: str
    code: str
    block_index: int | None
    message: str


@dataclass(slots=True)
class VerificationReport:
    language: str
    passed: bool
    issues: list[VerificationIssue]
    summary: dict[str, Any]


@dataclass(slots=True)
class ValidationResult:
    corrected_blocks: list[SubtitleBlock]
    report: VerificationReport


@dataclass(slots=True)
class LanguageArtifacts:
    language: str
    srt_path: Path
    report: VerificationReport
