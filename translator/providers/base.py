from __future__ import annotations

from abc import ABC, abstractmethod

from translator.models import (
    BatchTranslationItem,
    BatchTranslationRequest,
    TranslationRequest,
    TranslationResult,
)


class TranslationProvider(ABC):
    def translate(self, request: TranslationRequest) -> TranslationResult:
        results = self.translate_batch(
            BatchTranslationRequest(
                items=[
                    BatchTranslationItem(
                        index=int(request.metadata.get("subtitle_index", 0)),
                        source_subtitle_text=request.source_subtitle_text,
                        script_context=request.script_context,
                        previous_subtitle_text=request.previous_subtitle_text,
                        next_subtitle_text=request.next_subtitle_text,
                        metadata=request.metadata,
                    )
                ],
                source_language=request.source_language,
                target_language=request.target_language,
                style_profile=request.style_profile,
                glossary_terms=request.glossary_terms,
                do_not_translate=request.do_not_translate,
                protected_terms=request.protected_terms,
                protected_term_equivalents=request.protected_term_equivalents,
                forced_translations=request.forced_translations,
                deen_mode=request.deen_mode,
                target_language_name=request.target_language_name,
                rtl=request.rtl,
            )
        )
        return results[0]

    @abstractmethod
    def translate_batch(self, request: BatchTranslationRequest) -> list[TranslationResult]:
        raise NotImplementedError
