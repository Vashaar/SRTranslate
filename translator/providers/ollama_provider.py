from __future__ import annotations

import json
import os
import shutil
import subprocess
from urllib import error, request

from translator.models import BatchTranslationRequest, TranslationResult
from translator.providers.base import TranslationProvider
from translator.providers.structured import (
    TRANSLATION_JSON_CONTRACT,
    parse_batch_translation_payload,
)


def _gpu_available() -> bool:
    cuda_visible_devices = str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip()
    if cuda_visible_devices and cuda_visible_devices != "-1":
        return True
    rocm_visible_devices = str(os.getenv("ROCM_VISIBLE_DEVICES", "")).strip()
    if rocm_visible_devices and rocm_visible_devices != "-1":
        return True
    for tool_name in ("nvidia-smi", "rocm-smi"):
        tool_path = shutil.which(tool_name)
        if not tool_path:
            continue
        completed = subprocess.run(
            [tool_path],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0:
            return True
    return False


class OllamaTranslationProvider(TranslationProvider):
    """Local provider backed by an Ollama server running on localhost."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        prefer_gpu: bool = True,
        precision: str = "auto",
    ) -> None:
        configured = base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        self.base_url = configured.rstrip("/")
        self.model = model
        detected_gpu = _gpu_available()
        self.gpu_available = prefer_gpu and detected_gpu
        normalized_precision = str(precision or "auto").lower()
        if normalized_precision not in {"auto", "fp16", "fp32"}:
            normalized_precision = "auto"
        if normalized_precision == "auto":
            self.precision = "fp16" if self.gpu_available else "fp32"
        else:
            self.precision = normalized_precision

    def translate_batch(self, request_payload: BatchTranslationRequest) -> list[TranslationResult]:
        target_language = request_payload.target_language_name or request_payload.target_language
        system_prompt = (
            "You are an expert subtitle translator for religious, historical, and nuanced lectures. "
            "Preserve meaning, rhetoric, theological precision, and argument structure. "
            "Be conservative with religious names and terms: if uncertain, preserve or transliterate rather than inventing. "
            "Use the aligned script excerpt as the primary source of truth when it clearly corrects subtitle transcription mistakes. "
            f"{TRANSLATION_JSON_CONTRACT} "
            "Do not add commentary or extra keys."
        )
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
        prompt = f"""
Source language: {request_payload.source_language}
Target language: {target_language} ({request_payload.target_language})
Style profile: {request_payload.style_profile}
RTL language: {request_payload.rtl}
Do not translate: {", ".join(request_payload.do_not_translate) or "None"}
Protected terms: {", ".join(request_payload.protected_terms) or "None"}
Glossary rules:
{glossary_text or "- None"}

Instructions:
- Translate every subtitle block in order.
- Prefer the script excerpt where it clearly fixes transcription errors.
- Keep protected names and sacred terms conservative and consistent.
- Never merge, split, or reorder entries.

Batch:
{json.dumps(batch_payload, ensure_ascii=False, indent=2)}
"""
        payload = {
            "model": self.model,
            "format": "json",
            "stream": False,
            "system": system_prompt,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,
                "f16_kv": self.precision == "fp16",
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach the local Ollama server. Start Ollama or switch providers."
            ) from exc

        parsed = parse_batch_translation_payload(
            str(raw.get("response", "")).strip(),
            expected_indices=[item.index for item in request_payload.items],
        )
        shared_metadata = {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "batch_size": len(request_payload.items),
            "gpu_enabled": self.gpu_available,
            "precision": self.precision,
            "structured_output": parsed.metadata(),
        }
        results: list[TranslationResult] = []
        for item, text in zip(request_payload.items, parsed.texts, strict=True):
            notes: list[str] = []
            confidence = 0.65
            if item.index in parsed.missing_indices:
                notes.append("Missing translation entry was detected in the batch output.")
                confidence = 0.0
            if item.index in parsed.duplicate_indices:
                notes.append("Duplicate translation entry was detected and repaired by index.")
            results.append(
                TranslationResult(
                    translated_text=text,
                    confidence=confidence,
                    notes=notes,
                    provider_metadata=shared_metadata,
                )
            )
        return results
