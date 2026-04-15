from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import yaml


APP_NAME = "SRTranslate"
MANIFEST_FILENAME = "dictionary_manifest.json"
FREEDICT_GENERATED_INDEX_URL = "https://download.freedict.org/generated/"
LANGUAGE_CODE_MAP: dict[str, tuple[str, str]] = {
    "ar": ("Arabic", "ara"),
    "bn": ("Bengali", "ben"),
    "de": ("German", "deu"),
    "es": ("Spanish", "spa"),
    "fa": ("Persian", "fas"),
    "fr": ("French", "fra"),
    "id": ("Indonesian", "ind"),
    "ms": ("Malay", "msa"),
    "tr": ("Turkish", "tur"),
    "ur": ("Urdu", "urd"),
}


@dataclass(slots=True)
class StoredDictionary:
    name: str
    source_url: str | None
    filename: str
    downloaded_at: str
    original_format: str

    @property
    def display_name(self) -> str:
        return self.name


@dataclass(slots=True)
class LanguageDatasetOption:
    source_code: str
    target_code: str
    source_label: str
    target_label: str
    source_iso3: str
    target_iso3: str
    source_url: str

    @property
    def label(self) -> str:
        return f"{self.source_label} -> {self.target_label} (FreeDict)"


def app_storage_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir else _default_storage_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def dictionaries_dir(base_dir: str | Path | None = None) -> Path:
    target = app_storage_dir(base_dir) / "dictionaries"
    target.mkdir(parents=True, exist_ok=True)
    return target


def manifest_path(base_dir: str | Path | None = None) -> Path:
    return dictionaries_dir(base_dir) / MANIFEST_FILENAME


def list_dictionaries(base_dir: str | Path | None = None) -> list[StoredDictionary]:
    path = manifest_path(base_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[StoredDictionary] = []
    for item in payload.get("dictionaries", []):
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", ""))
        if not filename:
            continue
        if not (dictionaries_dir(base_dir) / filename).exists():
            continue
        records.append(
            StoredDictionary(
                name=str(item.get("name", Path(filename).stem)),
                source_url=str(item.get("source_url")) if item.get("source_url") else None,
                filename=filename,
                downloaded_at=str(item.get("downloaded_at", "")),
                original_format=str(item.get("original_format", "yaml")),
            )
        )
    records.sort(key=lambda item: item.name.lower())
    return records


def dictionary_path(record: StoredDictionary, base_dir: str | Path | None = None) -> Path:
    return dictionaries_dir(base_dir) / record.filename


def download_dictionary(
    source_url: str,
    name: str | None = None,
    *,
    base_dir: str | Path | None = None,
    timeout_seconds: int = 30,
) -> StoredDictionary:
    try:
        with request.urlopen(source_url, timeout=timeout_seconds) as response:
            payload = response.read()
            content_type = response.headers.get_content_type()
    except error.URLError as exc:
        raise RuntimeError(f"Could not download dictionary: {exc}") from exc

    inferred_name = name or _name_from_url(source_url)
    original_format = _infer_format(source_url, content_type, payload)
    glossary = _normalize_dictionary_payload(payload, original_format)
    return _save_dictionary(
        glossary=glossary,
        name=inferred_name,
        source_url=source_url,
        original_format=original_format,
        base_dir=base_dir,
    )


def list_language_dataset_options(
    *,
    source_code: str = "en",
    timeout_seconds: int = 20,
) -> list[LanguageDatasetOption]:
    source_label = "English" if source_code == "en" else source_code
    source_iso3 = "eng" if source_code == "en" else source_code
    try:
        with request.urlopen(FREEDICT_GENERATED_INDEX_URL, timeout=timeout_seconds) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except error.URLError as exc:
        raise RuntimeError(f"Could not load free language datasets: {exc}") from exc

    options: list[LanguageDatasetOption] = []
    for code, (label, iso3) in sorted(LANGUAGE_CODE_MAP.items(), key=lambda item: item[1][0].lower()):
        pair_slug = f"{source_iso3}-{iso3}"
        if re.search(rf'href="{re.escape(pair_slug)}/"', html, flags=re.IGNORECASE) is None:
            continue
        options.append(
            LanguageDatasetOption(
                source_code=source_code,
                target_code=code,
                source_label=source_label,
                target_label=label,
                source_iso3=source_iso3,
                target_iso3=iso3,
                source_url=_freedict_dataset_url(source_iso3, iso3),
            )
        )
    return options


def download_language_dataset(
    target_code: str,
    *,
    source_code: str = "en",
    base_dir: str | Path | None = None,
    timeout_seconds: int = 90,
) -> StoredDictionary:
    options = list_language_dataset_options(source_code=source_code, timeout_seconds=timeout_seconds)
    selected = next((option for option in options if option.target_code == target_code), None)
    if selected is None:
        target_label = LANGUAGE_CODE_MAP.get(target_code, (target_code, target_code))[0]
        raise ValueError(f"No free downloadable dataset is currently available for {source_code} -> {target_label}.")
    return download_dictionary(
        selected.source_url,
        name=f"FreeDict {selected.source_label} - {selected.target_label}",
        base_dir=base_dir,
        timeout_seconds=timeout_seconds,
    )


def import_dictionary(
    dictionary_path_input: str | Path,
    name: str | None = None,
    *,
    base_dir: str | Path | None = None,
) -> StoredDictionary:
    source_path = Path(dictionary_path_input)
    payload = source_path.read_bytes()
    original_format = _infer_format(source_path.name, None, payload)
    glossary = _normalize_dictionary_payload(payload, original_format)
    return _save_dictionary(
        glossary=glossary,
        name=name or source_path.stem,
        source_url=None,
        original_format=original_format,
        base_dir=base_dir,
    )


def remove_dictionary(name: str, *, base_dir: str | Path | None = None) -> None:
    records = list_dictionaries(base_dir)
    remaining: list[StoredDictionary] = []
    for record in records:
        if record.name != name:
            remaining.append(record)
            continue
        file_path = dictionary_path(record, base_dir)
        if file_path.exists():
            file_path.unlink()
    _write_manifest(remaining, base_dir)


def _save_dictionary(
    glossary: dict[str, object],
    name: str,
    source_url: str | None,
    original_format: str,
    *,
    base_dir: str | Path | None = None,
) -> StoredDictionary:
    filename = f"{_slugify(name)}.yaml"
    path = dictionaries_dir(base_dir) / filename
    path.write_text(
        yaml.safe_dump(glossary, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    record = StoredDictionary(
        name=name,
        source_url=source_url,
        filename=filename,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        original_format=original_format,
    )

    records = [item for item in list_dictionaries(base_dir) if item.name != name]
    records.append(record)
    _write_manifest(records, base_dir)
    return record


def _write_manifest(records: list[StoredDictionary], base_dir: str | Path | None = None) -> None:
    payload = {"dictionaries": [asdict(item) for item in sorted(records, key=lambda row: row.name.lower())]}
    manifest_path(base_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_dictionary_payload(payload: bytes, original_format: str) -> dict[str, object]:
    if original_format in {"tei", "xml"}:
        return _normalize_tei_dictionary_payload(payload)
    text = payload.decode("utf-8-sig")
    if original_format in {"yaml", "yml"}:
        loaded = yaml.safe_load(text) or {}
        return _normalize_loaded_dictionary(loaded)
    if original_format == "json":
        loaded = json.loads(text)
        return _normalize_loaded_dictionary(loaded)
    if original_format in {"csv", "tsv"}:
        delimiter = "\t" if original_format == "tsv" else ","
        return _normalize_tabular_dictionary(text, delimiter)
    if original_format == "txt":
        structured = _try_normalize_structured_text(text)
        if structured is not None:
            return structured
        return _normalize_text_dictionary(text)
    raise ValueError(f"Unsupported dictionary format: {original_format}")


def _normalize_loaded_dictionary(loaded: Any) -> dict[str, object]:
    if isinstance(loaded, dict):
        if {"terms", "do_not_translate", "protected_terms"} & set(loaded):
            terms_payload = loaded.get("terms", {})
            if isinstance(terms_payload, dict):
                terms = {str(key): str(value) for key, value in terms_payload.items()}
            else:
                terms = {}
            return {
                "terms": terms,
                "do_not_translate": [str(item) for item in loaded.get("do_not_translate", [])],
                "protected_terms": [str(item) for item in loaded.get("protected_terms", [])],
            }
        return {
            "terms": {str(key): str(value) for key, value in loaded.items()},
            "do_not_translate": [],
            "protected_terms": [],
        }
    if isinstance(loaded, list):
        terms: dict[str, str] = {}
        protected_terms: list[str] = []
        do_not_translate: list[str] = []
        for item in loaded:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or item.get("term") or item.get("key")
            target = item.get("target") or item.get("translation") or item.get("value")
            if source and target:
                terms[str(source)] = str(target)
                continue
            if item.get("do_not_translate"):
                do_not_translate.append(str(item.get("do_not_translate")))
            if item.get("protected_term"):
                protected_terms.append(str(item.get("protected_term")))
        return {
            "terms": terms,
            "do_not_translate": do_not_translate,
            "protected_terms": protected_terms,
        }
    raise ValueError("Dictionary data must be a mapping, list, CSV, or simple text pairs.")


def _normalize_tabular_dictionary(text: str, delimiter: str) -> dict[str, object]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames:
        terms: dict[str, str] = {}
        do_not_translate: list[str] = []
        protected_terms: list[str] = []
        for row in reader:
            lowered = {str(key).strip().lower(): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            source = lowered.get("source") or lowered.get("term") or lowered.get("key")
            target = lowered.get("target") or lowered.get("translation") or lowered.get("value")
            if source and target:
                terms[str(source)] = str(target)
            if lowered.get("do_not_translate"):
                do_not_translate.append(str(lowered["do_not_translate"]))
            if lowered.get("protected_term"):
                protected_terms.append(str(lowered["protected_term"]))
        if terms or do_not_translate or protected_terms:
            return {
                "terms": terms,
                "do_not_translate": do_not_translate,
                "protected_terms": protected_terms,
            }

    fallback_reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    terms = {}
    for row in fallback_reader:
        if len(row) >= 2:
            source = row[0].strip()
            target = row[1].strip()
            if source and target:
                terms[source] = target
    if not terms:
        raise ValueError("Could not parse dictionary rows from the downloaded table.")
    return {"terms": terms, "do_not_translate": [], "protected_terms": []}


def _normalize_text_dictionary(text: str) -> dict[str, object]:
    terms: dict[str, str] = {}
    do_not_translate: list[str] = []
    protected_terms: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("dnt:"):
            do_not_translate.append(line.split(":", 1)[1].strip())
            continue
        if lowered.startswith("protect:"):
            protected_terms.append(line.split(":", 1)[1].strip())
            continue
        for delimiter in ("\t", "=>", "=", ","):
            if delimiter in line:
                left, right = line.split(delimiter, 1)
                source = left.strip()
                target = right.strip()
                if source and target:
                    terms[source] = target
                break
    if not terms and not do_not_translate and not protected_terms:
        raise ValueError("Could not parse any dictionary entries from the provided text.")
    return {
        "terms": terms,
        "do_not_translate": do_not_translate,
        "protected_terms": protected_terms,
    }


def _try_normalize_structured_text(text: str) -> dict[str, object] | None:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if isinstance(loaded, (dict, list)):
        try:
            return _normalize_loaded_dictionary(loaded)
        except ValueError:
            return None
    return None


def _normalize_tei_dictionary_payload(payload: bytes) -> dict[str, object]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("Could not parse TEI/XML dictionary payload.") from exc

    terms: dict[str, str] = {}
    for entry in root.findall(".//{*}entry"):
        source_terms = [
            _collapse_whitespace("".join(node.itertext()))
            for node in entry.findall(".//{*}form/{*}orth")
        ]
        translation_nodes = entry.findall(".//{*}cit[@type='trans']/{*}quote")
        if not translation_nodes:
            translation_nodes = entry.findall(".//{*}sense/{*}def")
        translations = [
            _collapse_whitespace("".join(node.itertext()))
            for node in translation_nodes
        ]
        translations = [item for item in translations if item]
        if not translations:
            continue
        best_translation = translations[0]
        for source in source_terms:
            if source and source not in terms:
                terms[source] = best_translation

    if not terms:
        raise ValueError("No bilingual glossary terms could be extracted from the downloaded TEI dictionary.")
    return {"terms": terms, "do_not_translate": [], "protected_terms": []}


def _infer_format(source_hint: str, content_type: str | None, payload: bytes) -> str:
    suffix = Path(parse.urlparse(source_hint).path).suffix.lower().lstrip(".")
    if suffix in {"yaml", "yml", "json", "csv", "tsv", "txt", "tei", "xml"}:
        return suffix
    if content_type:
        if "json" in content_type:
            return "json"
        if "csv" in content_type:
            return "csv"
        if "yaml" in content_type or "yml" in content_type:
            return "yaml"
        if "text/plain" in content_type:
            return "txt"
    sniffed = payload[:64].decode("utf-8-sig", errors="ignore")
    if sniffed.lstrip().startswith("{") or sniffed.lstrip().startswith("["):
        return "json"
    if sniffed.lstrip().startswith("<?xml") or "<TEI" in sniffed or "<tei" in sniffed:
        return "tei"
    if "," in sniffed and "\n" in sniffed:
        return "csv"
    return "txt"


def _name_from_url(source_url: str) -> str:
    parsed = parse.urlparse(source_url)
    path_name = Path(parsed.path).stem
    if path_name:
        return path_name.replace("_", " ").replace("-", " ").title()
    return "Downloaded Dictionary"


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return cleaned or "dictionary"


def _freedict_dataset_url(source_iso3: str, target_iso3: str) -> str:
    pair = f"{source_iso3}-{target_iso3}"
    return f"{FREEDICT_GENERATED_INDEX_URL}{pair}/{pair}.tei"


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _default_storage_root() -> Path:
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / APP_NAME

    if os.name == "nt":
        return Path.home() / "AppData" / "Local" / APP_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME

    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME
