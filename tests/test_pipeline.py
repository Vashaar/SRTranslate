from pathlib import Path

import yaml

from translator.models import TranslationResult
from translator.config import load_config
from translator.pipeline import translate_project


def test_pipeline_creates_only_srt_output(tmp_path: Path) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"

    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nPeace be upon you.\n",
        encoding="utf-8",
    )
    script.write_text(
        "Peace be upon you. May mercy and guidance follow.",
        encoding="utf-8",
    )
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "mock",
                "model": "mock",
                "output": {"output_dir": str(out_dir), "write_review_csv": True},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)
    outputs = translate_project(str(srt), str(script), ["ur"], loaded, review_mode=True)

    assert "ur" in outputs
    assert (out_dir / "input.ur.srt").exists()
    assert not (out_dir / "input.ur.report.json").exists()
    assert not (out_dir / "input.ur.review.csv").exists()
    assert not (out_dir / "input.ur.flags.txt").exists()


def test_pipeline_supports_manual_provider(tmp_path: Path) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"

    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nAllah is merciful.\n",
        encoding="utf-8",
    )
    script.write_text("Allah is merciful and just.", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "manual",
                "model": "manual",
                "output": {"output_dir": str(out_dir), "write_review_csv": True},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)
    outputs = translate_project(str(srt), str(script), ["ur"], loaded, review_mode=True)

    assert "ur" in outputs
    content = (out_dir / "input.ur.srt").read_text(encoding="utf-8")
    assert "Allah is merciful." in content


def test_pipeline_falls_back_when_provider_cannot_initialize(tmp_path: Path, monkeypatch) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"

    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nPeace be upon you.\n",
        encoding="utf-8",
    )
    script.write_text(
        "Peace be upon you. May mercy and guidance follow.",
        encoding="utf-8",
    )
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "ollama",
                "model": "mock-model",
                "output": {"output_dir": str(out_dir), "write_review_csv": True},
            }
        ),
        encoding="utf-8",
    )

    def broken_build_provider(provider_name: str, model: str):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr("translator.pipeline.build_provider", broken_build_provider)

    loaded = load_config(config)
    outputs = translate_project(str(srt), str(script), ["ur"], loaded, review_mode=True)

    assert "ur" in outputs
    content = (out_dir / "input.ur.srt").read_text(encoding="utf-8")
    assert "Peace be upon you." in content


def test_pipeline_falls_back_when_provider_fails_mid_translation(
    tmp_path: Path, monkeypatch
) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"

    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nPeace be upon you.\n",
        encoding="utf-8",
    )
    script.write_text(
        "Peace be upon you. May mercy and guidance follow.",
        encoding="utf-8",
    )
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "ollama",
                "model": "mock-model",
                "output": {"output_dir": str(out_dir), "write_review_csv": True},
            }
        ),
        encoding="utf-8",
    )

    class FlakyProvider:
        def translate_batch(self, request):
            raise RuntimeError("network down")

    monkeypatch.setattr("translator.pipeline.build_provider", lambda *_: FlakyProvider())

    loaded = load_config(config)
    outputs = translate_project(str(srt), str(script), ["ur"], loaded, review_mode=True)

    assert "ur" in outputs
    content = (out_dir / "input.ur.srt").read_text(encoding="utf-8")
    assert "Peace be upon you." in content


def test_pipeline_batches_translation_requests_and_preserves_order(
    tmp_path: Path, monkeypatch
) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"

    blocks = []
    script_lines = []
    for index in range(1, 13):
        blocks.append(
            f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\nLine {index}.\n"
        )
        script_lines.append(f"Line {index}.")
    srt.write_text("\n".join(blocks), encoding="utf-8")
    script.write_text(" ".join(script_lines), encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "mock",
                "model": "mock",
                "translation": {"batch_size": 5},
                "output": {"output_dir": str(out_dir), "write_review_csv": True},
            }
        ),
        encoding="utf-8",
    )

    class RecordingProvider:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def translate_batch(self, request):
            self.batch_sizes.append(len(request.items))
            return [
                TranslationResult(
                    translated_text=f"translated-{item.index}",
                    confidence=0.95,
                    notes=[],
                    provider_metadata={"provider": "recording"},
                )
                for item in request.items
            ]

    provider = RecordingProvider()
    monkeypatch.setattr("translator.pipeline.build_provider", lambda *_: provider)

    loaded = load_config(config)
    outputs = translate_project(str(srt), str(script), ["ur"], loaded, review_mode=True)

    assert provider.batch_sizes == [5, 7]
    content = (out_dir / "input.ur.srt").read_text(encoding="utf-8")
    assert "1\n00:00:01,000 --> 00:00:02,000\ntranslated-1" in content
    assert "12\n00:00:12,000 --> 00:00:13,000\ntranslated-12" in content


def test_pipeline_reports_progress_updates(tmp_path: Path) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"
    progress_events: list[tuple[int, int, str]] = []

    srt.write_text(
        "\n".join(
            [
                "1\n00:00:01,000 --> 00:00:03,000\nPeace be upon you.\n",
                "2\n00:00:04,000 --> 00:00:06,000\nAnd mercy be upon you.\n",
            ]
        ),
        encoding="utf-8",
    )
    script.write_text(
        "Peace be upon you. And mercy be upon you.",
        encoding="utf-8",
    )
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "mock",
                "model": "mock",
                "translation": {"batch_size": 5},
                "output": {"output_dir": str(out_dir), "write_review_csv": True},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)
    outputs = translate_project(
        str(srt),
        str(script),
        ["ur"],
        loaded,
        review_mode=True,
        progress_callback=lambda current, total, message: progress_events.append((current, total, message)),
    )

    assert "ur" in outputs
    assert progress_events
    assert progress_events[0][2].startswith("Loaded ")
    assert progress_events[-1][2] == "UR: ready"
    assert progress_events[-1][0] == progress_events[-1][1]


def test_pipeline_limits_subtitles_before_batching(tmp_path: Path, monkeypatch) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"

    blocks = []
    script_lines = []
    for index in range(1, 31):
        blocks.append(
            f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\nLine {index}.\n"
        )
        script_lines.append(f"Line {index}.")
    srt.write_text("\n".join(blocks), encoding="utf-8")
    script.write_text(" ".join(script_lines), encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "mock",
                "model": "mock",
                "translation": {"batch_size": 12},
                "output": {"output_dir": str(out_dir), "write_review_csv": False},
            }
        ),
        encoding="utf-8",
    )

    class RecordingProvider:
        def __init__(self) -> None:
            self.indices: list[int] = []
            self.batch_sizes: list[int] = []

        def translate_batch(self, request):
            self.batch_sizes.append(len(request.items))
            self.indices.extend(item.index for item in request.items)
            return [
                TranslationResult(
                    translated_text=f"translated-{item.index}",
                    confidence=0.95,
                    notes=[],
                    provider_metadata={"provider": "recording"},
                )
                for item in request.items
            ]

    provider = RecordingProvider()
    monkeypatch.setattr("translator.pipeline.build_provider", lambda *args, **kwargs: provider)

    loaded = load_config(config)
    outputs = translate_project(
        str(srt),
        str(script),
        ["es"],
        loaded,
        review_mode=False,
        subtitle_limit=20,
    )

    assert "es" in outputs
    assert provider.batch_sizes == [12, 8]
    assert provider.indices == list(range(1, 21))
    content = (out_dir / "input.es.srt").read_text(encoding="utf-8")
    assert "translated-20" in content
    assert "translated-21" not in content


def test_pipeline_emits_debug_mapping_in_order(tmp_path: Path) -> None:
    srt = tmp_path / "input.srt"
    script = tmp_path / "script.txt"
    config = tmp_path / "config.yaml"
    out_dir = tmp_path / "outputs"
    mappings: list[tuple[str, int, str, str]] = []

    srt.write_text(
        "\n".join(
            [
                "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n",
                "2\n00:00:04,000 --> 00:00:06,000\nGeneral Kenobi.\n",
            ]
        ),
        encoding="utf-8",
    )
    script.write_text("Hello there. General Kenobi.", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "mock",
                "model": "mock",
                "translation": {"batch_size": 12},
                "output": {"output_dir": str(out_dir), "write_review_csv": False},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)
    translate_project(
        str(srt),
        str(script),
        ["es"],
        loaded,
        review_mode=False,
        debug_mapping_callback=lambda language, index, source_text, output_text: mappings.append(
            (language, index, source_text, output_text)
        ),
    )

    assert mappings == [
        ("es", 1, "Hello there.", "[es] Hello there."),
        ("es", 2, "General Kenobi.", "[es] General Kenobi."),
    ]
