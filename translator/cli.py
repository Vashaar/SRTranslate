from __future__ import annotations

import argparse
import logging

from translator.config import load_config
from translator.pipeline import translate_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files using a reference script for context and verification."
    )
    parser.add_argument("--srt", required=True, help="Path to the input .srt file")
    parser.add_argument(
        "--script",
        required=True,
        help="Path to the source script (.pdf, .txt, or .md)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        required=True,
        help="Target language codes, for example: ur ar es",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--glossary", help="Optional glossary YAML path")
    parser.add_argument("--profile", help="Style profile override")
    parser.add_argument("--provider", help="Provider override, for example: ollama or openai")
    parser.add_argument("--model", help="Model override for the selected provider")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only the first 20 subtitle blocks for a fast test run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N subtitle blocks. Overrides --test-mode when provided.",
    )
    parser.add_argument(
        "--review-mode",
        action="store_true",
        help="Enable extra in-memory review checks without generating spreadsheet exports.",
    )
    parser.add_argument(
        "--debug-performance",
        action="store_true",
        help="Print total runtime and average time per batch.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0.")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(args.config)
    if args.provider:
        config.raw["provider"] = args.provider
    if args.model:
        config.raw["model"] = args.model
    subtitle_limit = args.limit if args.limit is not None else (20 if args.test_mode else None)
    current_language: str | None = None

    def emit_debug_mapping(language: str, index: int, source_text: str, translated_text: str) -> None:
        nonlocal current_language
        if current_language != language:
            current_language = language
            print(f"{language}:")
        formatted_input = " | ".join(line.strip() for line in str(source_text).splitlines() if line.strip())
        formatted_output = " | ".join(line.strip() for line in str(translated_text).splitlines() if line.strip())
        print(f"[{index}] INPUT: {formatted_input}")
        print(f"[{index}] OUTPUT: {formatted_output}")

    outputs = translate_project(
        srt_path=args.srt,
        script_path=args.script,
        langs=args.langs,
        config=config,
        glossary_path=args.glossary,
        profile=args.profile,
        review_mode=args.review_mode,
        subtitle_limit=subtitle_limit,
        debug_mapping_callback=emit_debug_mapping,
        debug_performance=args.debug_performance,
    )
    for lang, path in outputs.items():
        print(f"{lang}: {path}")
    return 0
