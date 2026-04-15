# SRTranslate

This project is a local Python CLI for translating `.srt` subtitle files with a reference script in `.pdf`, `.txt`, or `.md` format. It is designed for careful, context-aware subtitle work where meaning, theological precision, historical accuracy, and natural target-language phrasing matter more than literal word substitution.

The default MVP is local-first: it targets a local Ollama server through a provider interface, while keeping OpenAI behind the same abstraction for users who want an API backend later.

The MVP already supports:

- SRT parsing and reconstruction
- Script parsing for `.txt`, `.md`, and `.pdf`
- Script-aware alignment for subtitle context
- Batch translation into multiple languages in one run
- Provider-based translation architecture with swappable backends
- Explicit provider contract for `source_subtitle_text`, `script_context`, `glossary_terms`, `target_language`, and `style_profile`
- Glossary and do-not-translate rules
- Conservative handling hooks for protected religious terms and names
- Verification reports, flags, and review CSV exports
- Optional line rebalancing
- RTL-aware validation for languages such as Urdu, Arabic, and Persian
- Unit tests for parser, alignment, pipeline, and verifier behavior

## Project Structure

```text
.
├── cli.py
├── app.py
├── config.yaml
├── .env.example
├── requirements.txt
├── glossaries/
│   └── sample_glossary.yaml
├── outputs/
├── parsers/
│   ├── alignment.py
│   ├── script_parser.py
│   └── srt_parser.py
├── tests/
│   ├── test_alignment.py
│   ├── test_pipeline.py
│   ├── test_srt_parser.py
│   └── test_verifier.py
├── translator/
│   ├── cli.py
│   ├── config.py
│   ├── factory.py
│   ├── glossary.py
│   ├── models.py
│   ├── pipeline.py
│   ├── reporting.py
│   ├── text.py
│   └── providers/
│       ├── base.py
│       ├── mock.py
│       └── openai_provider.py
└── verifier/
    └── checks.py
```

## How It Works

1. Load the source `.srt`
2. Load the source script from `.pdf`, `.txt`, or `.md`
3. Normalize and segment the script
4. Align each subtitle block to the closest script excerpt
5. Translate block-by-block using the chosen provider
6. Rebuild a clean target-language `.srt` with original numbering and timing preserved
7. Run verification checks
8. Export:
   - `{basename}.{lang}.srt`
   - `{basename}.{lang}.report.json`
   - `{basename}.{lang}.review.csv`
   - `{basename}.{lang}.flags.txt`

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Choose a backend

#### Local-first with Ollama

Install and run Ollama locally, then pull a capable instruction model. Example:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

The default `config.yaml` is already set up for this path.

#### OpenAI backend

If you prefer OpenAI, switch the provider in `config.yaml` or pass `--provider openai --model gpt-5-mini`.

#### Manual/local stub backend

If you want a review-first workflow with no automatic translation, use `--provider manual`.
This provider returns the source subtitle text with explicit uncertainty notes so the rest of the pipeline still runs locally.

### 4. Configure environment variables when using OpenAI

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in your shell or `.env` loader workflow when using the OpenAI provider.

## Usage

### Translate to multiple languages

```bash
python cli.py --srt input.srt --script script.pdf --langs ur ar es
```

### Run the local GUI

```bash
streamlit run app.py
```

### Use a style profile override

```bash
python cli.py --srt input.srt --script script.txt --langs ur ar es id --profile balanced
```

### Use a glossary and emit review artifacts

```bash
python cli.py --srt input.srt --script script.pdf --langs ur --glossary glossaries/sample_glossary.yaml --review-mode
```

### Override provider or model at runtime

```bash
python cli.py --srt input.srt --script script.txt --langs ur ar --provider openai --model gpt-5-mini
```

### Run in manual provider mode

```bash
python cli.py --srt input.srt --script script.txt --langs ur --provider manual --review-mode
```

### Dry-run the pipeline without a real model

For tests or plumbing checks only, switch `provider` to `mock` in `config.yaml`.
This is intentionally not a real translation backend.

## Provider Interface

Each provider implements a single method:

```python
translate(request: TranslationRequest) -> TranslationResult
```

The request includes:

- `source_subtitle_text`
- `script_context`
- `glossary_terms`
- `target_language`
- `style_profile`

The result includes:

- `translated_text`
- `confidence`
- `notes`

This keeps future providers easy to add without changing the pipeline.

## Configuration

`config.yaml` controls:

- model provider and model name
- provider-specific overrides
- source language
- style profile defaults
- low-confidence thresholds
- repair behavior
- line rebalancing
- output directory behavior
- per-language RTL settings
- protected religious terms and names

Example areas you can extend:

- per-language glossary overrides
- language-specific punctuation rules
- custom verifier thresholds
- alternate providers or local models

## Glossary Format

```yaml
terms:
  Moses: Musa
  Jesus: Isa
  Abraham: Ibrahim
do_not_translate:
  - Allah
  - Quran
protected_terms:
  - Allah
  - Muhammad
```

## Verification Pipeline

Each translated subtitle file is checked for:

- SRT structure
- numbering and timestamp preservation
- missing or empty blocks
- suspicious leftover source-language text
- repeated-source consistency
- glossary-backed and protected-term preservation
- weak alignment to the script
- low model confidence
- likely omissions based on compression heuristics
- readability issues such as very long lines
- RTL script presence for Urdu, Arabic, and Persian

The verifier is intentionally conservative. It uses layered heuristics and reports uncertainty, but it does not prove semantic correctness.

## Tests

Run:

```bash
pytest
```

## GUI

The Streamlit app is intended for non-technical local use on macOS, Windows, and Linux.

Features included:

- upload `.srt` and script files
- multi-language selection
- glossary upload
- style profile selection
- review mode toggle
- progress and status display
- download buttons for generated outputs
- translated subtitle preview
- RTL-friendly preview for Urdu, Arabic, and Persian
- lightweight session-based run history

## Limitations

- The current alignment method is heuristic and based on string similarity, not forced alignment or timestamp-aware script mapping.
- The verifier uses practical heuristics, not full semantic evaluation.
- The Ollama and OpenAI providers currently translate one subtitle block at a time, which is simple and safe but not yet optimal for larger discourse-level coherence.
- Repair mode is only partially scaffolded through confidence thresholds and warnings; it does not yet run a second model pass automatically.
- RTL validation checks for script presence and Unicode safety signals, but it is not a full renderer compatibility test across all subtitle players.
- PDF extraction quality depends heavily on the input PDF's text layer.
- The mock provider is only for tests and plumbing validation, not translation quality.

## Recommended Local Workflow For Improving Quality

Use this workflow to steadily improve trustworthiness:

1. Start with a strong reference script.
   A clean script dramatically improves correction, disambiguation, and terminology stability.

2. Build a domain glossary before large translation runs.
   Add names, scripture references, historical figures, theological terms, and non-translatable expressions.

3. Translate in batches by language.
   Review generated `.review.csv` files first, especially low-confidence or weak-alignment rows.

4. Tighten verifier rules as you observe real failures.
   Add new consistency checks, line-length thresholds, and protected-term rules per language.

5. Keep correction artifacts.
   When a reviewer fixes a translation, promote the fix into glossary rules, prompt guidance, or future model-side constraints.

6. Add language-specific post-processing over time.
   Urdu and Arabic often benefit from dedicated punctuation, honorific, and subtitle line-break handling.

7. Move toward chunk-level context windows.
   A future improvement is translating neighboring subtitle ranges together while preserving per-block timing in output.

8. Add a second-pass reviewer model.
   The best next architectural upgrade is a repair pass that only revisits flagged blocks using the first-pass translation, the source subtitle, and the aligned script excerpt.

## Production Direction

For a stronger production version, the highest-value next steps are:

- richer subtitle-to-script alignment
- segment-group translation with local context windows
- automatic repair pass for flagged blocks
- per-language termbanks
- reviewer feedback ingestion
- renderer-focused RTL testing
- provider support for local or self-hosted models
