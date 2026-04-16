from __future__ import annotations
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from urllib import error, request
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from desktop_paths import AppPaths
from parsers.srt_parser import parse_srt
from translator.config import AppConfig, load_config
from translator.dictionary_store import (
    StoredDictionary,
    dictionary_path,
    import_dictionary,
    list_dictionaries,
)
from translator.models import LanguageArtifacts, LanguageConfig
from translator.pipeline import translate_project_with_artifacts


PRESET_OPTIONS = {
    "Fast": {
        "style_profile": "natural",
        "batch_size": 12,
        "retry_low_confidence": False,
    },
    "Accurate": {
        "style_profile": "balanced",
        "batch_size": 12,
        "retry_low_confidence": True,
    },
    "Religious-safe": {
        "style_profile": "literal",
        "batch_size": 12,
        "retry_low_confidence": True,
    },
}
THEME = {
    "bg": "#08171C",
    "panel": "#10252B",
    "panel_alt": "#16333A",
    "panel_edge": "#21454E",
    "accent": "#2FC7B6",
    "accent_hover": "#47D8C8",
    "gold": "#D6B447",
    "gold_soft": "#FFE38A",
    "text": "#F5FFFC",
    "muted": "#A9D5CF",
    "entry": "#0B1F24",
    "entry_border": "#24545D",
    "success": "#A6F29B",
    "warning": "#FFD87B",
    "error": "#FF9B8E",
}

logger = logging.getLogger(__name__)
OLLAMA_STARTUP_TIMEOUT_SECONDS = 20.0


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.paths = AppPaths.detect()
        self._configure_logging()

        self.title("SRTranslate")
        self.geometry("1080x820")
        self.minsize(980, 760)
        self.configure(bg=THEME["bg"])

        self.selected_srt_path: Path | None = None
        self.selected_script_path: Path | None = None
        self.dictionary_records: list[StoredDictionary] = []
        self.glossary_options: dict[str, Path | None] = {"None": None}
        self.current_output_dir: Path | None = None
        self.current_artifacts: dict[str, LanguageArtifacts] = {}
        self.window_icon: ImageTk.PhotoImage | None = None
        self.hero_image: ImageTk.PhotoImage | None = None
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.output_base_dir = self.paths.output_root
        self.provider_ready = False
        self.provider_lock = threading.Lock()
        self.current_provider_name = ""

        default_config = load_config(self.paths.config_path)
        self.current_provider_name = default_config.provider
        self.language_options: list[LanguageConfig] = default_config.supported_languages()
        self.language_by_code = {language.code: language for language in self.language_options}
        self.language_by_label = {language.label: language for language in self.language_options}

        self.srt_path_var = tk.StringVar(value="No subtitle file selected yet.")
        self.script_path_var = tk.StringVar(value="No reference script selected.")
        self.preset_var = tk.StringVar(value="Accurate")
        self.dictionary_var = tk.StringVar(value="None")
        self.review_mode_var = tk.BooleanVar(
            value=bool(default_config.raw.get("output", {}).get("write_review_csv", True))
        )
        self.test_mode_var = tk.BooleanVar(value=False)
        self.limit_var = tk.StringVar(value="")
        self.deen_mode_var = tk.BooleanVar(value=default_config.deen_mode)
        default_target = self.language_by_code.get(default_config.target_language)
        self.target_language_var = tk.StringVar(value=default_target.label if default_target else "Spanish")
        self.status_var = tk.StringVar(value="Select an SRT file to begin.")
        self.output_var = tk.StringVar(value=str(self.output_base_dir))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Waiting to start")
        self.device_var = tk.StringVar(value="Runtime: waiting")
        self.total_time_var = tk.StringVar(value="Total time: --")
        self.batch_time_var = tk.StringVar(value="Latest batch: --")
        self.processed_count_var = tk.StringVar(value="Subtitles processed: 0")
        self.status_state_var = tk.StringVar(value="Ready")
        self.success_var = tk.StringVar(value="")
        self.warning_var = tk.StringVar(value="")
        self.advanced_open = tk.BooleanVar(value=False)
        self.advanced_button_var = tk.StringVar(value="More Options")
        self.processed_subtitle_total = 0

        self._configure_theme()
        self._load_branding()
        self._build_ui()
        self._select_default_languages()
        self._refresh_glossary_views()
        self._start_dependency_bootstrap(default_config)
        self.after(150, self._poll_events)

    def _configure_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        log_path = self.paths.log_path.resolve()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path:
                return

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root_logger.addHandler(file_handler)

    def _configure_theme(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=THEME["bg"], foreground=THEME["text"])
        style.configure("App.TFrame", background=THEME["bg"])
        style.configure("Panel.TFrame", background=THEME["panel"])
        style.configure("Hero.TFrame", background=THEME["panel_alt"])
        style.configure("Card.TFrame", background=THEME["panel"], borderwidth=1, relief="solid")
        style.configure("App.TLabel", background=THEME["bg"], foreground=THEME["text"])
        style.configure("Muted.TLabel", background=THEME["bg"], foreground=THEME["muted"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=THEME["panel"], foreground=THEME["text"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=THEME["bg"], foreground=THEME["gold_soft"], font=("Segoe UI Semibold", 10))
        style.configure("CardTitle.TLabel", background=THEME["panel"], foreground=THEME["gold_soft"], font=("Segoe UI Semibold", 10))
        style.configure(
            "Success.TLabel",
            background=THEME["bg"],
            foreground=THEME["success"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Warning.TLabel",
            background=THEME["bg"],
            foreground=THEME["warning"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Error.TLabel",
            background=THEME["bg"],
            foreground=THEME["error"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "StatusBadge.TLabel",
            background=THEME["panel_alt"],
            foreground=THEME["gold_soft"],
            font=("Segoe UI Semibold", 9),
            padding=(10, 4),
        )
        style.configure(
            "HeroTitle.TLabel",
            background=THEME["panel_alt"],
            foreground=THEME["gold_soft"],
            font=("Segoe UI Semibold", 24),
        )
        style.configure(
            "HeroBody.TLabel",
            background=THEME["panel_alt"],
            foreground=THEME["muted"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "TLabelframe",
            background=THEME["panel"],
            foreground=THEME["gold_soft"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=THEME["panel"],
            foreground=THEME["gold_soft"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Primary.TButton",
            background=THEME["accent"],
            foreground=THEME["bg"],
            borderwidth=0,
            focusthickness=0,
            focuscolor=THEME["accent"],
            padding=(16, 11),
            font=("Segoe UI Semibold", 11),
            relief="flat",
        )
        style.map(
            "Primary.TButton",
            background=[("active", THEME["accent_hover"]), ("disabled", THEME["entry_border"])],
            foreground=[("disabled", THEME["muted"])],
        )
        style.configure(
            "Secondary.TButton",
            background=THEME["panel_alt"],
            foreground=THEME["text"],
            borderwidth=0,
            focusthickness=0,
            padding=(14, 9),
            font=("Segoe UI", 10),
            relief="flat",
        )
        style.map(
            "Secondary.TButton",
            background=[("active", THEME["panel_edge"]), ("disabled", THEME["entry_border"])],
            foreground=[("disabled", THEME["muted"])],
        )
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=THEME["entry"],
            bordercolor=THEME["entry_border"],
            background=THEME["gold"],
            lightcolor=THEME["gold"],
            darkcolor=THEME["gold"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=THEME["entry"],
            background=THEME["entry"],
            foreground=THEME["text"],
            arrowcolor=THEME["gold"],
            bordercolor=THEME["entry_border"],
            padding=6,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["entry"])],
            selectbackground=[("readonly", THEME["entry"])],
        )
        style.configure("TCheckbutton", background=THEME["panel"], foreground=THEME["text"], font=("Segoe UI", 10))

        self.option_add("*TCombobox*Listbox.background", THEME["entry"])
        self.option_add("*TCombobox*Listbox.foreground", THEME["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", THEME["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", THEME["bg"])

    def _load_branding(self) -> None:
        logo_path = self.paths.logo_path
        if not logo_path.exists():
            return

        try:
            image = Image.open(logo_path)
        except OSError as exc:
            logger.warning("Could not load logo from %s (%s)", logo_path, exc)
            return

        icon_image = image.copy()
        icon_image.thumbnail((128, 128))
        hero_image = image.copy()
        hero_image.thumbnail((118, 118))
        self.window_icon = ImageTk.PhotoImage(icon_image)
        self.hero_image = ImageTk.PhotoImage(hero_image)
        self.iconphoto(True, self.window_icon)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, style="App.TFrame")
        outer.pack(fill="both", expand=True, padx=16, pady=16)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        self._build_header(outer)
        self._build_home(outer)

    def _start_dependency_bootstrap(self, config: AppConfig) -> None:
        if config.provider != "ollama":
            self.provider_ready = True
            return
        self.status_var.set("Starting local translation engine...")
        threading.Thread(
            target=self._dependency_bootstrap_worker,
            args=(config,),
            daemon=True,
        ).start()

    def _dependency_bootstrap_worker(self, config: AppConfig) -> None:
        try:
            self._ensure_provider_ready(config, start_if_needed=True)
            self.event_queue.put(("dependency-ready", "Local translation engine is ready."))
        except Exception as exc:
            logger.exception("Could not auto-start the local translation engine")
            self.event_queue.put(("dependency-error", str(exc)))

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Hero.TFrame", padding=20)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        if self.hero_image is not None:
            image_label = ttk.Label(header, image=self.hero_image, style="App.TLabel")
            image_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 18))
            image_label.configure(background=THEME["panel_alt"])

        ttk.Label(header, text="SRTranslate", style="HeroTitle.TLabel").grid(
            row=0,
            column=1,
            sticky="sw",
        )
        ttk.Label(
            header,
            text="Local AI Subtitle Translator",
            style="HeroBody.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=1, sticky="nw", pady=(6, 0))

    def _build_home(self, parent: ttk.Frame) -> None:
        shell = ttk.Frame(parent, style="App.TFrame")
        shell.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.home_canvas = tk.Canvas(
            shell,
            bg=THEME["bg"],
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.home_canvas.grid(row=0, column=0, sticky="nsew")

        shell_scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.home_canvas.yview)
        shell_scrollbar.grid(row=0, column=1, sticky="ns")
        self.home_canvas.configure(yscrollcommand=shell_scrollbar.set)

        body = ttk.Frame(self.home_canvas, style="App.TFrame")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)
        self._home_window = self.home_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _event: self.home_canvas.configure(scrollregion=self.home_canvas.bbox("all")))
        self.home_canvas.bind(
            "<Configure>",
            lambda event: self.home_canvas.itemconfigure(self._home_window, width=event.width),
        )
        self.home_canvas.bind("<Enter>", lambda _event: self._bind_home_scroll())
        self.home_canvas.bind("<Leave>", lambda _event: self._unbind_home_scroll())

        top_section = ttk.Frame(body, style="App.TFrame")
        top_section.grid(row=0, column=0, sticky="ew")
        top_section.columnconfigure(0, weight=1)

        title_block = ttk.Frame(top_section, style="App.TFrame")
        title_block.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(title_block, text="Project Setup", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_block,
            text="Pick an SRT file, choose the target language, and launch the local translation run.",
            style="Muted.TLabel",
            wraplength=920,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ttk.LabelFrame(body, text="Main Controls", padding=16)
        actions.grid(row=1, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=0)

        ttk.Label(actions, text="SRT File", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Choose File", command=self._choose_srt, style="Secondary.TButton").grid(
            row=0,
            column=1,
            sticky="e",
        )
        ttk.Label(
            actions,
            textvariable=self.srt_path_var,
            style="Card.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 16))

        ttk.Label(actions, text="Target Language", style="CardTitle.TLabel").grid(row=2, column=0, sticky="w")
        self.target_language_combo = ttk.Combobox(
            actions,
            textvariable=self.target_language_var,
            values=[language.label for language in self.language_options],
            state="readonly",
        )
        self.target_language_combo.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 16))
        self.target_language_combo.bind("<<ComboboxSelected>>", self._handle_target_language_change)

        ttk.Checkbutton(
            actions,
            text="Deen Mode",
            variable=self.deen_mode_var,
        ).grid(row=4, column=0, sticky="w", pady=(0, 16))

        ttk.Label(actions, text="Reference Script (Optional)", style="CardTitle.TLabel").grid(row=5, column=0, sticky="w")
        ttk.Button(
            actions,
            text="Choose File",
            command=self._choose_script,
            style="Secondary.TButton",
        ).grid(row=5, column=1, sticky="e")
        ttk.Label(
            actions,
            textvariable=self.script_path_var,
            style="Card.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 16))

        ttk.Label(actions, text="Output Folder", style="CardTitle.TLabel").grid(row=7, column=0, sticky="w")
        ttk.Button(
            actions,
            text="Choose Folder",
            command=self._choose_output_dir,
            style="Secondary.TButton",
        ).grid(row=7, column=1, sticky="e")
        ttk.Label(
            actions,
            textvariable=self.output_var,
            style="Card.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        action_bar = ttk.Frame(body, style="App.TFrame")
        action_bar.grid(row=2, column=0, sticky="ew", pady=(16, 10))
        action_bar.columnconfigure(0, weight=1)
        self.translate_button = ttk.Button(
            action_bar,
            text="Translate",
            command=self._start_translation,
            style="Primary.TButton",
        )
        self.translate_button.grid(row=0, column=0, sticky="ew")

        ttk.Button(
            body,
            textvariable=self.advanced_button_var,
            command=self._toggle_advanced,
            style="Secondary.TButton",
        ).grid(row=3, column=0, sticky="w", pady=(0, 10))

        self.advanced_frame = ttk.LabelFrame(body, text="More Options", padding=16)
        self.advanced_frame.grid(row=4, column=0, sticky="ew")
        self.advanced_frame.columnconfigure(1, weight=1)

        ttk.Label(self.advanced_frame, text="Translation mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            self.advanced_frame,
            textvariable=self.preset_var,
            values=list(PRESET_OPTIONS),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=(0, 12))

        ttk.Checkbutton(
            self.advanced_frame,
            text="Test Mode",
            variable=self.test_mode_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(self.advanced_frame, text="Limit").grid(row=2, column=0, sticky="w")
        ttk.Entry(
            self.advanced_frame,
            textvariable=self.limit_var,
        ).grid(row=2, column=1, sticky="ew", pady=(0, 12))

        ttk.Label(self.advanced_frame, text="Additional target languages").grid(row=3, column=0, sticky="w")
        language_panel = ttk.Frame(self.advanced_frame, style="Panel.TFrame")
        language_panel.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 14))
        self.language_vars: dict[str, tk.BooleanVar] = {}
        for index, language in enumerate(self.language_options):
            variable = tk.BooleanVar(value=False)
            self.language_vars[language.code] = variable
            ttk.Checkbutton(
                language_panel,
                text=language.label,
                variable=variable,
                command=lambda code=language.code: self._sync_primary_language_from_checkboxes(code),
            ).grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 18), pady=4)

        ttk.Label(self.advanced_frame, text="Glossary").grid(row=5, column=0, sticky="w")
        self.dictionary_combo = ttk.Combobox(
            self.advanced_frame,
            textvariable=self.dictionary_var,
            state="readonly",
        )
        self.dictionary_combo.grid(row=5, column=1, sticky="ew", pady=(0, 10))

        glossary_actions = ttk.Frame(self.advanced_frame, style="Panel.TFrame")
        glossary_actions.grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 10))
        ttk.Button(
            glossary_actions,
            text="Refresh Glossaries",
            command=self._refresh_glossary_views,
        ).grid(row=0, column=0)
        ttk.Button(
            glossary_actions,
            text="Import Glossary",
            command=self._import_dictionary,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Checkbutton(
            self.advanced_frame,
            text="Review mode (in-memory only)",
            variable=self.review_mode_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        self.advanced_frame.grid_remove()

        status = ttk.LabelFrame(body, text="Status", padding=16)
        status.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        status.columnconfigure(0, weight=1)
        status.columnconfigure(1, weight=0)

        ttk.Label(status, textvariable=self.status_state_var, style="StatusBadge.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        self.progress = ttk.Progressbar(
            status,
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew")

        ttk.Label(
            status,
            textvariable=self.status_var,
            style="Muted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(status, textvariable=self.progress_text_var, style="Muted.TLabel").grid(
            row=3,
            column=0,
            sticky="w",
            pady=(6, 0),
        )
        metrics = ttk.Frame(status, style="App.TFrame")
        metrics.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        ttk.Label(metrics, textvariable=self.device_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(metrics, textvariable=self.total_time_var, style="Muted.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(metrics, textvariable=self.batch_time_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(metrics, textvariable=self.processed_count_var, style="Muted.TLabel").grid(row=1, column=1, sticky="w", pady=(4, 0))

        ttk.Label(status, text="Output folder", style="Muted.TLabel").grid(
            row=5,
            column=0,
            sticky="w",
            pady=(12, 0),
        )
        ttk.Label(
            status,
            textvariable=self.output_var,
            style="Muted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            status,
            textvariable=self.success_var,
            style="Success.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(
            status,
            textvariable=self.warning_var,
            style="Error.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        debug_panel = ttk.LabelFrame(body, text="Footer Log", padding=10)
        debug_panel.grid(row=6, column=0, sticky="nsew", pady=(14, 0))
        debug_panel.columnconfigure(0, weight=1)
        debug_panel.rowconfigure(0, weight=1)

        self.debug_text = tk.Text(
            debug_panel,
            height=12,
            wrap="word",
            bg=THEME["entry"],
            fg=THEME["text"],
            insertbackground=THEME["text"],
            highlightbackground=THEME["entry_border"],
            relief="flat",
            font=("Consolas", 10),
        )
        self.debug_text.grid(row=0, column=0, sticky="nsew")
        self.debug_text.configure(state="disabled")

        debug_scrollbar = ttk.Scrollbar(debug_panel, orient="vertical", command=self.debug_text.yview)
        debug_scrollbar.grid(row=0, column=1, sticky="ns")
        self.debug_text.configure(yscrollcommand=debug_scrollbar.set)

    def _bind_home_scroll(self) -> None:
        self.home_canvas.bind_all("<MouseWheel>", self._on_home_scroll)

    def _unbind_home_scroll(self) -> None:
        self.home_canvas.unbind_all("<MouseWheel>")

    def _on_home_scroll(self, event: tk.Event) -> None:
        if getattr(event, "delta", 0):
            self.home_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _select_default_languages(self) -> None:
        selected_language = self.language_by_label.get(self.target_language_var.get())
        if selected_language is None:
            selected_language = self.language_by_code.get("es") or next(iter(self.language_options), None)
        if selected_language is None:
            return
        self.target_language_var.set(selected_language.label)
        self._set_selected_languages(selected_language.code)

    def _set_selected_languages(self, primary_language_code: str, *, clear_existing: bool = True) -> None:
        for code, variable in self.language_vars.items():
            if clear_existing:
                variable.set(code == primary_language_code)
            elif code == primary_language_code:
                variable.set(True)

    def _handle_target_language_change(self, _event: object | None = None) -> None:
        selected_language = self.language_by_label.get(self.target_language_var.get())
        if selected_language is None:
            return
        self._set_selected_languages(selected_language.code)

    def _sync_primary_language_from_checkboxes(self, code: str) -> None:
        variable = self.language_vars.get(code)
        if variable is None or not variable.get():
            return
        language = self.language_by_code.get(code)
        if language is not None:
            self.target_language_var.set(language.label)

    def _toggle_advanced(self) -> None:
        if self.advanced_open.get():
            self.advanced_frame.grid_remove()
            self.advanced_button_var.set("More Options")
            self.advanced_open.set(False)
            return
        self.advanced_frame.grid()
        self.advanced_button_var.set("Hide Options")
        self.advanced_open.set(True)

    def _choose_srt(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose subtitle file",
            filetypes=[("SRT files", "*.srt")],
        )
        if selected:
            self.selected_srt_path = Path(selected)
            self.srt_path_var.set(selected)

    def _choose_script(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose reference script",
            filetypes=[("Script files", "*.pdf *.txt *.md"), ("All files", "*.*")],
        )
        if selected:
            self.selected_script_path = Path(selected)
            self.script_path_var.set(selected)

    def _choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=str(self.output_base_dir),
            mustexist=False,
        )
        if selected:
            self.output_base_dir = Path(selected)
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            self.output_var.set(str(self.output_base_dir))

    def _selected_language_codes(self) -> list[str]:
        selected_codes = [
            language.code
            for language in self.language_options
            if self.language_vars.get(language.code) and self.language_vars[language.code].get()
        ]
        if selected_codes:
            return selected_codes
        selected_language = self.language_by_label.get(self.target_language_var.get())
        return [selected_language.code] if selected_language is not None else []

    def _built_in_glossaries(self) -> list[Path]:
        directory = self.paths.bundled_glossaries_dir
        if not directory.exists():
            return []
        patterns = ("*.yaml", "*.yml", "*.json", "*.csv", "*.tsv", "*.txt")
        results: list[Path] = []
        for pattern in patterns:
            results.extend(directory.glob(pattern))
        return sorted({path.resolve() for path in results}, key=lambda item: item.stem.lower())

    def _refresh_glossary_views(self) -> None:
        current_value = self.dictionary_var.get()
        self.dictionary_records = list_dictionaries(self.paths.storage_root)
        options: dict[str, Path | None] = {"None": None}

        for glossary_path in self._built_in_glossaries():
            label = f"Built-in: {glossary_path.stem.replace('_', ' ').title()}"
            options[label] = glossary_path

        for record in self.dictionary_records:
            label = f"Library: {record.name}"
            options[label] = dictionary_path(record, self.paths.storage_root)

        self.glossary_options = options
        labels = list(options)
        self.dictionary_combo.configure(values=labels)
        if current_value in options:
            self.dictionary_var.set(current_value)
        else:
            self.dictionary_var.set("None")

    def _import_dictionary(self) -> None:
        selected = filedialog.askopenfilename(
            title="Import glossary",
            filetypes=[
                ("Dictionary files", "*.yaml *.yml *.json *.csv *.tsv *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        self._set_busy(True, "Importing glossary...")
        threading.Thread(
            target=self._import_dictionary_worker,
            args=(selected,),
            daemon=True,
        ).start()

    def _import_dictionary_worker(self, path: str) -> None:
        try:
            record = import_dictionary(path, None, base_dir=self.paths.storage_root)
            self.event_queue.put(("dictionary-success", record))
        except Exception as exc:
            logger.exception("Glossary import failed")
            self.event_queue.put(("dictionary-error", str(exc)))

    def _selected_glossary_path(self) -> str | None:
        selected = self.dictionary_var.get()
        glossary_path = self.glossary_options.get(selected)
        return str(glossary_path) if glossary_path else None

    def _set_inline_feedback(
        self,
        state: str,
        message: str,
        *,
        success: str = "",
        warning: str = "",
    ) -> None:
        self.status_state_var.set(state)
        self.status_var.set(message)
        self.success_var.set(success)
        self.warning_var.set(warning)

    def _start_translation(self) -> None:
        if self.selected_srt_path is None:
            self._set_inline_feedback(
                "Error",
                "Please select an SRT subtitle file before starting.",
                warning="Missing subtitle file.",
            )
            return

        languages = self._selected_language_codes()
        if not languages:
            self._set_inline_feedback(
                "Error",
                "Choose at least one target language before starting.",
                warning="No target language selected.",
            )
            return
        try:
            subtitle_limit = self._resolve_subtitle_limit()
        except ValueError as exc:
            self._set_inline_feedback("Error", str(exc), warning="Invalid limit.")
            return

        self.current_output_dir = None
        self.current_artifacts = {}
        self.output_var.set(str(self.output_base_dir))
        self.progress_var.set(0)
        self.progress_text_var.set("Starting translation")
        self._reset_runtime_insights()
        self._set_inline_feedback("Processing...", "Running translation...")
        self._set_busy(True, "Processing...")

        threading.Thread(
            target=self._translation_worker,
            args=(
                str(self.selected_srt_path),
                str(self.selected_script_path) if self.selected_script_path else None,
                languages,
                self._selected_glossary_path(),
                self.preset_var.get(),
                self.review_mode_var.get(),
                str(self.output_base_dir),
                subtitle_limit,
            ),
            daemon=True,
        ).start()

    def _translation_worker(
        self,
        srt_path: str,
        script_path: str | None,
        languages: list[str],
        glossary_path: str | None,
        preset_name: str,
        review_mode: bool,
        output_dir: str,
        subtitle_limit: int | None,
    ) -> None:
        def report_progress(current: int, total: int, message: str) -> None:
            self.event_queue.put(
                (
                    "progress",
                    {
                        "current": current,
                        "total": total,
                        "message": message,
                    },
                )
            )

        def report_debug_mapping(language: str, index: int, source_text: str, translated_text: str) -> None:
            self.event_queue.put(
                (
                    "debug-log",
                    {
                        "language": language,
                        "index": index,
                        "source_text": source_text,
                        "translated_text": translated_text,
                    },
                )
            )

        def report_runtime_info(device: str, precision: str) -> None:
            self.event_queue.put(
                (
                    "runtime-info",
                    {
                        "device": device,
                        "precision": precision,
                    },
                )
            )

        def report_batch_metrics(
            language: str,
            current_batch: int,
            total_batches: int,
            subtitle_count: int,
            elapsed_seconds: float,
        ) -> None:
            self.event_queue.put(
                (
                    "batch-metric",
                    {
                        "language": language,
                        "current_batch": current_batch,
                        "total_batches": total_batches,
                        "subtitle_count": subtitle_count,
                        "elapsed_seconds": elapsed_seconds,
                    },
                )
            )

        def report_performance_summary(
            total_runtime: float,
            average_batch_time: float,
            processed_subtitles: int,
        ) -> None:
            self.event_queue.put(
                (
                    "performance-summary",
                    {
                        "total_runtime": total_runtime,
                        "average_batch_time": average_batch_time,
                        "processed_subtitles": processed_subtitles,
                    },
                )
            )

        try:
            config, style_profile = self._config_for_preset(preset_name, review_mode)
            self.current_provider_name = config.provider
            self._ensure_provider_ready(config, start_if_needed=True)
            run_dir = Path(output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            config.raw.setdefault("output", {})
            config.raw["output"]["output_dir"] = str(run_dir)

            resolved_script_path = self._prepare_reference_script(
                srt_path=srt_path,
                script_path=script_path,
                run_dir=run_dir,
            )

            artifacts = translate_project_with_artifacts(
                srt_path=srt_path,
                script_path=str(resolved_script_path),
                langs=languages,
                config=config,
                glossary_path=glossary_path,
                profile=style_profile,
                review_mode=review_mode,
                progress_callback=report_progress,
                subtitle_limit=subtitle_limit,
                debug_mapping_callback=report_debug_mapping,
                debug_performance=True,
                runtime_info_callback=report_runtime_info,
                batch_metrics_callback=report_batch_metrics,
                performance_summary_callback=report_performance_summary,
            )
            self.event_queue.put(("translation-success", (run_dir, artifacts)))
        except Exception as exc:
            logger.exception("Translation failed")
            self.event_queue.put(("translation-error", str(exc)))

    def _config_for_preset(self, preset_name: str, review_mode: bool) -> tuple[AppConfig, str]:
        config = load_config(self.paths.config_path)
        preset = PRESET_OPTIONS.get(preset_name, PRESET_OPTIONS["Accurate"])
        style_profile = str(preset["style_profile"])
        selected_language = self.language_by_label.get(self.target_language_var.get())

        config.raw["style_profile"] = style_profile
        config.raw["deen_mode"] = bool(self.deen_mode_var.get() or preset_name == "Religious-safe")
        if selected_language is not None:
            config.raw["target_language"] = selected_language.code
        config.raw.setdefault("translation", {})
        config.raw["translation"]["batch_size"] = int(preset["batch_size"])
        config.raw["translation"]["retry_low_confidence"] = bool(preset["retry_low_confidence"])

        config.raw.setdefault("output", {})
        config.raw["output"]["write_review_csv"] = bool(review_mode)

        return config, style_profile

    def _prepare_reference_script(
        self,
        srt_path: str,
        script_path: str | None,
        run_dir: Path,
    ) -> Path:
        if script_path:
            return Path(script_path)

        subtitle_blocks = parse_srt(srt_path)
        reference_text = "\n\n".join(block.text for block in subtitle_blocks if block.text.strip())
        reference_path = run_dir / f"{Path(srt_path).stem}.reference.txt"
        reference_path.write_text(reference_text, encoding="utf-8")
        return reference_path

    def _ensure_provider_ready(self, config: AppConfig, *, start_if_needed: bool = False) -> None:
        if config.provider != "ollama":
            self.provider_ready = True
            return

        with self.provider_lock:
            if self.provider_ready and self._ollama_is_healthy(config):
                return

            if self._ollama_is_healthy(config):
                self.provider_ready = True
                return

            if not start_if_needed:
                raise RuntimeError(
                    "SRTranslate could not reach the local Ollama service. "
                    "Please start Ollama, confirm it is running, and try again."
                )

            ollama_executable = self._find_ollama_executable()
            if ollama_executable is None:
                raise RuntimeError(
                    "SRTranslate could not find Ollama installed on this computer."
                )

            self._launch_ollama(ollama_executable)
            deadline = time.monotonic() + OLLAMA_STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if self._ollama_is_healthy(config):
                    self.provider_ready = True
                    return
                time.sleep(0.5)

            raise RuntimeError(
                "SRTranslate started Ollama but could not connect to it in time. "
                "Please wait a few seconds and try again."
            )

    def _ollama_is_healthy(self, config: AppConfig) -> bool:
        base_url = str(
            config.provider_settings("ollama").get("base_url", "http://127.0.0.1:11434")
        ).rstrip("/")
        health_url = f"{base_url}/api/tags"
        try:
            with request.urlopen(health_url, timeout=2) as response:
                return response.status < 400
        except (error.URLError, RuntimeError):
            return False

    @staticmethod
    def _find_ollama_executable() -> Path | None:
        direct_match = shutil.which("ollama")
        if direct_match:
            return Path(direct_match)

        local_appdata = Path(os.getenv("LOCALAPPDATA", ""))
        program_files = Path(os.getenv("ProgramFiles", ""))
        candidates = [
            local_appdata / "Programs" / "Ollama" / "ollama.exe",
            local_appdata / "AMD" / "AI_Bundle" / "Ollama" / "ollama.exe",
            program_files / "Ollama" / "ollama.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _launch_ollama(executable: Path) -> None:
        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS

        subprocess.Popen(
            [str(executable), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def _count_fallback_lines(self, artifacts: dict[str, LanguageArtifacts]) -> int:
        fallback_count = 0
        for artifact in artifacts.values():
            try:
                fallback_count += int(artifact.report.summary.get("fallback_count", 0))
            except (TypeError, ValueError):
                logger.warning("Invalid fallback count for language %s", artifact.language)
        return fallback_count

    def _count_total_blocks(self, artifacts: dict[str, LanguageArtifacts]) -> int:
        total_blocks = 0
        for artifact in artifacts.values():
            try:
                total_blocks += int(artifact.report.summary.get("translated_blocks", 0))
            except (TypeError, ValueError):
                logger.warning("Invalid translated block count for language %s", artifact.language)
        return total_blocks

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event, payload)
        self.after(150, self._poll_events)

    def _handle_event(self, event: str, payload: object) -> None:
        if event == "dictionary-success":
            record = payload
            self._refresh_glossary_views()
            imported_label = f"Library: {record.name}"
            if imported_label in self.glossary_options:
                self.dictionary_var.set(imported_label)
            self._set_inline_feedback("Ready", f"Glossary '{record.name}' is ready.")
            self._set_busy(False, self.status_var.get())
            return

        if event == "dependency-ready":
            self.provider_ready = True
            if not self.translate_button.instate(["disabled"]):
                self._set_inline_feedback("Ready", str(payload))
            return

        if event == "dependency-error":
            self.provider_ready = False
            if not self.translate_button.instate(["disabled"]):
                self._set_inline_feedback(
                    "Error",
                    "Local translation engine is not ready yet.",
                    warning=str(payload),
                )
            return

        if event == "dictionary-error":
            self._set_inline_feedback("Error", "Glossary import failed.", warning=str(payload))
            self._set_busy(False, self.status_var.get())
            return

        if event == "progress":
            progress_payload = payload
            current = int(progress_payload["current"])
            total = max(1, int(progress_payload["total"]))
            message = str(progress_payload["message"])
            percent = round((current / total) * 100, 1)
            self.progress_var.set(percent)
            self.progress_text_var.set(f"{percent:.1f}% complete")
            self.status_state_var.set("Processing...")
            self.status_var.set(message)
            return

        if event == "translation-success":
            run_dir, artifacts = payload
            self.current_output_dir = run_dir
            self.current_artifacts = artifacts
            self.output_var.set(str(run_dir))
            self.progress_var.set(100)
            self.progress_text_var.set("100% complete")

            fallback_count = self._count_fallback_lines(artifacts)
            total_blocks = self._count_total_blocks(artifacts)
            if total_blocks and fallback_count >= total_blocks:
                self.progress_var.set(0)
                self.progress_text_var.set("Translation failed")
                self._set_inline_feedback(
                    "Error",
                    "Translation failed.",
                    warning=(
                        "No translated subtitles were produced. "
                        f"Start {self.current_provider_name} and run the translation again."
                    ),
                )
                self._set_busy(False, self.status_var.get())
                return

            if fallback_count:
                warning_text = f"{fallback_count} lines could not be translated properly."
                self._set_inline_feedback(
                    "Completed",
                    f"Finished with {fallback_count} flagged lines.",
                    success="Translation finished with review flags.",
                    warning=warning_text,
                )
                self._set_busy(False, self.status_var.get())
                self._open_path(run_dir)
                return

            self._set_inline_feedback(
                "Completed",
                f"Output saved to {run_dir}",
                success="Translation finished successfully.",
            )
            self._set_busy(False, self.status_var.get())
            self._open_path(run_dir)
            return

        if event == "translation-error":
            self.progress_var.set(0)
            self.progress_text_var.set("Translation failed")
            self._set_inline_feedback("Error", "Translation failed.", warning=str(payload))
            self._set_busy(False, self.status_var.get())
            return

        if event == "runtime-info":
            runtime_payload = payload
            device = str(runtime_payload["device"]).strip()
            precision = str(runtime_payload["precision"]).lower()
            if device.upper().startswith("GPU"):
                label = device if "(" in device else f"GPU ({precision})"
                label = f"Running on {label}" if not label.startswith("Running on ") else label
            else:
                label = "Running on CPU"
            self.device_var.set(label)
            self._append_debug_line(f"[runtime] {label}")
            return

        if event == "batch-metric":
            metric_payload = payload
            language = str(metric_payload["language"]).upper()
            current_batch = int(metric_payload["current_batch"])
            total_batches = int(metric_payload["total_batches"])
            subtitle_count = int(metric_payload["subtitle_count"])
            elapsed_seconds = float(metric_payload["elapsed_seconds"])
            self.processed_subtitle_total += subtitle_count
            self.batch_time_var.set(
                f"Latest batch: {language} {current_batch}/{total_batches} in {elapsed_seconds:.2f}s"
            )
            self.processed_count_var.set(f"Subtitles processed: {self.processed_subtitle_total}")
            self._append_debug_line(
                f"[perf] {language} batch {current_batch}/{total_batches}: "
                f"{subtitle_count} subtitles in {elapsed_seconds:.2f}s"
            )
            return

        if event == "performance-summary":
            summary_payload = payload
            total_runtime = float(summary_payload["total_runtime"])
            average_batch_time = float(summary_payload["average_batch_time"])
            processed_subtitles = int(summary_payload["processed_subtitles"])
            self.total_time_var.set(
                f"Total time: {total_runtime:.2f}s (avg batch {average_batch_time:.2f}s)"
            )
            self.processed_count_var.set(f"Subtitles processed: {processed_subtitles}")
            self._append_debug_line(
                f"[perf] total runtime {total_runtime:.2f}s | avg batch {average_batch_time:.2f}s"
            )
            return

        if event == "debug-log":
            debug_payload = payload
            index = int(debug_payload["index"])
            source_text = " | ".join(
                line.strip()
                for line in str(debug_payload["source_text"]).splitlines()
                if line.strip()
            )
            translated_text = " | ".join(
                line.strip()
                for line in str(debug_payload["translated_text"]).splitlines()
                if line.strip()
            )
            self._append_debug_line(f"[{index}] INPUT: {source_text}")
            self._append_debug_line(f"[{index}] OUTPUT: {translated_text}")

    def _set_busy(self, busy: bool, status: str) -> None:
        self.status_var.set(status)
        self.translate_button.configure(state="disabled" if busy else "normal")
        self.translate_button.configure(text="Processing..." if busy else "Translate")

    def _resolve_subtitle_limit(self) -> int | None:
        raw_limit = self.limit_var.get().strip()
        if raw_limit:
            try:
                value = int(raw_limit)
            except ValueError as exc:
                raise ValueError("Limit must be a whole number.") from exc
            if value <= 0:
                raise ValueError("Limit must be greater than 0.")
            return value
        if self.test_mode_var.get():
            return 20
        return None

    def _reset_runtime_insights(self) -> None:
        self.processed_subtitle_total = 0
        self.status_state_var.set("Ready")
        self.device_var.set("Runtime: waiting")
        self.total_time_var.set("Total time: --")
        self.batch_time_var.set("Latest batch: --")
        self.processed_count_var.set("Subtitles processed: 0")
        self.success_var.set("")
        self.warning_var.set("")
        self._clear_debug_output()

    def _clear_debug_output(self) -> None:
        self.debug_text.configure(state="normal")
        self.debug_text.delete("1.0", tk.END)
        self.debug_text.configure(state="disabled")

    def _append_debug_line(self, line: str) -> None:
        self.debug_text.configure(state="normal")
        self.debug_text.insert(tk.END, f"{line}\n")
        self.debug_text.see(tk.END)
        self.debug_text.configure(state="disabled")

    @staticmethod
    def _open_path(path: Path) -> None:
        try:
            os.startfile(path)  # type: ignore[attr-defined]
            return
        except AttributeError:
            pass

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return
        subprocess.Popen(["xdg-open", str(path)])


def main() -> int:
    app = DesktopApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
