from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from translator.config import load_config
from translator.dictionary_store import (
    StoredDictionary,
    app_storage_dir,
    dictionary_path,
    import_dictionary,
    list_dictionaries,
)
from translator.models import LanguageArtifacts
from translator.pipeline import translate_project_with_artifacts


LANGUAGE_OPTIONS = [
    ("ur", "Urdu"),
    ("ar", "Arabic"),
    ("es", "Spanish"),
    ("id", "Indonesian"),
    ("tr", "Turkish"),
    ("fr", "French"),
    ("de", "German"),
    ("bn", "Bengali"),
    ("fa", "Persian"),
    ("ms", "Malay"),
]
STYLE_OPTIONS = ["literal", "balanced", "natural"]
PROVIDER_OPTIONS = ["manual", "mock", "ollama", "openai"]
THEME = {
    "bg": "#08171C",
    "panel": "#10252B",
    "panel_alt": "#16333A",
    "accent": "#38D7C7",
    "accent_soft": "#88F1DE",
    "gold": "#F4C542",
    "gold_soft": "#FFE38A",
    "text": "#F5FFFC",
    "muted": "#A9D5CF",
    "entry": "#0B1F24",
    "entry_border": "#24545D",
}


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", runtime_root()))


class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SRTranslate")
        self.geometry("1040x720")
        self.minsize(960, 680)
        self.configure(bg=THEME["bg"])

        self.runtime_dir = runtime_root()
        self.bundle_dir = bundle_root()
        self.storage_root = app_storage_dir()
        self.outputs_root = self.storage_root / "outputs" / "desktop_runs"
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.config_path = self.bundle_dir / "config.yaml"
        self.logo_path = self._resolve_logo_path()

        default_config = load_config(self.config_path)

        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.dictionary_records: list[StoredDictionary] = []
        self.current_output_dir: Path | None = None
        self.current_artifacts: dict[str, LanguageArtifacts] = {}
        self.window_icon: ImageTk.PhotoImage | None = None
        self.hero_image: ImageTk.PhotoImage | None = None

        self.srt_path_var = tk.StringVar()
        self.script_path_var = tk.StringVar()
        self.provider_var = tk.StringVar(value=default_config.provider)
        self.model_var = tk.StringVar(value=default_config.model)
        self.style_var = tk.StringVar(value=default_config.style_profile)
        self.review_mode_var = tk.BooleanVar(value=True)
        self.dictionary_var = tk.StringVar(value="None")
        self.status_var = tk.StringVar(value="Ready.")
        self.output_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Idle")

        self._configure_theme()
        self._load_branding()
        self._build_ui()
        self._refresh_dictionary_views()
        self._apply_provider_defaults()
        self.after(150, self._poll_events)

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
        style.configure("App.TLabel", background=THEME["bg"], foreground=THEME["text"])
        style.configure("Muted.TLabel", background=THEME["bg"], foreground=THEME["muted"])
        style.configure(
            "HeroTitle.TLabel",
            background=THEME["panel_alt"],
            foreground=THEME["gold_soft"],
            font=("Georgia", 24, "bold"),
        )
        style.configure(
            "HeroBody.TLabel",
            background=THEME["panel_alt"],
            foreground=THEME["muted"],
            font=("Segoe UI", 10),
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
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "TButton",
            background=THEME["accent"],
            foreground=THEME["bg"],
            borderwidth=0,
            focusthickness=0,
            focuscolor=THEME["accent"],
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", THEME["gold"]), ("disabled", THEME["entry_border"])],
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
            "TEntry",
            fieldbackground=THEME["entry"],
            foreground=THEME["text"],
            bordercolor=THEME["entry_border"],
            insertcolor=THEME["accent_soft"],
        )
        style.map("TEntry", bordercolor=[("focus", THEME["accent"])])
        style.configure(
            "TCombobox",
            fieldbackground=THEME["entry"],
            background=THEME["entry"],
            foreground=THEME["text"],
            arrowcolor=THEME["gold"],
            bordercolor=THEME["entry_border"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["entry"])],
            selectbackground=[("readonly", THEME["entry"])],
        )
        style.configure("TCheckbutton", background=THEME["panel"], foreground=THEME["text"])

        self.option_add("*TCombobox*Listbox.background", THEME["entry"])
        self.option_add("*TCombobox*Listbox.foreground", THEME["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", THEME["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", THEME["bg"])

    def _resolve_logo_path(self) -> Path | None:
        candidates = [
            self.bundle_dir / "assets" / "app_logo.png",
            self.runtime_dir / "assets" / "app_logo.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_branding(self) -> None:
        if not self.logo_path:
            return
        image = Image.open(self.logo_path)
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
        self._build_translate_panel(outer)

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Hero.TFrame", padding=18)
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
            text="Script-aware subtitle translation with glossary support and review exports.",
            style="HeroBody.TLabel",
            wraplength=760,
            justify="left",
        ).grid(row=1, column=1, sticky="nw", pady=(6, 0))

    def _build_translate_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="App.TFrame")
        panel.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(4, weight=1)

        files_frame = ttk.LabelFrame(panel, text="Files", padding=12)
        files_frame.grid(row=0, column=0, sticky="ew")
        files_frame.columnconfigure(1, weight=1)

        ttk.Label(files_frame, text="Subtitle file (.srt)").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(files_frame, textvariable=self.srt_path_var).grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=(0, 8))
        ttk.Button(files_frame, text="Browse...", command=self._choose_srt).grid(row=0, column=2, pady=(0, 8))

        ttk.Label(files_frame, text="Script file (.pdf, .txt, .md)").grid(row=1, column=0, sticky="w")
        ttk.Entry(files_frame, textvariable=self.script_path_var).grid(row=1, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(files_frame, text="Browse...", command=self._choose_script).grid(row=1, column=2)

        settings_frame = ttk.LabelFrame(panel, text="Settings", padding=12)
        settings_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(3, weight=1)

        ttk.Label(settings_frame, text="Provider").grid(row=0, column=0, sticky="w")
        provider_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.provider_var,
            values=PROVIDER_OPTIONS,
            state="readonly",
        )
        provider_combo.grid(row=0, column=1, sticky="ew", padx=(8, 16))
        provider_combo.bind("<<ComboboxSelected>>", lambda *_: self._apply_provider_defaults())

        ttk.Label(settings_frame, text="Model").grid(row=0, column=2, sticky="w")
        self.model_entry = ttk.Entry(settings_frame, textvariable=self.model_var)
        self.model_entry.grid(row=0, column=3, sticky="ew")

        ttk.Label(settings_frame, text="Style").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            settings_frame,
            textvariable=self.style_var,
            values=STYLE_OPTIONS,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(8, 16), pady=(10, 0))

        ttk.Label(settings_frame, text="Glossary").grid(row=1, column=2, sticky="w", pady=(10, 0))
        self.dictionary_combo = ttk.Combobox(settings_frame, textvariable=self.dictionary_var, state="readonly")
        self.dictionary_combo.grid(row=1, column=3, sticky="ew", pady=(10, 0))

        ttk.Checkbutton(
            settings_frame,
            text="Write review CSV",
            variable=self.review_mode_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

        glossary_actions = ttk.Frame(settings_frame, style="Panel.TFrame")
        glossary_actions.grid(row=2, column=2, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(glossary_actions, text="Refresh", command=self._refresh_dictionary_views).grid(row=0, column=0)
        ttk.Button(glossary_actions, text="Import Glossary", command=self._import_dictionary).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(glossary_actions, text="Open Folder", command=self._open_dictionary_folder).grid(row=0, column=2, padx=(8, 0))

        language_frame = ttk.LabelFrame(panel, text="Target Languages", padding=12)
        language_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        language_frame.columnconfigure(0, weight=1)

        self.language_list = tk.Listbox(
            language_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=6,
            bg=THEME["entry"],
            fg=THEME["text"],
            highlightbackground=THEME["entry_border"],
            highlightcolor=THEME["accent"],
            selectbackground=THEME["accent"],
            selectforeground=THEME["bg"],
            relief="flat",
        )
        for _, label in LANGUAGE_OPTIONS:
            self.language_list.insert(tk.END, label)
        self.language_list.grid(row=0, column=0, sticky="ew")
        language_scroll = ttk.Scrollbar(language_frame, orient="vertical", command=self.language_list.yview)
        language_scroll.grid(row=0, column=1, sticky="ns")
        self.language_list.configure(yscrollcommand=language_scroll.set)

        action_frame = ttk.Frame(panel, style="App.TFrame")
        action_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        action_frame.columnconfigure(1, weight=1)
        self.translate_button = ttk.Button(action_frame, text="Translate", command=self._start_translation)
        self.translate_button.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(
            action_frame,
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.grid(row=0, column=1, sticky="ew", padx=12)
        ttk.Button(action_frame, text="Open Outputs", command=self._open_output_folder).grid(row=0, column=2, sticky="e")

        status_frame = ttk.Frame(panel, style="App.TFrame")
        status_frame.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(1, weight=1)

        top_status = ttk.Frame(status_frame, style="App.TFrame")
        top_status.grid(row=0, column=0, sticky="ew")
        top_status.columnconfigure(1, weight=1)
        ttk.Label(top_status, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top_status, textvariable=self.progress_text_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")

        output_frame = ttk.LabelFrame(status_frame, text="Output Folder", padding=10)
        output_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.output_var, state="readonly").grid(row=0, column=0, sticky="ew")

        results_frame = ttk.LabelFrame(status_frame, text="Generated Files", padding=10)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.results_text = tk.Text(
            results_frame,
            height=12,
            wrap="word",
            state="disabled",
            bg=THEME["entry"],
            fg=THEME["text"],
            insertbackground=THEME["accent_soft"],
            relief="flat",
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

    def _choose_srt(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose subtitle file",
            filetypes=[("SRT files", "*.srt")],
        )
        if selected:
            self.srt_path_var.set(selected)

    def _choose_script(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose script file",
            filetypes=[("Script files", "*.pdf *.txt *.md"), ("All files", "*.*")],
        )
        if selected:
            self.script_path_var.set(selected)

    def _selected_language_codes(self) -> list[str]:
        selections = self.language_list.curselection()
        return [LANGUAGE_OPTIONS[index][0] for index in selections]

    def _apply_provider_defaults(self) -> None:
        provider = self.provider_var.get()
        if provider in {"manual", "mock"}:
            self.model_entry.configure(state="disabled")
            self.model_var.set(provider)
            return

        self.model_entry.configure(state="normal")
        if self.model_var.get() in {"manual", "mock", ""}:
            defaults = load_config(self.config_path)
            if provider == "openai":
                self.model_var.set(
                    str(defaults.raw.get("providers", {}).get("openai", {}).get("model", "gpt-5-mini"))
                )
            else:
                self.model_var.set(defaults.model)

    def _refresh_dictionary_views(self) -> None:
        self.dictionary_records = list_dictionaries(self.storage_root)
        names = ["None"] + [record.name for record in self.dictionary_records]
        self.dictionary_combo.configure(values=names)
        if self.dictionary_var.get() not in names:
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
            record = import_dictionary(path, None, base_dir=self.storage_root)
            self.event_queue.put(("dictionary-success", record))
        except Exception as exc:
            self.event_queue.put(("dictionary-error", str(exc)))

    def _open_dictionary_folder(self) -> None:
        self._open_path(self.storage_root / "dictionaries")

    def _open_output_folder(self) -> None:
        self._open_path(self.current_output_dir or self.outputs_root)

    def _start_translation(self) -> None:
        srt_path = self.srt_path_var.get().strip()
        script_path = self.script_path_var.get().strip()
        if not srt_path:
            messagebox.showerror("Missing subtitle file", "Choose an SRT subtitle file.")
            return
        if not script_path:
            messagebox.showerror("Missing script file", "Choose a script file.")
            return

        languages = self._selected_language_codes()
        if not languages:
            messagebox.showerror("Missing languages", "Select at least one target language.")
            return

        dictionary_record = self._selected_dictionary()
        glossary_path = str(dictionary_path(dictionary_record, self.storage_root)) if dictionary_record else None

        self._clear_text_widget(self.results_text)
        self.output_var.set("")
        self.current_output_dir = None
        self.current_artifacts = {}
        self.progress_var.set(0)
        self.progress_text_var.set("Starting...")
        self._set_busy(True, "Running translation...")

        threading.Thread(
            target=self._translation_worker,
            args=(
                srt_path,
                script_path,
                languages,
                glossary_path,
                self.provider_var.get(),
                self.model_var.get().strip(),
                self.style_var.get(),
                self.review_mode_var.get(),
            ),
            daemon=True,
        ).start()

    def _translation_worker(
        self,
        srt_path: str,
        script_path: str,
        languages: list[str],
        glossary_path: str | None,
        provider: str,
        model: str,
        style: str,
        review_mode: bool,
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

        try:
            config = load_config(self.config_path)
            config.raw["provider"] = provider
            if model:
                config.raw["model"] = model
            config.raw["style_profile"] = style
            run_dir = self.outputs_root / datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            config.raw.setdefault("output", {})
            config.raw["output"]["output_dir"] = str(run_dir)

            artifacts = translate_project_with_artifacts(
                srt_path=srt_path,
                script_path=script_path,
                langs=languages,
                config=config,
                glossary_path=glossary_path,
                profile=style,
                review_mode=review_mode,
                progress_callback=report_progress,
            )
            self.event_queue.put(("translation-success", (run_dir, artifacts)))
        except Exception as exc:
            self.event_queue.put(("translation-error", str(exc)))

    def _selected_dictionary(self) -> StoredDictionary | None:
        selected_name = self.dictionary_var.get()
        if selected_name == "None":
            return None
        for record in self.dictionary_records:
            if record.name == selected_name:
                return record
        return None

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
            self._refresh_dictionary_views()
            self.dictionary_var.set(record.name)
            self._set_busy(False, f"Glossary '{record.name}' is ready.")
            return

        if event == "dictionary-error":
            self._set_busy(False, "Glossary import failed.")
            messagebox.showerror("Glossary error", str(payload))
            return

        if event == "progress":
            current = int(payload["current"])
            total = max(1, int(payload["total"]))
            message = str(payload["message"])
            percent = round((current / total) * 100, 1)
            self.progress_var.set(percent)
            self.progress_text_var.set(f"{percent:.1f}%")
            self.status_var.set(message)
            return

        if event == "translation-success":
            run_dir, artifacts = payload
            self.current_output_dir = run_dir
            self.current_artifacts = artifacts
            self.output_var.set(str(run_dir))
            self._append_results(artifacts)
            self.progress_var.set(100)
            self.progress_text_var.set("100%")
            self._set_busy(False, f"Translation complete. Output saved to {run_dir}")
            messagebox.showinfo("Translation complete", f"Output saved to:\n{run_dir}")
            return

        if event == "translation-error":
            self.progress_text_var.set("Failed")
            self._set_busy(False, "Translation failed.")
            messagebox.showerror("Translation error", str(payload))

    def _append_results(self, artifacts: dict[str, LanguageArtifacts]) -> None:
        lines: list[str] = []
        for language, artifact in artifacts.items():
            lines.extend(
                [
                    f"{language.upper()}",
                    f"SRT: {artifact.srt_path}",
                    f"Report: {artifact.report_path}",
                    f"Flags: {artifact.flags_path}",
                ]
            )
            if artifact.review_path:
                lines.append(f"Review: {artifact.review_path}")
            lines.append("")
        self._append_text(self.results_text, "\n".join(lines).strip())

    def _set_busy(self, busy: bool, status: str) -> None:
        self.status_var.set(status)
        if busy:
            self.translate_button.configure(state="disabled")
        else:
            self.translate_button.configure(state="normal")
            if self.progress_var.get() <= 0:
                self.progress_text_var.set("Idle")

    @staticmethod
    def _append_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text + ("" if text.endswith("\n") else "\n"))
        widget.see(tk.END)
        widget.configure(state="disabled")

    @staticmethod
    def _clear_text_widget(widget: tk.Text) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.configure(state="disabled")

    @staticmethod
    def _open_path(path: Path) -> None:
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            subprocess.Popen(["xdg-open", str(path)])


def main() -> int:
    app = DesktopApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
