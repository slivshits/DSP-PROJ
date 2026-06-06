"""Fuzz Face — desktop GUI.

A small Tkinter front-end for the Fuzz Face effect (see fuzzface.py). The user
picks an input audio file, adjusts the fuzz parameters with sliders, renders the
processed file, and can play the dry input / wet output. Packaged into a
standalone app for macOS and Windows with PyInstaller (see build/ and README).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    BooleanVar,
    DoubleVar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from tkinter import ttk

import fuzzface as ff

MODELS = list(ff.MODELS)
OVERSAMPLE_CHOICES = ["1", "2", "4", "8", "16"]
AUDIO_TYPES = [("Audio files", "*.wav *.aif *.aiff *.flac *.ogg"), ("All files", "*.*")]


def play_file(path: str) -> "subprocess.Popen | None":
    """Play an audio file with the OS default player. Returns a handle if any."""
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.Popen(["afplay", path])
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
            return None
        return subprocess.Popen(["xdg-open", path])
    except Exception as exc:  # pragma: no cover - playback is best-effort
        messagebox.showwarning("Playback", f"Could not play file:\n{exc}")
        return None


class FuzzFaceApp(ttk.Frame):
    def __init__(self, master: Tk):
        super().__init__(master, padding=14)
        master.title("Fuzz Face")
        master.minsize(560, 620)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._player: "subprocess.Popen | None" = None
        self._busy = False

        # --- state variables (defaults mirror fuzzface.process) ---
        self.input_path = StringVar()
        self.output_path = StringVar()
        self.model = StringVar(value="asym")
        self.gain = DoubleVar(value=20.0)
        self.neg_level = DoubleVar(value=0.6)
        self.Q = DoubleVar(value=-0.2)
        self.dist = DoubleVar(value=8.0)
        self.oversample = StringVar(value="8")
        self.rh = DoubleVar(value=0.995)
        self.rl = DoubleVar(value=0.5)
        self.mix = DoubleVar(value=1.0)
        self.auto_volume = BooleanVar(value=True)
        self.volume = DoubleVar(value=0.8)
        self.status = StringVar(value="Choose an input file to begin.")

        self._build()
        self._on_model_change()

    # ---------------------------------------------------------------- UI build
    def _build(self) -> None:
        ttk.Label(self, text="Fuzz Face", font=("", 20, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Label(
            self, text="Asymmetric guitar fuzz \u2014 choose a file and shape the sound.",
            foreground="#666",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        self._build_files(row=2)
        self._build_params(row=3)
        self._build_actions(row=4)

        status = ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=6)
        status.grid(row=5, column=0, sticky="ew", pady=(12, 0))

    def _build_files(self, row: int) -> None:
        box = ttk.LabelFrame(self, text="Files", padding=10)
        box.grid(row=row, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Input").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(box, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(box, text="Browse\u2026", command=self._pick_input).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(box, text="Output").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(box, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(box, text="Browse\u2026", command=self._pick_output).grid(row=1, column=2, padx=(8, 0))

    def _slider(self, parent, row, label, var, lo, hi, fmt="{:.2f}"):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        val = ttk.Label(parent, width=7, anchor="e")
        val.grid(row=row, column=2, sticky="e", padx=(8, 0))

        def update(_=None):
            val.config(text=fmt.format(var.get()))

        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, command=update)
        scale.grid(row=row, column=1, sticky="ew", padx=8, pady=3)
        update()
        return scale

    def _build_params(self, row: int) -> None:
        box = ttk.LabelFrame(self, text="Effect", padding=10)
        box.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Model").grid(row=0, column=0, sticky="w", pady=3)
        model_cb = ttk.Combobox(box, textvariable=self.model, values=MODELS, state="readonly", width=12)
        model_cb.grid(row=0, column=1, sticky="w", padx=8, pady=3)
        model_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_model_change())

        self._slider(box, 1, "Gain (drive)", self.gain, 1.0, 60.0, "{:.1f}")
        self.s_neg = self._slider(box, 2, "Neg. clip (asym)", self.neg_level, 0.0, 1.0)
        self.s_Q = self._slider(box, 3, "Work point Q (tube)", self.Q, -1.0, 0.0)
        self.s_dist = self._slider(box, 4, "Hardness (tube)", self.dist, 1.0, 20.0, "{:.1f}")

        ttk.Label(box, text="Oversample").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(
            box, textvariable=self.oversample, values=OVERSAMPLE_CHOICES, state="readonly", width=6
        ).grid(row=5, column=1, sticky="w", padx=8, pady=3)

        self._slider(box, 6, "DC block (rh)", self.rh, 0.90, 0.9999, "{:.4f}")
        self._slider(box, 7, "Tone LP (rl)", self.rl, 0.0, 0.95)
        self._slider(box, 8, "Dry/Wet mix", self.mix, 0.0, 1.0)

        vol_row = ttk.Frame(box)
        vol_row.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        vol_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            vol_row, text="Auto output level", variable=self.auto_volume,
            command=self._on_volume_toggle,
        ).grid(row=0, column=0, sticky="w")
        self.s_vol = self._slider(vol_row, 1, "Volume", self.volume, 0.0, 1.0)

    def _build_actions(self, row: int) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=row, column=0, sticky="ew", pady=(14, 0))
        for i in range(4):
            bar.columnconfigure(i, weight=1)
        ttk.Button(bar, text="Play input", command=self._play_input).grid(row=0, column=0, sticky="ew", padx=2)
        self.render_btn = ttk.Button(bar, text="Apply fuzz \u2192 render", command=self._render)
        self.render_btn.grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(bar, text="Play output", command=self._play_output).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(bar, text="Reset", command=self._reset).grid(row=0, column=3, sticky="ew", padx=2)

    # ---------------------------------------------------------------- handlers
    def _on_model_change(self) -> None:
        is_asym = self.model.get() == "asym"
        is_tube = self.model.get() == "tube"
        self.s_neg.state(["!disabled"] if is_asym else ["disabled"])
        for w in (self.s_Q, self.s_dist):
            w.state(["!disabled"] if is_tube else ["disabled"])

    def _on_volume_toggle(self) -> None:
        self.s_vol.state(["disabled"] if self.auto_volume.get() else ["!disabled"])

    def _suggest_output(self, in_path: str) -> str:
        p = Path(in_path)
        return str(p.with_name(f"{p.stem}_fuzz.wav"))

    def _pick_input(self) -> None:
        path = filedialog.askopenfilename(title="Choose input audio", filetypes=AUDIO_TYPES)
        if not path:
            return
        self.input_path.set(path)
        if not self.output_path.get():
            self.output_path.set(self._suggest_output(path))
        self.status.set(f"Loaded: {Path(path).name}")

    def _pick_output(self) -> None:
        initial = self.output_path.get() or (
            self._suggest_output(self.input_path.get()) if self.input_path.get() else "output_fuzz.wav"
        )
        path = filedialog.asksaveasfilename(
            title="Save processed audio as", defaultextension=".wav",
            initialfile=Path(initial).name, filetypes=[("WAV", "*.wav")],
        )
        if path:
            self.output_path.set(path)

    def _play_input(self) -> None:
        self._play(self.input_path.get(), "input")

    def _play_output(self) -> None:
        self._play(self.output_path.get(), "output")

    def _play(self, path: str, which: str) -> None:
        if not path or not os.path.exists(path):
            messagebox.showinfo("Play", f"No {which} file to play yet.")
            return
        if self._player and self._player.poll() is None:
            self._player.terminate()
        self._player = play_file(path)

    def _reset(self) -> None:
        self.model.set("asym")
        self.gain.set(20.0)
        self.neg_level.set(0.6)
        self.Q.set(-0.2)
        self.dist.set(8.0)
        self.oversample.set("8")
        self.rh.set(0.995)
        self.rl.set(0.5)
        self.mix.set(1.0)
        self.auto_volume.set(True)
        self.volume.set(0.8)
        self._build()  # rebuild to refresh slider value labels
        self._on_model_change()
        self.status.set("Reset to defaults.")

    # ---------------------------------------------------------------- render
    def _render(self) -> None:
        if self._busy:
            return
        in_path = self.input_path.get().strip()
        out_path = self.output_path.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showerror("Input", "Please choose a valid input file.")
            return
        if not out_path:
            out_path = self._suggest_output(in_path)
            self.output_path.set(out_path)

        params = dict(
            gain=float(self.gain.get()),
            model=self.model.get(),
            Q=float(self.Q.get()),
            dist=float(self.dist.get()),
            neg_level=float(self.neg_level.get()),
            oversample=int(self.oversample.get()),
            rh=float(self.rh.get()),
            rl=float(self.rl.get()),
            mix=float(self.mix.get()),
            volume=None if self.auto_volume.get() else float(self.volume.get()),
        )

        self._busy = True
        self.render_btn.config(text="Processing\u2026", state="disabled")
        self.status.set("Processing\u2026")
        threading.Thread(target=self._render_worker, args=(in_path, out_path, params), daemon=True).start()

    def _render_worker(self, in_path: str, out_path: str, params: dict) -> None:
        try:
            ff.process_file(in_path, out_path, **params)
        except Exception as exc:
            self.after(0, self._render_failed, exc)
        else:
            self.after(0, self._render_done, out_path)

    def _render_done(self, out_path: str) -> None:
        self._busy = False
        self.render_btn.config(text="Apply fuzz \u2192 render", state="normal")
        self.status.set(f"Done \u2192 {out_path}")

    def _render_failed(self, exc: Exception) -> None:
        self._busy = False
        self.render_btn.config(text="Apply fuzz \u2192 render", state="normal")
        self.status.set("Error during processing.")
        messagebox.showerror("Processing failed", str(exc))


def main() -> None:
    root = Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    FuzzFaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
