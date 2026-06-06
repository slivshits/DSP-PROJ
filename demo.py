"""Render the Fuzz Face demonstration: wet audio for both dry samples + figures.

Outputs:
  samples/wet/<name>_fuzz.wav     wet renders with default settings
  figures/characteristic_curve.png
  figures/waveform_before_after.png
  figures/spectrum_before_after.png
  figures/aliasing_oversampling.png

Run:  python demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep matplotlib's cache inside the project (avoids non-writable HOME warnings).
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import fuzzface as ff

ROOT = Path(__file__).parent
DRY = ROOT / "samples" / "dry"
WET = ROOT / "samples" / "wet"
FIG = ROOT / "figures"

DRY_FILES = [
    "speechshort.wav",
    "casta.wav",
    "jimi-hendrix-on-classical-guitar-purple-haze-Jãomusico.wav",
    "piano-nqalab.wav",
]
DEFAULTS = dict(gain=20.0, model="asym", neg_level=0.6, oversample=8, rh=0.995, rl=0.5)

# ── Italian dark blue-turquoise palette ──────────────────────────────────────
C_NAVY     = "#1B2A4A"   # deep Italian navy
C_TEAL     = "#006D77"   # dark teal / turquoise
C_AQUA     = "#17A5B8"   # bright turquoise (accent)
C_MIST     = "#83C5BE"   # pale turquoise (light detail)
C_SLATE    = "#2E4057"   # mid blue-slate

# Model colours for the characteristic-curve figure
MODEL_COLORS = {
    "asym":     C_TEAL,
    "tube":     C_AQUA,
    "exp":      C_SLATE,
    "softclip": C_MIST,
    "hard":     C_NAVY,
}

def _style() -> None:
    """Apply the shared Italian dark blue-turquoise style to the current figure."""
    plt.rcParams.update({
        "figure.facecolor":  "#F4F8FB",
        "axes.facecolor":    "#EAF2F5",
        "axes.edgecolor":    C_NAVY,
        "axes.labelcolor":   C_NAVY,
        "xtick.color":       C_NAVY,
        "ytick.color":       C_NAVY,
        "text.color":        C_NAVY,
        "grid.color":        "#AACDD6",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  C_NAVY,
        "font.family":       "sans-serif",
    })


def render_wet() -> None:
    WET.mkdir(parents=True, exist_ok=True)
    for name in DRY_FILES:
        src = DRY / name
        if not src.exists():
            print(f"  skip (not found): {name}")
            continue
        dst = WET / f"{Path(name).stem}_fuzz.wav"
        ff.process_file(str(src), str(dst), subtype="PCM_24", **DEFAULTS)
        print(f"  {src.name} -> {dst.relative_to(ROOT)}")


def fig_characteristic() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for model in ("asym", "tube", "exp", "softclip", "hard"):
        x, y = ff.characteristic_curve(model=model, gain=DEFAULTS["gain"],
                                        neg_level=DEFAULTS["neg_level"])
        y = y / np.max(np.abs(y))
        ax.plot(x, y, label=model, color=MODEL_COLORS[model], lw=1.8)
    ax.axhline(0, color=C_NAVY, lw=0.6)
    ax.axvline(0, color=C_NAVY, lw=0.6)
    ax.set_title(f"Characteristic curves  y = f(gain·x),  gain={DEFAULTS['gain']:g}",
                 color=C_NAVY, fontweight="bold")
    ax.set_xlabel("input x")
    ax.set_ylabel("output y (normalised)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG / "characteristic_curve.png", dpi=130)
    plt.close(fig)


def _sine(freq=300.0, sr=48000, dur=0.05):
    t = np.arange(int(sr * dur)) / sr
    return 0.9 * np.sin(2 * np.pi * freq * t), sr


def fig_waveform() -> None:
    _style()
    x, sr = _sine()
    y = ff.process(x, sr, **DEFAULTS)
    n = int(sr * 0.02)
    t = np.arange(n) / sr * 1000.0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, x[:n], label="dry input",   color=C_NAVY, lw=1.6)
    ax.plot(t, y[:n], label="fuzz output", color=C_TEAL, lw=1.6, alpha=0.9)
    ax.set_title("Waveform before/after (300 Hz sine) — asymmetric clipping",
                 color=C_NAVY, fontweight="bold")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amplitude")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG / "waveform_before_after.png", dpi=130)
    plt.close(fig)


def _spectrum_db(sig, sr, ref_peak=None):
    """Return (freqs, magnitude_dB). Normalise to ref_peak if given, else to own peak."""
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    peak = ref_peak if ref_peak is not None else (np.max(spec) + 1e-12)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    return freqs, 20 * np.log10(spec / peak + 1e-12)


def fig_spectrum() -> None:
    _style()
    x, sr = _sine(freq=220.0, dur=0.5)
    y = ff.process(x, sr, **DEFAULTS)

    win = np.hanning(len(x))
    dry_peak = np.max(np.abs(np.fft.rfft(x * win))) + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    sx = 20 * np.log10(np.abs(np.fft.rfft(x * win)) / dry_peak + 1e-12)
    sy = 20 * np.log10(np.abs(np.fft.rfft(y * np.hanning(len(y)))) / dry_peak + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs, sy, label="fuzz output", color=C_TEAL,  lw=1.4, alpha=0.9)
    ax.plot(freqs, sx, label="dry input",   color=C_NAVY,  lw=1.8,
            linestyle="--", zorder=5)
    ax.set_title("Spectrum before/after (220 Hz) — even + odd harmonics",
                 color=C_NAVY, fontweight="bold")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB, normalised to dry peak)")
    ax.set_xlim(0, 6000)
    ax.set_ylim(-90, 5)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG / "spectrum_before_after.png", dpi=130)
    plt.close(fig)


def fig_aliasing() -> None:
    _style()
    # 48000 / 1837 ≈ 26.1  → aliased harmonics land at inharmonic positions
    x, sr = _sine(freq=1837.0, dur=0.3)
    params = dict(DEFAULTS)
    params.update(gain=60.0)
    y1 = ff.process(x, sr, **{**params, "oversample": 1})
    y8 = ff.process(x, sr, **{**params, "oversample": 8})
    f1, s1 = _spectrum_db(y1, sr)
    f8, s8 = _spectrum_db(y8, sr)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f1, s1, label="oversample = 1 (aliased)", color=C_SLATE, lw=1.0, alpha=0.85)
    ax.plot(f8, s8, label="oversample = 8",           color=C_AQUA,  lw=1.4)
    ax.set_title("Aliasing reduction via oversampling (1837 Hz, heavy drive)",
                 color=C_NAVY, fontweight="bold")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_xlim(0, sr / 2)
    ax.set_ylim(-90, 5)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIG / "aliasing_oversampling.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print("Rendering wet samples:")
    render_wet()
    print("Rendering figures:")
    fig_characteristic()
    fig_waveform()
    fig_spectrum()
    fig_aliasing()
    for f in sorted(FIG.glob("*.png")):
        print(f"  {f.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
