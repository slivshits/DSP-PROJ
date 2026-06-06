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

DRY_FILES = ["speechshort.wav", "casta.wav", "jimi-hendrix-on-classical-guitar-purple-haze-Jãomusico.wav", "piano-nqalab.wav"]
DEFAULTS = dict(gain=20.0, model="asym", neg_level=0.6, oversample=8, rh=0.995, rl=0.5)


def render_wet() -> None:
    WET.mkdir(parents=True, exist_ok=True)
    for name in DRY_FILES:
        src = DRY / name
        dst = WET / f"{Path(name).stem}_fuzz.wav"
        ff.process_file(str(src), str(dst), subtype="PCM_24", **DEFAULTS)
        print(f"  {src.name} -> {dst.relative_to(ROOT)}")


def fig_characteristic() -> None:
    plt.figure(figsize=(6, 5))
    for model in ("asym", "tube", "exp", "softclip", "hard"):
        x, y = ff.characteristic_curve(model=model, gain=DEFAULTS["gain"], neg_level=DEFAULTS["neg_level"])
        y = y / np.max(np.abs(y))  # normalise for visual comparison
        plt.plot(x, y, label=model)
    plt.axhline(0, color="k", lw=0.5)
    plt.axvline(0, color="k", lw=0.5)
    plt.title(f"Characteristic curves  y = f(gain*x),  gain={DEFAULTS['gain']:g}")
    plt.xlabel("input x")
    plt.ylabel("output y (normalised)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "characteristic_curve.png", dpi=130)
    plt.close()


def _sine(freq=300.0, sr=48000, dur=0.05):
    t = np.arange(int(sr * dur)) / sr
    return 0.9 * np.sin(2 * np.pi * freq * t), sr


def fig_waveform() -> None:
    x, sr = _sine()
    y = ff.process(x, sr, **DEFAULTS)
    n = int(sr * 0.02)
    t = np.arange(n) / sr * 220.0
    plt.figure(figsize=(7, 4))
    plt.plot(t, x[:n], label="dry input", lw=1.2)
    plt.plot(t, y[:n], label="fuzz output", lw=1.2)
    plt.title("Waveform before/after (220) Hz sine) - note asymmetric clipping")
    plt.xlabel("time (ms)")
    plt.ylabel("amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "waveform_before_after.png", dpi=130)
    plt.close()


def _spectrum_db(sig, sr, ref_peak=None):
    """Return (freqs, magnitude_dB). Normalise to ref_peak if given, else to own peak."""
    win = np.hanning(len(sig))
    spec = np.abs(np.fft.rfft(sig * win))
    peak = ref_peak if ref_peak is not None else (np.max(spec) + 1e-12)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    return freqs, 20 * np.log10(spec / peak + 1e-12)


def fig_spectrum() -> None:
    x, sr = _sine(freq=220.0, dur=0.5)
    y = ff.process(x, sr, **DEFAULTS)

    win = np.hanning(len(x))
    # Normalise both spectra to the dry signal's peak so the dry fundamental
    # sits at 0 dB and the fuzz harmonics appear relative to it.
    dry_peak = np.max(np.abs(np.fft.rfft(x * win))) + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    sx = 20 * np.log10(np.abs(np.fft.rfft(x * win)) / dry_peak + 1e-12)
    sy = 20 * np.log10(np.abs(np.fft.rfft(y * np.hanning(len(y)))) / dry_peak + 1e-12)

    plt.figure(figsize=(7, 4))
    # Draw fuzz first, then dry on top with a dashed line so both are visible.
    plt.plot(freqs, sy, label="fuzz output", lw=1.2, color="tab:orange", alpha=0.9)
    plt.plot(freqs, sx, label="dry input", lw=1.5, color="tab:blue",
             linestyle="--", zorder=5)
    plt.title("Spectrum before/after (220 Hz) — new even + odd harmonics")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude (dB, normalised to dry peak)")
    plt.xlim(0, 6000)
    plt.ylim(-90, 5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "spectrum_before_after.png", dpi=130)
    plt.close()


def fig_aliasing() -> None:
    # High, non-divisor fundamental + heavy drive: harmonics above Nyquist fold
    # back to clearly inharmonic frequencies, making aliasing obvious.
    x, sr = _sine(freq=1837.0, dur=0.3)
    params = dict(DEFAULTS)
    params.update(gain=60.0)
    y1 = ff.process(x, sr, **{**params, "oversample": 1})
    y8 = ff.process(x, sr, **{**params, "oversample": 8})
    f1, s1 = _spectrum_db(y1, sr)
    f8, s8 = _spectrum_db(y8, sr)
    plt.figure(figsize=(7, 4))
    plt.plot(f1, s1, label="oversample = 1 (aliased)", lw=1.0)
    plt.plot(f8, s8, label="oversample = 8", lw=1.0, alpha=0.85)
    plt.title("Aliasing reduction via oversampling (1837 Hz, heavy drive)")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude (dB)")
    plt.xlim(0, sr / 2)
    plt.ylim(-90, 5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "aliasing_oversampling.png", dpi=130)
    plt.close()


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
