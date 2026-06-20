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
    """Process every dry sample through the Fuzz Face and save the result.

    Reads each file from ``samples/dry/``, applies the effect with the default
    settings, and writes a new file to ``samples/wet/`` with ``_fuzz`` appended
    to the filename. The output is saved as 24-bit WAV to match the quality of
    the original recordings.
    """
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
    """Plot the transfer (input → output) curves for every clipping model.

    A transfer curve shows what the waveshaper does to each possible input
    amplitude. A straight diagonal line would mean no distortion at all — the
    more the curve bends near the top and bottom, the harder the clipping and
    the richer the harmonic content. The key thing to notice is that the
    ``asym`` (Fuzz Face) curve bends *differently* on the positive and negative
    sides, which is what produces even-order harmonics and gives the pedal its
    distinctive warm, vocal sound.
    """
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
    """Plot a single sine wave before and after the Fuzz Face.

    Shows the effect in the time domain: the clean sine becomes asymmetrically
    clipped — the top of the wave is flattened more than the bottom (or vice
    versa depending on the model). This asymmetry is the root cause of the
    even-order harmonics. The figure also shows the slight DC shift that appears
    before the DC-blocking filter removes it, and the overall compression of
    the waveform's dynamic range (the peaks are limited, so quiet parts get
    proportionally louder).
    """
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
    """Plot the frequency spectrum of a sine wave before and after the Fuzz Face.

    A pure sine wave has a single spike in its spectrum — one frequency, no
    others. After the Fuzz Face, that single tone becomes a full harmonic series:
    spikes appear at every integer multiple of the original frequency (440 Hz,
    660 Hz, 880 Hz, …). Both even and odd multiples are present because the
    clipping is asymmetric. Both are normalised to the same reference
    (the dry signal's peak) so we can see exactly how strong each harmonic is
    relative to the original tone. The dry signal is drawn as a dashed line so
    it remains visible behind the fuzz output.
    """
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
    """Compare the spectrum with and without oversampling to show aliasing.

    When a nonlinearity is applied at the original sample rate (oversample=1),
    it creates harmonics that extend above the Nyquist frequency (half the
    sample rate). Those high harmonics can't be represented at the given sample
    rate and instead "fold back" into the audible range at random, inharmonic
    frequencies — this is aliasing, and it sounds like harsh, unpleasant noise.

    By choosing a fundamental (1837 Hz) that does NOT divide evenly into the
    sample rate (48000 / 1837 ≈ 26.1), the folded-back harmonics land at
    clearly inharmonic positions, making the aliasing visually obvious as a
    dense "grass" of spurious frequency components.

    With oversample=8 the nonlinearity runs at 8× the sample rate, so the first
    aliased harmonic appears above 8 × 24000 Hz = 192 kHz — far outside the
    audible range. The spectrum is clean.
    """
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
    """Run the full demonstration pipeline and save all outputs to disk.

    This is the single entry point for the demo. Running ``python demo.py``
    does the following steps in order:

    1. **Render wet audio** — applies the Fuzz Face effect (with default
       settings) to every dry sample in ``samples/dry/`` and saves the result
       as a 24-bit WAV in ``samples/wet/``. This gives you side-by-side audio
       files you can listen to and compare.

    2. **Characteristic curves** — saves ``figures/characteristic_curve.png``,
       showing how each clipping model distorts the signal amplitude.

    3. **Waveform before/after** — saves ``figures/waveform_before_after.png``,
       showing the time-domain shape of a sine wave before and after the effect.

    4. **Spectrum before/after** — saves ``figures/spectrum_before_after.png``,
       showing the new harmonics the fuzz adds in the frequency domain.

    5. **Aliasing comparison** — saves ``figures/aliasing_oversampling.png``,
       contrasting a heavily driven signal with and without oversampling to make
       the aliasing artefacts visible.

    All output files are overwritten if they already exist, so re-running
    ``demo.py`` always reflects the current state of ``fuzzface.py``.
    """
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
