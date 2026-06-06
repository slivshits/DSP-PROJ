"""Fuzz Face guitar fuzz effect.

A memoryless nonlinear waveshaper modelled on the classic Fuzz Face pedal
(asymmetric germanium-transistor clipping). The signal chain is:

    input -> input gain -> [oversample -> waveshaper -> downsample]
          -> DC-block high-pass -> tone low-pass -> output volume -> dry/wet mix

The default ``asym`` model uses the asymmetric soft-clipping nonlinearity from
Zoelzer's DAFX (after Bendiksen), which clips the two half-waves differently and
therefore produces both even- and odd-order harmonics -- the defining trait of
the Fuzz Face. Other models (exponential, symmetric three-region, hard clip) are
provided for comparison.

References:
  - U. Zoelzer (ed.), DAFX: Digital Audio Effects, 2nd ed., Wiley 2011, Ch. 4.
  - J. Reiss & A. McPherson, Audio Effects, CRC Press 2015, Ch. 7.
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

MODELS = ("asym", "tube", "exp", "softclip", "hard")

# Clamp arguments to exp() so very high drive does not overflow to inf. The
# nonlinearities saturate well before this, so the clamp does not change output.
_EXP_CLAMP = 50.0


def _exp(z: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(z, -_EXP_CLAMP, _EXP_CLAMP))


def asym_clip(q: np.ndarray, neg_level: float = 0.6) -> np.ndarray:
    """Asymmetric soft clipping -- the primary Fuzz Face model.

    Both half-waves are soft-clipped with tanh (giving sustain / a near-square
    shape under drive), but the negative half-wave saturates at a *lower* level
    than the positive one (``neg_level`` in 0..1). This reproduces the Fuzz
    Face's asymmetric clipping ("the negative clipping level is lower than the
    positive", DAFX Sec. 4.3.2) and therefore generates both even- and
    odd-order harmonics. DC offset is removed downstream by the high-pass.
    """
    q = np.asarray(q, dtype=np.float64)
    s = np.tanh(q)
    return np.where(s >= 0.0, s, s * neg_level)


def tube_clip(q: np.ndarray, Q: float = -0.2, dist: float = 8.0) -> np.ndarray:
    """Asymmetric soft clipping (DAFX Eq. 4.13, after Bendiksen) -- tube-style.

    f(x) = (x-Q)/(1 - e^{-dist*(x-Q)}) + Q/(1 - e^{dist*Q}),  x != Q
    with the removable 0/0 singularity at x == Q replaced by its limit.

    ``Q`` is the work point (more negative -> more linear at low level); ``dist``
    controls clipping hardness. Limits large negative inputs while staying nearly
    linear for positive inputs, so it is a milder, preamp-like asymmetry.
    """
    q = np.asarray(q, dtype=np.float64)
    eps = 1e-9
    if Q == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = q / (1.0 - _exp(-dist * q))
        limit = 1.0 / dist
        return np.where(np.abs(q) < eps, limit, z)
    offset = Q / (1.0 - _exp(dist * Q))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (q - Q) / (1.0 - _exp(-dist * (q - Q))) + offset
    limit = 1.0 / dist + offset
    return np.where(np.abs(q - Q) < eps, limit, z)


def exp_clip(q: np.ndarray) -> np.ndarray:
    """Symmetric exponential soft clipping (DAFX Eq. 4.15): sgn(x)(1 - e^{-|x|})."""
    q = np.asarray(q, dtype=np.float64)
    return np.sign(q) * (1.0 - _exp(-np.abs(q)))


def soft_clip(q: np.ndarray) -> np.ndarray:
    """Symmetric three-region soft clip (Schetzen / DAFX Eq. 4.14, 'overdrive')."""
    q = np.asarray(q, dtype=np.float64)
    a = np.abs(q)
    out = np.empty_like(q)
    lin = a < 1.0 / 3.0
    mid = (a >= 1.0 / 3.0) & (a < 2.0 / 3.0)
    sat = a >= 2.0 / 3.0
    out[lin] = 2.0 * q[lin]
    out[mid] = np.sign(q[mid]) * (3.0 - (2.0 - 3.0 * a[mid]) ** 2) / 3.0
    out[sat] = np.sign(q[sat])
    return out


def hard_clip(q: np.ndarray) -> np.ndarray:
    """Symmetric hard clip to [-1, 1]."""
    return np.clip(np.asarray(q, dtype=np.float64), -1.0, 1.0)


def waveshape(q: np.ndarray, model: str, Q: float, dist: float, neg_level: float) -> np.ndarray:
    if model == "asym":
        return asym_clip(q, neg_level=neg_level)
    if model == "tube":
        return tube_clip(q, Q=Q, dist=dist)
    if model == "exp":
        return exp_clip(q)
    if model == "softclip":
        return soft_clip(q)
    if model == "hard":
        return hard_clip(q)
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


def characteristic_curve(
    model: str = "asym",
    gain: float = 1.0,
    Q: float = -0.2,
    dist: float = 8.0,
    neg_level: float = 0.6,
    n: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, f(gain*x)) over x in [-1, 1] for plotting the transfer curve."""
    x = np.linspace(-1.0, 1.0, n)
    y = waveshape(gain * x, model, Q, dist, neg_level)
    return x, y


def _dc_block(y: np.ndarray, rh: float) -> np.ndarray:
    """Second-order DC-blocking high-pass: H(z)=(1-2z^-1+z^-2)/(1-2*rh z^-1+rh^2 z^-2)."""
    b = [1.0, -2.0, 1.0]
    a = [1.0, -2.0 * rh, rh * rh]
    return lfilter(b, a, y, axis=0)


def _tone_lp(y: np.ndarray, rl: float) -> np.ndarray:
    """First-order tone low-pass: H(z)=(1-rl)/(1-rl z^-1). Smaller rl -> brighter."""
    return lfilter([1.0 - rl], [1.0, -rl], y, axis=0)


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad ``y`` along axis 0 to exactly ``n`` samples."""
    if y.shape[0] == n:
        return y
    if y.shape[0] > n:
        return y[:n]
    pad = np.zeros((n - y.shape[0],) + y.shape[1:], dtype=y.dtype)
    return np.concatenate([y, pad], axis=0)


def process(
    x: np.ndarray,
    sr: int,
    gain: float = 20.0,
    model: str = "asym",
    Q: float = -0.2,
    dist: float = 8.0,
    neg_level: float = 0.6,
    oversample: int = 8,
    rh: float = 0.995,
    rl: float = 0.5,
    volume: float | None = None,
    mix: float = 1.0,
) -> np.ndarray:
    """Apply the Fuzz Face effect to a (n,) or (n, channels) float signal in [-1, 1].

    Returns a float64 array of the same shape, peak-limited to [-1, 1]. When
    ``volume`` is None the output peak is restored to the input peak; otherwise
    the (peak-normalised) output is scaled by ``volume``.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]

    peak_in = float(np.max(np.abs(x)))
    if peak_in == 0.0:
        return x[:, 0] if squeeze else x

    # --- nonlinear stage (optionally oversampled to suppress aliasing) ---
    xs = resample_poly(x, oversample, 1, axis=0) if oversample > 1 else x
    q = (xs / peak_in) * gain
    zs = waveshape(q, model, Q, dist, neg_level)
    z = resample_poly(zs, 1, oversample, axis=0) if oversample > 1 else zs
    z = _fit_length(z, n)

    # --- dry/wet mix (both normalised to unit peak) ---
    z_peak = float(np.max(np.abs(z)))
    wet = z / z_peak if z_peak > 0 else z
    dry = x / peak_in
    y = mix * wet + (1.0 - mix) * dry

    # --- post filtering: remove asymmetry-induced DC, then optional tone roll-off ---
    y = _dc_block(y, rh)
    if 0.0 < rl < 1.0:
        y = _tone_lp(y, rl)

    # --- output level ---
    out_peak = float(np.max(np.abs(y)))
    if out_peak > 0:
        y = y / out_peak
    y = y * (peak_in if volume is None else volume)
    y = np.clip(y, -1.0, 1.0)

    return y[:, 0] if squeeze else y


def process_file(input_path: str, output_path: str, subtype: str | None = "PCM_24", **params) -> None:
    """Read ``input_path``, apply the effect, and write to ``output_path``."""
    x, sr = sf.read(input_path, always_2d=True, dtype="float64")
    y = process(x, sr, **params)
    if y.ndim == 1:
        y = y[:, None]
    sf.write(output_path, y, sr, subtype=subtype)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fuzz Face effect: read an audio file, apply asymmetric fuzz, write the result.",
    )
    p.add_argument("input", help="input audio file (WAV)")
    p.add_argument("output", help="output audio file (WAV)")
    p.add_argument("--gain", type=float, default=20.0, help="input drive before the nonlinearity (default 20)")
    p.add_argument("--model", choices=MODELS, default="asym", help="nonlinearity model (default asym)")
    p.add_argument("--neg-level", dest="neg_level", type=float, default=0.6,
                   help="negative-half clip level for the asym model, 0..1 (default 0.6)")
    p.add_argument("--Q", type=float, default=-0.2, help="work point for the tube model (default -0.2)")
    p.add_argument("--dist", type=float, default=8.0, help="clipping hardness for the tube model (default 8)")
    p.add_argument("--oversample", type=int, default=8, help="oversampling factor N>=1 (default 8)")
    p.add_argument("--rh", type=float, default=0.995, help="DC-block high-pass pole, <1 (default 0.995)")
    p.add_argument("--rl", type=float, default=0.5, help="tone low-pass pole in 0..1; <=0 disables (default 0.5)")
    p.add_argument("--volume", type=float, default=None, help="output gain 0..1 (default: match input peak)")
    p.add_argument("--mix", type=float, default=1.0, help="dry/wet mix, 1=fully wet (default 1.0)")
    p.add_argument(
        "--subtype",
        default="PCM_24",
        help="output WAV subtype, e.g. PCM_16/PCM_24/FLOAT (default PCM_24)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    process_file(
        args.input,
        args.output,
        subtype=args.subtype,
        gain=args.gain,
        model=args.model,
        Q=args.Q,
        dist=args.dist,
        neg_level=args.neg_level,
        oversample=args.oversample,
        rh=args.rh,
        rl=args.rl,
        volume=args.volume,
        mix=args.mix,
    )
    print(f"Wrote {args.output} (model={args.model}, gain={args.gain}, oversample={args.oversample})")


if __name__ == "__main__":
    main()
