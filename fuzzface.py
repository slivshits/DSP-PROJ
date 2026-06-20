"""Fuzz Face guitar fuzz effect.

A memoryless nonlinear waveshaper modelled on the classic Fuzz Face pedal
(asymmetric germanium-transistor clipping). The default 'asym' model clips
the two half-waves differently, producing both even- and odd-order harmonics
-- the defining trait of the Fuzz Face.

References:
  U. Zoelzer (ed.), DAFX: Digital Audio Effects, 2nd ed., Wiley 2011, Ch. 4.
  J. Reiss & A. McPherson, Audio Effects, CRC Press 2015, Ch. 7.
"""

import argparse
import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

MODELS = ("asym", "tube", "exp", "softclip", "hard")

# exp() overflows above ~710; clamp at 50 since all curves saturate well before that.
_EXP_CLAMP = 50.0


def _exp(z):
    # plain np.exp overflows to inf for large arguments, which breaks the tube
    # and exp models at high drive. clamping at ±50 is safe because all curves
    # are fully saturated long before that point.
    return np.exp(np.clip(z, -_EXP_CLAMP, _EXP_CLAMP))


def asym_clip(q, neg_level=0.6):
    """Asymmetric tanh soft-clip — primary Fuzz Face model.

    Both halves are soft-clipped with tanh, but the negative half saturates at
    neg_level (< 1) of the positive peak. This asymmetry generates even-order
    harmonics. The resulting DC offset is removed downstream by the high-pass.
    """
    s = np.tanh(q)
    return np.where(s >= 0, s, s * neg_level)


def tube_clip(q, Q=-0.2, dist=8.0):
    """Asymmetric soft clipping (DAFX Eq. 4.13) — tube/valve style.

    f(x) = (x-Q) / (1 - exp(-dist*(x-Q))) + offset,  x != Q

    Q is the bias point (more negative = more linear at low level); dist sets
    clipping hardness. The 0/0 singularity at x == Q is replaced by its limit.
    """
    eps = 1e-9

    if Q == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = q / (1 - _exp(-dist * q))
        return np.where(np.abs(q) < eps, 1 / dist, z)

    offset = Q / (1 - _exp(dist * Q))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (q - Q) / (1 - _exp(-dist * (q - Q))) + offset
    return np.where(np.abs(q - Q) < eps, 1 / dist + offset, z)


def exp_clip(q):
    """Symmetric exponential soft clip (DAFX Eq. 4.15): sgn(x)(1 - e^{-|x|})."""
    return np.sign(q) * (1 - _exp(-np.abs(q)))


def soft_clip(q):
    """Symmetric three-region soft clip (Schetzen / DAFX Eq. 4.14).

    |x| < 1/3   — linear (gain 2)
    1/3..2/3    — smooth polynomial transition
    |x| >= 2/3  — hard limit at ±1
    """
    a = np.abs(q)
    out = np.empty_like(q)
    lin = a < 1/3
    mid = (a >= 1/3) & (a < 2/3)
    sat = a >= 2/3
    out[lin] = 2 * q[lin]
    out[mid] = np.sign(q[mid]) * (3 - (2 - 3*a[mid])**2) / 3
    out[sat] = np.sign(q[sat])
    return out


def hard_clip(q):
    """Symmetric hard clip to [-1, 1]."""
    return np.clip(q, -1, 1)


def waveshape(q, model, Q, dist, neg_level):
    """Dispatch to the selected clipping function. q is already gain-scaled."""
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


def characteristic_curve(model="asym", gain=1.0, Q=-0.2, dist=8.0, neg_level=0.6, n=1024):
    """Return (x, f(gain*x)) over x in [-1, 1] for plotting the transfer curve."""
    x = np.linspace(-1, 1, n)
    return x, waveshape(gain * x, model, Q, dist, neg_level)


def _dc_block(y, rh):
    """2nd-order DC-blocking high-pass: H(z) = (1-z^-1)^2 / (1-rh*z^-1)^2."""
    b = [1, -2, 1]
    a = [1, -2*rh, rh*rh]
    return lfilter(b, a, y, axis=0)


def _tone_lp(y, rl):
    """1st-order tone low-pass: H(z) = (1-rl)/(1-rl*z^-1). Higher rl = darker."""
    return lfilter([1 - rl], [1, -rl], y, axis=0)


def _fit_length(y, n):
    """Trim or zero-pad to n samples — resample_poly can be off by one at boundaries."""
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
    """Apply the Fuzz Face signal chain and return a float64 array clipped to [-1, 1].

    Runs the full pedal chain: normalise input level, oversample, push through
    the clipping curve, downsample back, blend with dry signal, remove DC offset,
    tone-shape, then restore the original loudness.

    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = x.shape[0]

    peak_in = np.max(np.abs(x))
    if peak_in == 0:
        return x[:, 0] if squeeze else x

    # upsample then waveshape at N*fs then downsample
    xs = resample_poly(x, oversample, 1, axis=0) if oversample > 1 else x
    q = (xs / peak_in) * gain
    zs = waveshape(q, model, Q, dist, neg_level)
    z = resample_poly(zs, 1, oversample, axis=0) if oversample > 1 else zs
    z = _fit_length(z, n)

    # blend (normalise both so mix ratio is independent of clipping depth)
    z_peak = np.max(np.abs(z))
    wet = z / z_peak if z_peak > 0 else z
    dry = x / peak_in
    y = mix * wet + (1 - mix) * dry

    y = _dc_block(y, rh)
    if 0 < rl < 1:
        y = _tone_lp(y, rl)

    # restore loudness
    out_peak = np.max(np.abs(y))
    if out_peak > 0:
        y = y / out_peak
    y = y * (peak_in if volume is None else volume)
    y = np.clip(y, -1, 1)

    return y[:, 0] if squeeze else y


def process_file(input_path: str, output_path: str, subtype: str | None = "PCM_24", **params) -> None:
    """Read a WAV, apply the effect, write the result. Preserves sample rate and channels."""
    x, sr = sf.read(input_path, always_2d=True, dtype="float64")
    y = process(x, sr, **params)
    if y.ndim == 1:
        y = y[:, None]
    sf.write(output_path, y, sr, subtype=subtype)


def _build_parser():
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
    p.add_argument("--volume", type=float, default=None, help="output peak 0..1 (default: match input)")
    p.add_argument("--mix", type=float, default=1.0, help="dry/wet mix, 1=fully wet (default 1.0)")
    p.add_argument("--subtype", default="PCM_24",
                   help="output WAV subtype: PCM_16/PCM_24/FLOAT (default PCM_24)")
    return p


def main(argv=None):
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
