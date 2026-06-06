# The Fuzz Face Effect — Implementation Report

**Course:** Digital Signal Processing (0512.4200), Semester A 2025–2026
**Assigned effect:** Fuzz Face

---

## 1. What the effect does (the sound)

The Fuzz Face is a classic guitar **fuzz** pedal — the aggressive, saturated tone
heard on much of Jimi Hendrix's playing. Perceptually it transforms a clean,
dynamic input into a thick, buzzy, *sustaining* tone. Two qualities define it:

- **Sustain / compression.** The effect clips the waveform toward a near-square
  shape, so quiet parts are lifted and loud parts are limited. Notes appear to
  "bloom" and ring on far longer than the dry signal. Measured on the supplied
  samples, the crest factor (peak-to-RMS) collapses from 18.4 dB to 7.2 dB for
  `speechshort` and from 30.2 dB to 16.4 dB for `casta` — i.e. the dynamics are
  strongly compressed, which the ear hears as sustain and "thickness".
- **Harmonic richness with a vocal character.** Because the clipping is
  **asymmetric**, the output contains both **even- and odd-order harmonics**
  (Figure 3). The even harmonics give the Fuzz Face its slightly hollow, vocal
  colour that distinguishes it from purely symmetric distortions. It works best
  on single notes; on full chords the same nonlinearity also produces
  inharmonic intermodulation tones, which is why fuzz is most often used for
  lead lines and power chords.

## 2. How it works (the DSP)

### Signal flow

```
x -> input gain g -> [ upsample xN -> nonlinear waveshaper -> downsample xN ]
  -> DC-block high-pass -> tone low-pass -> output level -> dry/wet mix -> y
```

**Nonlinear waveshaping.** The heart of any distortion/fuzz is a *memoryless*
nonlinearity `y[n] = f(g·x[n])`: each output sample depends only on the current
input sample. Unlike linear effects (filters, delays), a nonlinearity creates
**new** frequency components that were not present in the input. The input gain
`g` sets how hard the signal is pushed into the nonlinear region — more gain
means more distortion and more sustain, while the output level barely changes
because the curve saturates.

**Asymmetric clipping (the Fuzz Face model).** The implemented default model
soft-clips both half-waves with a `tanh` curve, but saturates the **negative**
half-wave at a lower level than the positive one (parameter `neg_level`,
default 0.6):

```
f(x) = tanh(g·x)            if tanh(g·x) ≥ 0
       neg_level · tanh(g·x) if tanh(g·x) < 0
```

This mirrors the real circuit, where "the negative clipping level is lower than
the positive clipping value" (Zölzer, *DAFX*, §4.3.2). The asymmetry is exactly
what produces **even** harmonics: an odd-symmetric function `f(−x) = −f(x)` can
only generate odd harmonics, so breaking that symmetry is required to obtain the
even harmonics audible in Figure 3. The transfer curve is shown in Figure 1
(`asym`), alongside the DAFX asymmetric tube curve (Eq. 4.13, available as the
`tube` model) and symmetric references.

![Characteristic curves](figures/characteristic_curve.png)
*Figure 1 — Characteristic (transfer) curves. The `asym` Fuzz Face curve clips
the positive half-wave at +1 and the negative at −0.6; symmetric models clip
both equally.*

**Post-filtering.** Asymmetric clipping adds a DC offset, which is removed by a
second-order **DC-blocking high-pass** `H(z) = (1−2z⁻¹+z⁻²)/(1−2·rh·z⁻¹+rh²·z⁻²)`
(`rh = 0.995`), reproducing the AC coupling of the analog pedal. A first-order
**tone low-pass** `H(z) = (1−rl)/(1−rl·z⁻¹)` then tames the harshest highs. The
combined AC-coupling gives the output its characteristic tilted/charge-discharge
shape (Figure 2).

![Waveform before/after](figures/waveform_before_after.png)
*Figure 2 — A 220 Hz sine before and after the effect. The output is clipped
asymmetrically (positive and negative peaks limited differently) and AC-coupled.*

**Harmonic generation.** Figure 3 shows the spectrum of a 220 Hz sine before and
after processing: the single input tone becomes a full harmonic series at every
multiple of 220 Hz, with both even and odd harmonics present — the spectral
signature of asymmetric distortion.

![Spectrum before/after](figures/spectrum_before_after.png)
*Figure 3 — Spectrum before/after. New harmonics appear at all integer multiples
of the fundamental (even + odd), confirming asymmetric nonlinear distortion.*

**Aliasing and oversampling.** A nonlinearity generates *infinitely many*
harmonics; any harmonic above the Nyquist frequency aliases back into the
audible band as inharmonic, non-removable noise. To suppress this, the signal is
**oversampled** by `N×` (default 8) before the waveshaper and downsampled
afterwards (polyphase resampling with the built-in anti-alias filters). Figure 4
contrasts `N=1` and `N=8` for a heavily driven 1837 Hz tone: without
oversampling the spectrum is filled with a dense "grass" of aliased components,
while 8× oversampling leaves a clean harmonic series.

![Aliasing vs oversampling](figures/aliasing_oversampling.png)
*Figure 4 — Heavy drive on a non-divisor fundamental. `oversample=1` (blue)
produces dense inharmonic aliasing; `oversample=8` (orange) is clean.*

## 3. Implementation choices

- **Language / libraries:** Python with NumPy (vectorised waveshaping),
  SciPy (`resample_poly` for oversampling, `lfilter` for the filters), and
  `soundfile` for 24-bit WAV I/O. The DSP is a pure function (`process`)
  independent of file I/O.
- **Model and parameters:** the default is the asymmetric `tanh` clip with
  `gain = 20`, `neg_level = 0.6` — chosen to give an obviously fuzzy, sustaining
  tone with clear even harmonics on the supplied samples. The DAFX Eq. 4.13
  asymmetric curve is also implemented (`--model tube`); it gives a milder,
  preamp-like asymmetry because it leaves the positive half-wave nearly linear,
  which is why a dedicated bounded-asymmetric curve was chosen as the primary
  Fuzz Face model.
- **Normalisation:** the input is normalised by its peak before the gain stage
  so the `gain` control behaves consistently regardless of input level; the
  output is peak-normalised and then restored to the input's peak level (or
  scaled by `--volume`) and hard-limited to [−1, 1] to prevent output clipping.
- **Oversampling factor:** 8× by default — a good trade-off between alias
  suppression and CPU cost; selectable via `--oversample` (use `1` to hear the
  aliasing for comparison).
- **Robustness:** the `tube` model's 0/0 singularity at the work point is
  replaced by its analytic limit; `exp()` arguments are clamped to avoid
  overflow at very high drive; mono and stereo inputs and silent inputs are all
  handled, and the input sample rate / bit depth are preserved on output.

## References

1. U. Zölzer (ed.), *DAFX: Digital Audio Effects*, 2nd ed., Wiley, 2011 —
   Ch. 4 "Nonlinear processing", §4.3.1–4.3.2 (valve simulation, overdrive /
   distortion / fuzz, Fuzz Face; Eqs. 4.13–4.15).
2. J. D. Reiss & A. P. McPherson, *Audio Effects: Theory, Implementation and
   Application*, CRC Press, 2015 — Ch. 7 "Overdrive, Distortion, and Fuzz"
   (characteristic curves, hard/soft clipping, symmetry & harmonics, aliasing
   and oversampling, filtering).
