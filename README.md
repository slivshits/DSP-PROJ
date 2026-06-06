# Fuzz Face — DSP Special Assignment (0512.4200)

A digital implementation of the **Fuzz Face** guitar fuzz effect: an asymmetric,
high-gain nonlinear waveshaper with oversampling (anti-aliasing), a DC-blocking
high-pass, and a tone low-pass. Reads an audio file and writes the processed
("wet") file.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Basic: apply the Fuzz Face with default settings
python fuzzface.py samples/dry/casta.wav samples/wet/casta_fuzz.wav

# Tweak the sound
python fuzzface.py input.wav output.wav --gain 25 --model asym --oversample 8 \
    --neg-level 0.6 --rl 0.5 --mix 1.0
```

Reproduce the submitted wet samples and the report figures in one step:

```bash
python demo.py
```

This renders `samples/wet/speechshort_fuzz.wav` and `samples/wet/casta_fuzz.wav`
(default settings) and writes the figures in `figures/`.

## Desktop app (GUI)

A graphical version (`gui.py`) lets you pick an input file, adjust the fuzz with
sliders (model, gain, asymmetry, oversampling, DC-block, tone, mix, volume),
render, and play the input/output. Run it from source:

```bash
python gui.py          # requires a Python with Tk (tkinter) available
```

### Standalone executables (macOS .app / Windows .exe)

The app is packaged with [PyInstaller](https://pyinstaller.org). Because
PyInstaller cannot cross-compile, each OS is built on its own machine:

- **Build locally:**
  - macOS: `bash packaging/build_macos.sh`  → `dist/FuzzFace.app`
  - Windows: `packaging\build_windows.bat`   → `dist\FuzzFace.exe`
- **Build via CI (both at once):** the GitHub Actions workflow
  `.github/workflows/build.yml` builds the macOS `.app` and Windows `.exe` and
  uploads them as downloadable artifacts. Trigger it from the repo's
  **Actions → Build desktop apps → Run workflow**, or by pushing a `v*` tag.

> Note: the bundled apps are unsigned. On macOS, right-click → Open the first
> time to bypass Gatekeeper; on Windows, allow it past SmartScreen.

## Parameters

| Flag | Meaning | Default |
|------|---------|---------|
| `--gain` | input drive before the nonlinearity | `20` |
| `--model` | `asym` (Fuzz Face), `tube` (DAFX Eq. 4.13), `exp`, `softclip`, `hard` | `asym` |
| `--neg-level` | negative-half clip level for `asym` (0..1; lower = more asymmetric) | `0.6` |
| `--Q`, `--dist` | work point / hardness for the `tube` model | `-0.2`, `8` |
| `--oversample` | oversampling factor N≥1 (anti-aliasing); `1` = off | `8` |
| `--rh` | DC-block high-pass pole (<1) | `0.995` |
| `--rl` | tone low-pass pole (0..1; ≤0 disables) | `0.5` |
| `--volume` | output gain 0..1 (default: match input peak) | auto |
| `--mix` | dry/wet blend (1 = fully wet) | `1.0` |
| `--subtype` | output WAV subtype (`PCM_16`/`PCM_24`/`FLOAT`) | `PCM_24` |

## Files

```
fuzzface.py            effect implementation + CLI
gui.py                 desktop GUI (Tkinter)
demo.py                renders wet samples and figures
packaging/             PyInstaller build scripts (macOS + Windows)
.github/workflows/     CI to build the desktop apps
requirements.txt       runtime deps (incl. matplotlib for demo)
requirements-build.txt deps for packaging the GUI (no matplotlib)
samples/dry/           dry inputs (speechshort.wav, casta.wav)
samples/wet/           processed outputs
figures/               characteristic curve, waveform, spectrum, aliasing plots
report.md              ~2-page written report
```

Sample rate, bit depth, and channel count of the input are preserved.
