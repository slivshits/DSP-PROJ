# Fuzz Face 

A digital implementation of the **Fuzz Face** guitar fuzz effect: an asymmetric,
high-gain nonlinear waveshaper. Reads an audio file and writes the processed
("wet") file.

## Requirements

Python 3.10 or later — numpy, scipy, soundfile, matplotlib.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python fuzzface.py <input.wav> <output.wav> [options]
```

```bash
# Apply the Fuzz Face with default settings
python fuzzface.py samples/dry/jimi-hendrix-on-classical-guitar-purple-haze-Jãomusico.wav samples/wet/jimi-hendrix_fuzz_example.wav

# Heavier drive with the tube model
python fuzzface.py input.wav output.wav --model tube --gain 35
```

## Parameters

| Flag | Required | Meaning | Default |
|------|----------|---------|---------|
| `input` | ✓ | input WAV file path | — |
| `output` | ✓ | output WAV file path | — |
| `--gain` | optional | drive into the nonlinearity — higher = more fuzz and sustain | `20` |
| `--model` | optional | clipping curve: `asym` (Fuzz Face), `tube`, `exp`, `softclip`, `hard` | `asym` |
| `--neg-level` | optional | negative-half clip level for `asym` (0..1; lower = more asymmetric) | `0.6` |
| `--Q` | optional | bias point for `tube` model | `-0.2` |
| `--dist` | optional | clipping hardness for `tube` model | `8` |
| `--oversample` | optional | oversampling factor N≥1; `1` = off (aliasing audible at high drive) | `8` |
| `--rh` | optional | DC-block high-pass pole (<1) | `0.995` |
| `--rl` | optional | tone low-pass pole (0..1; 0 = bypass) | `0.5` |
| `--volume` | optional | output peak level 0..1; omit to match input peak | auto |
| `--mix` | optional | dry/wet blend (1 = fully wet, 0 = dry) | `1.0` |
| `--subtype` | optional | output WAV bit depth: `PCM_16`, `PCM_24`, `FLOAT` | `PCM_24` |
