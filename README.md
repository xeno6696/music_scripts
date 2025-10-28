# Music Tools to assist with various problms 
This repository is a simple collection of various scripts (some of which are AI generated) that are designed to assiss with real problems.  

##Analog RYTM Scripts

###src/Python/wav_extract.py  

Check usage to understand:
```
python .\wav_extract.py
usage: wav_extract.py [-h] --in INPUT --out OUT_DIR [--fs FS] [--n-out N_OUT] [--amp AMP] [--freq-c5 FREQ_C5]
                      [--num-waves NUM_WAVES] [--count COUNT] [--silence-db SILENCE_DB] [--min-on-ms MIN_ON_MS]
                      [--pre-ms PRE_MS] [--post-ms POST_MS] [--rms-win-ms RMS_WIN_MS] [--dehum] [--dehum-q DEHUM_Q]
                      [--clean-strength CLEAN_STRENGTH] [--no-zero-cross] [--no-force-zero] [--edge-fade EDGE_FADE]
```

The input file is assumed to be a sequence of C5 (261.625580Hz)  notes sampled at a regular interval. 

WAV Extract CLI — Quick Reference (AR LFO Single‑Cycle Workflow)

Use this tool to turn a long synth sweep into looping single‑cycle WAVs that the Analog Rytm MkII can play/loop as tempo‑matchable LFO sources.

Basic usage
`python wav_extract.py --in sweep.wav --out cycles`

Creates:

cycles/DIRTY/WAVE01.wav … (raw)

cycles/CLEAN/WAVE01.wav … (mildly band‑limited)

cycles/Preview.pdf (time + spectrum per wave)

cycles/Summary.csv (peak, DC, detected period)

cycles.zip (packaged set)

AR naming: files are WAVE01…WAVE64 (≤8 chars), loop points embedded.

Common flags
```
--count 8 → process first N waves (smoke test)

--fs 48000 → target sample rate (Rytm‑friendly)

--n-out 65536 → samples per exported single‑cycle (power‑of‑two = good)

--amp 0.999 → peak normalization per file

--silence-db -40 → envelope threshold (raise to avoid hum triggers)

--min-on-ms 1200 → minimum tone length to accept (ms)

--clean-strength 0.6 → CLEAN roll‑off (lower = darker)

--edge-fade 8 → micro fade at loop edges (samples)

--dehum → enable 60/120/180/240 Hz notches (off by default)
```
Zero‑cross handling (on by default):
```
--no-zero-cross to disable rotating to a positive‑going zero crossing

--no-force-zero to keep edge samples equal (not forced to 0)
```
Presets you’ll likely use

`Full 64‑wave pack at 48k / 65,536 samples:

python wav_extract.py --in sweep.wav --out cycles`

Quick test (first 8 waves):

`python wav_extract.py --in sweep.wav --out cycles --count 8`

Darker CLEAN set:

`python wav_extract.py --in sweep.wav --out cycles --clean-strength 0.5`

Edge shaping off (raw loops):

`python wav_extract.py --in sweep.wav --out cycles --no-zero-cross --no-force-zero --edge-fade 0`
Troubleshooting

Blank/flat files → segmentation drifted into silence. Lower --silence-db (e.g., -45) or increase --min-on-ms.

Clicks at loop point → leave zero‑cross handling enabled; increase --edge-fade to 12–16.

Wrong pitch period → your source wasn’t exactly C5; adjust --freq-c5 to the note you recorded.

Hum biasing detection → try --dehum or raise --silence-db (e.g., -35).
##WARNING:  In my experience, hum biasing hasn't produced useful results, i.e., it ends up outputing completely zeroed-out waves.  Use at your own risk,
PR's accepted.  (I just fixed the hum in the studio and gave up on it.)  

###src/Python/ar_lfo_tune.py

You feed in:
**
BPM (tempo)**

Steps per cycle (e.g., 64 = 4 bars)

…and it tells you the total semitone shift your Rytm needs (TRIG NOTE + TUNE sum).
```
python ar_lfo_tune.py 167 64
=== AR LFO Tuning ===
BPM: 167    Steps per cycle: 64
N: 65536 samples    Fs: 48000 Hz
Total semitone shift (TRIG NOTE + TUNE): -24.88 st (-2488 cents)
Recommended breakdown →  TRIG NOTE: -25 st    TUNE: +0.12 st

```
