# Music Tools to assist with various problms 
This repository is a simple collection of various scripts (some of which are AI generated) that are designed to assist with real problems.  

## Analog RYTM Scripts
HOW TO USE WITH MY AR?

1.  Expand the ZIP and use elektron transfer to drop the files over to your AR.
2.  Choose a pad, I use CY or CB, turn down the analog source and then select a wave in the sample slot.  Make sure loop mode is on if you want LFO
3.  I would suggest going into the audio routing menu and turning off that pad from exiting via the main outs.  (in most cases you're sub-audio anyway)
4.  Apply a patch cable from the output of CY/CB and to the Input of Control In.
5.  SETUP YOUR CONTROL IN:  Gear button -> CONTROL -> Control Input 1  (This is your signal "send" at this point)
6.  MODE = CV, CV ZERO LEVEL = 439, CV MAX LEVEL -> 1872
7.  If you set a trig to play your single-cycle assigned in step 2 in loop you should seethe levels wiggling
8.  SET UP KIT MACRO
9.  FUNC + PLAY to enter kit mode
10.  Scroll down to Control In 1 mod
11.  Press encoder to select a pad, use top row encoders to pick a modulation destination, use lower level encoders to set an amount
12.  As the system plays you should see the sensor on the far left changing numbers and possibly bars moving up and down.
13.  If you pick something like a dual VCO on BD or SD, set the target to "FILT FRQ" and then you should be in business.  Make sure you put long trigs down in order to hear the results
14.  BRAP ON!
 
### src/Python/wav_extract.py  
This script started when I read a comment from ExpectResistance's page on YT that suggested that you could use arbitrary waves as input to
the AR's CV inputs (Control-In) and immediately it hit me upside the head--I also have a blackbox and one of the things that's really amazing about it
is that you can use it to sample CV and play it back to modular gear.  After confirming with ExpectResistance that he was using say, the instrument out
of the AR as input to Control In, I got to work.  My first target was simply to transform a collection of Access Virus waves into slow cycles that I could use as a modulation source in precisely this way.  My first batch of scripts were built using ChatGPT trying to emulate Virus waveshapes, and while there were a handful of interesting shapes to use as CV sources, I wasn't happy and ultimately ended up sampling all 64 of the spectral shapes and
using this script to elongate the cycles so that instead of centering around C5 (261.625580) they center down near 3Hz.  Playing a note at C0 should result in a VERY glacial but still smooth wave.  As a part of this process I decided to share my learning--I learned *alot* about digital audio as 
a result of this process and hope you will as well.  You will notice a "CLEAN" and "DIRTY" folder in artifacts/.  Dirty are unaliased waves, though at the glacial pace we're playing with here, steps might not even be audible.  However if you pitch these waves WAY UP you should get some interesting results.  You WILL LOSE A VOICE but you will gain an arbitrary modulation source that also responds to sample edits, i.e. you you can alter the LFO shape in real time by messing with sample start/end points.  

Why all this work to get shapes from my virus synth?  Raw samples might work, but there's alot of noise involved with sampling.  (Not *always* ideal.)  I wanted to get scientficially crafted artifact-free virus waves.  The Access Virus allowed you to use all of the wave types as LFO sources, and I wanted to expand that capability.  Unfortunately, the AR's sampling engine automatically normalizes all inputs, so while you can record CV output from modular gear (my first attempt) like you can do with the blackbox, because normalization assumes audible material, the only audible thing is the noise floor, so that gets raised up and buries the CV signal.  (There might be an interesting experiment to possibly control that such that the CV signal is mildly influenced by noise...)  



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

### src/Python/ar_lfo_tune.py

[![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a})](https://latex.codecogs.com/svg.image?&space;s=12log_{2}\left(\frac{BPM*N}{15*steps*F_{s}}\right))


You feed in:
**BPM (tempo)**

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
