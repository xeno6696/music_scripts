#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ar_lfo_tune.py — Compute Analog Rytm tuning to tempo-match a looping single-cycle.

Given:
  • BPM (tempo)
  • X   (steps, i.e., 16ths per one loop cycle)

Assuming a single-cycle sample of length N at sample rate Fs, the required
total semitone shift is:

    s = 12 * log2( (BPM * N) / (15 * X * Fs) )

On AR, effective pitch = TRIG NOTE (integer semitone transpose) + TUNE (±24 st).
This tool picks TRIG NOTE as the nearest integer to s, and TUNE = s − TRIG NOTE,
which is guaranteed to be within ±0.5 st (comfortably inside ±24).

You can override N and Fs via flags. Defaults: N=65536, Fs=48000.

Usage:
  python ar_lfo_tune.py <BPM> <X> [--n 65536] [--fs 48000] [--prec 2]
                        [--max-tune 24] [--show-achieved]
"""
import argparse, math

def semitones_for(bpm: float, steps: float, n: int, fs: float) -> float:
    # s = 12 * log2( (BPM * N) / (15 * X * Fs) )
    ratio = (bpm * n) / (15.0 * steps * fs)
    return 12.0 * math.log2(ratio)

def breakdown_to_note_tune(s: float, max_tune: float = 24.0):
    """
    Choose an integer TRIG NOTE so that residual (TUNE) is within ±max_tune.
    Strategy: nearest integer to s → residual always within ±0.5 (<< 24).
    """
    note = int(round(s))
    tune = s - note
    # Safety clamp if someone uses a tiny max_tune (not needed for default 24):
    if abs(tune) > max_tune:
        # shift note until tune in range
        delta = math.copysign(math.ceil((abs(tune) - max_tune)), tune)
        note += int(delta)
        tune = s - note
    return note, tune

def achieved_steps(bpm: float, n: int, fs: float, total_semitones: float) -> float:
    """
    Inverse of the main formula:
      X = (BPM * N) / (15 * Fs * 2^(s/12))
    """
    return (bpm * n) / (15.0 * fs * (2.0 ** (total_semitones / 12.0)))

def main():
    ap = argparse.ArgumentParser(description="Compute AR tuning for a tempo-matched single-cycle loop.")
    ap.add_argument("bpm", type=float, help="Tempo in BPM (e.g., 167)")
    ap.add_argument("steps", type=float, help="Steps (16ths) per loop (e.g., 64)")
    ap.add_argument("--n", type=int, default=65536, help="Single-cycle length in samples (default: 65536)")
    ap.add_argument("--fs", type=float, default=48000.0, help="Sample rate in Hz (default: 48000)")
    ap.add_argument("--max-tune", type=float, default=24.0, help="Max ±TUNE range you’ll allow (default: 24)")
    ap.add_argument("--prec", type=int, default=2, help="Decimal places for printout (default: 2)")
    ap.add_argument("--show-achieved", action="store_true",
                    help="Also print achieved steps for the recommended TRIG NOTE + TUNE sum.")
    args = ap.parse_args()

    s = semitones_for(args.bpm, args.steps, args.n, args.fs)
    note, tune = breakdown_to_note_tune(s, max_tune=args.max_tune)

    # Pretty output
    cents = s * 100.0
    total_str = f"{s:.{args.prec}f} st ({cents:.0f} cents)"
    tune_str = f"{tune:+.{args.prec}f} st"
    print("=== AR LFO Tuning ===")
    print(f"BPM: {args.bpm:g}    Steps per cycle: {args.steps:g}")
    print(f"N: {args.n} samples    Fs: {args.fs:g} Hz")
    print(f"Total semitone shift (TRIG NOTE + TUNE): {total_str}")
    print(f"Recommended breakdown →  TRIG NOTE: {note:+d} st    TUNE: {tune_str}")

    if args.show_achieved:
        s_used = note + tune
        X_ach = achieved_steps(args.bpm, args.n, args.fs, s_used)
        err_steps = X_ach - args.steps
        err_pct = (err_steps / args.steps) * 100.0
        print(f"Achieved steps with {note:+d} + {tune_str}: {X_ach:.3f} steps  "
              f"(error {err_steps:+.3f} steps, {err_pct:+.3f}%)")

    # Quick hint for AR: effective pitch is (TRIG NOTE + TUNE); either adjust TRIG NOTE (integer semitones)
    # or tweak TUNE until their sum ≈ the 'Total semitone shift' above.

if __name__ == "__main__":
    main()
