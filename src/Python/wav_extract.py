#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wav_extract.py — Generic single-cycle extractor for synth sweeps (Elektron-friendly)

WHAT THIS DOES
--------------
Takes a long recording of multiple held notes/waves (with small silences between),
detects each sounding region robustly via RMS envelope, recenters on the most
stable (loudest) part of each region, estimates the period around C5 (by default),
extracts a single cycle, resamples that cycle to an exact fixed length, normalizes
and DC-removes it, aligns it to a zero-crossing, and exports:
  • DIRTY/  → raw single-cycles
  • CLEAN/  → mildly band-limited versions (safer for resampling)
Both are 48 kHz, 16-bit mono WAVs of exactly N_OUT samples with loop points.
It also generates a PDF preview (time + spectrum), a CSV summary, and a ZIP pack.

WHY THE CHOICES
---------------
• Envelope segmentation prevents “drifting into silence” compared to equal slicing.
• Loud-window recentering finds the most reliable part of each held tone.
• Autocorrelation near an expected period (C5 by default) avoids octave mistakes.
• Zero-cross alignment + micro-fades eliminate tiny loop ticks for CV/LFO use.
• 65,536 samples default is power-of-two, great for pitch scaling and slow LFOs.

USAGE (required flags: --in and --out)
--------------------------------------
python wav_extract.py --in sweep.wav --out cycles

Common options:
  --count 32              # only export first 32 regions (fast test)
  --fs 48000              # target sample rate (Elektron AR default)
  --n-out 65536           # samples per output cycle (affects base LFO rate)
  --silence-db -40        # envelope threshold for segmentation
  --min-on-ms 1200        # min duration of “sound-on” to accept
  --clean-strength 0.6    # CLEAN top-band rolloff strength (0..1)
  --edge-fade 8           # samples of cosine micro-fade at loop edges
  --dehum                 # optional mains notch filters (60/120/180/240 Hz)
"""

import sys
import math
import struct
import zipfile
import shutil
import time
import csv
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample, resample_poly, iirnotch, filtfilt

import matplotlib
matplotlib.use("Agg")  # render PDF without a display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ──────────────────────────────────────────────────────────────────────────────
# Small progress bar for user feedback in the console
# ──────────────────────────────────────────────────────────────────────────────
def show_progress(prefix: str, current_index: int, total_count: int, bar_width: int = 28) -> None:
    """Print a single-line progress bar with percentage."""
    percentage = int((current_index / total_count) * 100)
    filled_width = int(bar_width * current_index / total_count)
    bar = "█" * filled_width + "·" * (bar_width - filled_width)
    sys.stdout.write(f"\r{prefix} [{bar}] {percentage:3d}%")
    sys.stdout.flush()
    if current_index == total_count:
        sys.stdout.write("\n")


# ──────────────────────────────────────────────────────────────────────────────
# WAV writer that embeds a SMPL chunk (loop points recognized by many samplers)
# ──────────────────────────────────────────────────────────────────────────────
def write_wav_with_loop(
    path: Path,
    mono_audio_float32: np.ndarray,
    sample_rate_hz: int,
    loop_start_index: int = 0,
    loop_end_index: int | None = None,
) -> None:
    """
    Write mono 16-bit PCM WAV with a single forward loop from loop_start to loop_end.
    Elektron reads this and loops continuously.

    Parameters
    ----------
    path : destination .wav filepath
    mono_audio_float32 : numpy array in [-1, 1]
    sample_rate_hz : output sample rate
    loop_start_index : start sample (inclusive)
    loop_end_index : end sample (inclusive); defaults to last sample
    """
    if loop_end_index is None:
        loop_end_index = len(mono_audio_float32) - 1

    # Clamp for safety, then convert to 16-bit little-endian PCM
    audio_clamped = np.clip(mono_audio_float32.astype(np.float32), -1.0, 1.0)
    pcm_bytes = (audio_clamped * 32767.0).astype("<i2").tobytes()

    # Build minimal RIFF/WAVE with fmt, data, and smpl chunks
    byte_rate = sample_rate_hz * 2  # mono × 16-bit
    block_align = 2
    fmt_chunk = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate_hz, byte_rate, block_align, 16)
    data_chunk = b"data" + struct.pack("<I", len(pcm_bytes)) + pcm_bytes

    # smpl chunk fields (1 loop)
    manufacturer = product = 0
    sample_period_ns = int(1e9 / sample_rate_hz)  # for metadata only
    midi_unity_note = 60
    midi_pitch_fraction = smpte_format = smpte_offset = 0
    num_loops = 1
    sampler_data = 0
    cue_point_id = loop_type = loop_fraction = loop_play_count = 0

    smpl_header = struct.pack(
        "<IIIIIIII", manufacturer, product, sample_period_ns,
        midi_unity_note, midi_pitch_fraction, smpte_format, smpte_offset, num_loops
    ) + struct.pack("<I", sampler_data)

    smpl_loop = struct.pack(
        "<IIIIII", cue_point_id, loop_type, int(loop_start_index), int(loop_end_index),
        loop_fraction, loop_play_count
    )

    smpl_chunk = b"smpl" + struct.pack("<I", len(smpl_header) + len(smpl_loop)) + smpl_header + smpl_loop
    riff_payload = fmt_chunk + data_chunk + smpl_chunk
    riff_header = b"RIFF" + struct.pack("<I", 4 + len(riff_payload)) + b"WAVE"

    with open(path, "wb") as file_handle:
        file_handle.write(riff_header + riff_payload)


# ──────────────────────────────────────────────────────────────────────────────
# DSP utilities (normalization, simple band-limit for CLEAN, spectrum)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_and_remove_dc(input_wave: np.ndarray, peak_target: float = 0.999) -> np.ndarray:
    """Remove DC offset and scale to a peak just below 0 dBFS for headroom."""
    zero_mean_wave = input_wave - np.mean(input_wave)
    peak_value = float(np.max(np.abs(zero_mean_wave)))
    if peak_value > 1e-12:
        return (zero_mean_wave / peak_value) * peak_target
    return zero_mean_wave


def mild_bandlimit_for_clean(input_wave: np.ndarray, keep_fraction: float = 0.6) -> np.ndarray:
    """
    Apply a gentle cosine roll-off to the top (1-keep_fraction) of the spectrum.
    This yields a 'CLEAN' variant less prone to aliasing when pitched.
    """
    complex_spectrum = np.fft.rfft(input_wave)
    spectrum_bins = len(complex_spectrum)
    cutoff_bin = int(spectrum_bins * keep_fraction)

    if cutoff_bin < spectrum_bins:
        cosine_taper = 0.5 * (1 + np.cos(np.linspace(0, np.pi, spectrum_bins - cutoff_bin)))
        complex_spectrum[cutoff_bin:] *= cosine_taper

    return np.fft.irfft(complex_spectrum, n=len(input_wave))


def magnitude_spectrum_db(input_wave: np.ndarray) -> np.ndarray:
    """Return a normalized magnitude spectrum in decibels for plotting."""
    windowed = input_wave * np.hanning(len(input_wave))
    spectrum = np.fft.rfft(windowed)
    magnitude_db = 20 * np.log10(np.maximum(1e-9, np.abs(spectrum)))
    return magnitude_db - np.max(magnitude_db)


def apply_notch_series(
    audio_signal: np.ndarray, sample_rate_hz: int, frequencies_hz: list[float], quality_factor: float = 40.0
) -> np.ndarray:
    """Optional: chain of narrow notches (e.g., mains 60/120/180/240 Hz)."""
    filtered = audio_signal
    for center_frequency in frequencies_hz:
        biquad_b, biquad_a = iirnotch(center_frequency, quality_factor, sample_rate_hz)
        filtered = filtfilt(biquad_b, biquad_a, filtered)
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Loop click prevention: start at positive-going zero-cross; ensure seamless wrap
# ─
