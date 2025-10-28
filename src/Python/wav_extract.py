#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wav_extract.py — Generic single-cycle extractor for synth sweeps (Elektron-friendly)

WHAT THIS DOES
--------------
Takes a long WAV of a synth sweep that plays a series of steady notes/waves,
finds each sounding region automatically (via RMS envelope), picks the most
stable portion, detects the fundamental period (around C5 by default),
extracts exactly one cycle, resamples that cycle to a fixed sample-count
(e.g., 65,536), normalizes/DC-removes it, makes the loop point clickless
(positive-going zero-cross at the start, optional micro-fade), and exports:

• DIRTY/  → raw single-cycles (48 kHz, 16-bit, mono), loop points embedded
• CLEAN/  → mildly band-limited copies (same format), loop points embedded
• Preview.pdf  → a visual check of waveform + spectrum for each file
• Summary.csv  → peak, DC offset, detected period used
• <out>.zip    → everything zipped for easy transfer

WHY POWERS OF TWO?
------------------
Using 65,536 (or 32,768 / 131,072) samples per cycle plays nicely with
resampling math and makes tempo-matching predictable on samplers.

USAGE (required arguments)
--------------------------
python wav_extract.py --in sweep.wav --out output_folder

OPTIONAL (useful) FLAGS
-----------------------
--count N           process only the first N detected waves (smoke test)
--fs 48000          target sample rate (Hz) for exports (Elektron = 48k)
--n-out 65536       samples per exported single-cycle (power-of-two ideal)
--amp 0.999         peak normalization per file
--freq-c5 523.251   expected base frequency (Hz) used by the period finder
--silence-db -40    envelope threshold (dBFS) to separate sound from silence
--min-on-ms 1200    minimum tone length (ms) accepted as a valid region
--clean-strength .6 roll-off strength for CLEAN copies (0..1, lower = darker)
--edge-fade 8       micro cosine fade (samples) at start/end to smooth slope
--dehum             enable notch filters at 60/120/180/240 Hz (off by default)
--no-zero-cross     disable phase-aligning loop to positive zero-cross
--no-force-zero     keep edge samples equal but not forced to exactly zero

DEPENDENCIES
------------
pip install numpy soundfile scipy matplotlib
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
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ==============================
# Progress bar (cosmetic only)
# ==============================
def show_progress(prefix: str, completed: int, total: int, bar_width: int = 28) -> None:
    """
    Print a simple progress bar on one line.

    prefix:      label to show before the bar
    completed:   number of items done
    total:       total number of items
    bar_width:   width of the bar in characters
    """
    percent_int = int((completed / total) * 100)
    filled_chars = int(bar_width * completed / total)
    bar = "█" * filled_chars + "·" * (bar_width - filled_chars)
    sys.stdout.write(f"\r{prefix} [{bar}] {percent_int:3d}%")
    sys.stdout.flush()
    if completed == total:
        sys.stdout.write("\n")


# ======================================
# WAV writer that embeds a loop section
# ======================================
def write_wav_with_loop(
    output_path: Path,
    mono_float_audio: np.ndarray,
    sample_rate_hz: int,
    loop_start_sample: int = 0,
    loop_end_sample: int | None = None,
) -> None:
    """
    Write a mono 16-bit WAV with an embedded 'smpl' chunk defining a forward loop.

    output_path:        file destination
    mono_float_audio:   float32 array in [-1.0, 1.0]
    sample_rate_hz:     output sample rate (e.g., 48000)
    loop_start_sample:  loop region start index (default 0)
    loop_end_sample:    loop region end index (default last sample)
    """
    if loop_end_sample is None:
        loop_end_sample = len(mono_float_audio) - 1

    # clamp to [-1, 1] and convert to 16-bit little-endian PCM
    clamped = np.clip(mono_float_audio.astype(np.float32), -1.0, 1.0)
    pcm_bytes = (clamped * 32767.0).astype("<i2").tobytes()

    # Standard PCM 'fmt ' and 'data' chunks
    bytes_per_second = sample_rate_hz * 2  # mono 16-bit
    block_align_bytes = 2
    fmt_chunk = b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, sample_rate_hz, bytes_per_second, block_align_bytes, 16
    )
    data_chunk = b"data" + struct.pack("<I", len(pcm_bytes)) + pcm_bytes

    # 'smpl' chunk with one loop
    manufacturer = 0
    product = 0
    sample_period_ns = int(1e9 / sample_rate_hz)  # in nanoseconds
    midi_unity_note = 60
    midi_pitch_fraction = 0
    smpte_format = 0
    smpte_offset = 0
    number_of_loops = 1
    sampler_data = 0
    cue_point_id = 0
    loop_type_forward = 0
    loop_fraction = 0
    loop_play_count = 0

    smpl_header = (
        struct.pack(
            "<IIIIIIII",
            manufacturer,
            product,
            sample_period_ns,
            midi_unity_note,
            midi_pitch_fraction,
            smpte_format,
            smpte_offset,
            number_of_loops,
        )
        + struct.pack("<I", sampler_data)
    )
    smpl_loop = struct.pack(
        "<IIIIII",
        cue_point_id,
        loop_type_forward,
        int(loop_start_sample),
        int(loop_end_sample),
        loop_fraction,
        loop_play_count,
    )
    smpl_chunk = b"smpl" + struct.pack("<I", len(smpl_header) + len(smpl_loop)) + smpl_header + smpl_loop

    riff_payload = fmt_chunk + data_chunk + smpl_chunk
    riff_header = b"RIFF" + struct.pack("<I", 4 + len(riff_payload)) + b"WAVE"
    with open(output_path, "wb") as wav_file:
        wav_file.write(riff_header + riff_payload)


# =========================================
# DSP helpers: normalization and band-limit
# =========================================
def remove_dc_and_normalize_peak(
    waveform: np.ndarray, target_peak: float = 0.999
) -> np.ndarray:
    """
    Remove DC offset and normalize the peak to target_peak.
    """
    waveform = waveform - np.mean(waveform)
    peak_value = float(np.max(np.abs(waveform)))
    if peak_value <= 1e-12:
        return waveform
    return (waveform / peak_value) * target_peak


def make_clean_bandlimited_copy(
    waveform: np.ndarray, keep_fraction: float = 0.6
) -> np.ndarray:
    """
    Mild high-end rolloff by cosine windowing the top (1 - keep_fraction) of FFT bins.
    keep_fraction = 0.6 keeps the lowest 60% of bins untouched.
    """
    spectrum = np.fft.rfft(waveform)
    num_bins = len(spectrum)
    first_rolled_bin = int(num_bins * keep_fraction)
    if first_rolled_bin < num_bins:
        cosine_window = 0.5 * (1 + np.cos(np.linspace(0, np.pi, num_bins - first_rolled_bin)))
        spectrum[first_rolled_bin:] *= cosine_window
    return np.fft.irfft(spectrum, n=len(waveform))


def magnitude_spectrum_db(waveform: np.ndarray) -> np.ndarray:
    """
    Return a magnitude spectrum in dB, normalized to 0 dB peak.
    """
    windowed = waveform * np.hanning(len(waveform))
    spectrum = np.fft.rfft(windowed)
    magnitude = 20 * np.log10(np.maximum(1e-9, np.abs(spectrum)))
    return magnitude - np.max(magnitude)


def apply_series_notch_filters(
    audio_data: np.ndarray, sample_rate_hz: int, notch_hz_list: list[float], quality_factor: float = 40.0
) -> np.ndarray:
    """
    Apply narrow IIR notch filters at the specified frequencies (e.g., 60/120/180/240 Hz).
    """
    filtered = audio_data
    for notch_hz in notch_hz_list:
        b_coeff, a_coeff = iirnotch(notch_hz, quality_factor, sample_rate_hz)
        filtered = filtfilt(b_coeff, a_coeff, filtered)
    return filtered


# =========================================================
# Zero-cross alignment + edge conditioning for loop seams
# =========================================================
def index_of_positive_zero_cross(waveform: np.ndarray) -> int:
    """
    Find an index 'i' where waveform[i] <= 0 and waveform[i+1] > 0 (positive-going),
    preferring crossings with the smallest absolute sample value.
    """
    crossing_indices = np.where(np.diff(np.signbit(waveform)))[0]
    if len(crossing_indices) == 0:
        # fallback: pick the nearest-to-zero sample (start index)
        return int(np.argmin(np.abs(waveform[:-1])))
    positive_goers = [i for i in crossing_indices if waveform[i] <= 0 and waveform[i + 1] > 0]
    candidate_indices = positive_goers if positive_goers else crossing_indices.tolist()
    best_index = min(candidate_indices, key=lambda i: min(abs(waveform[i]), abs(waveform[i + 1])))
    return int(best_index)


def rotate_waveform_to_start_at_pos_zero_cross(waveform: np.ndarray) -> np.ndarray:
    """
    Circularly rotate the waveform so index 0 lands at a positive-going zero crossing.
    """
    crossing_index = index_of_positive_zero_cross(waveform)
    start_index = crossing_index if abs(waveform[crossing_index]) <= abs(waveform[crossing_index + 1]) else crossing_index + 1
    return np.r_[waveform[start_index:], waveform[:start_index]]


def enforce_clickless_loop(
    waveform: np.ndarray, force_hard_zero_edges: bool = True, edge_fade_samples: int = 8
) -> np.ndarray:
    """
    Make loop boundaries clickless:
    1) rotate to a positive-going zero-crossing,
    2) either force the first/last samples to exactly 0.0 and apply tiny cosine fades,
       or ensure start==end by averaging (if force_hard_zero_edges=False).
    """
    rotated = rotate_waveform_to_start_at_pos_zero_cross(waveform).copy()

    if force_hard_zero_edges:
        rotated[0] = 0.0
        rotated[-1] = 0.0
        if edge_fade_samples > 0:
            fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge_fade_samples, endpoint=True)))
            rotated[:edge_fade_samples] *= fade
            rotated[-edge_fade_samples:] *= fade[::-1]
    else:
        average_edge_value = 0.5 * (rotated[0] + rotated[-1])
        rotated[0] = average_edge_value
        rotated[-1] = average_edge_value

    return rotated


# ====================================================
# Segmentation (RMS envelope) + single-cycle extraction
# ====================================================
def segment_sweep_by_envelope(
    audio_data: np.ndarray,
    sample_rate_hz: int,
    expected_wave_count: int,
    minimum_on_milliseconds: int,
    prepad_milliseconds: int,
    postpad_milliseconds: int,
    rms_window_milliseconds: int,
    silence_threshold_db: float,
) -> list[tuple[int, int]]:
    """
    Detect up to 'expected_wave_count' sounding regions using an RMS envelope.
    Returns a list of (start_sample, end_sample) tuples.

    Notes:
    - 'minimum_on_milliseconds' filters out spurious short blips.
    - pre/post pad place a cushion around the detected region center so
      later analysis stays in the stable portion of the tone.
    """
    rms_window_length = max(1, int(sample_rate_hz * rms_window_milliseconds / 1000.0))
    running_rms = np.sqrt(
        np.convolve(audio_data * audio_data, np.ones(rms_window_length) / rms_window_length, mode="same")
        + 1e-12
    )
    rms_db = 20 * np.log10(np.maximum(1e-12, running_rms))
    is_sound_on = rms_db > silence_threshold_db

    # rising edges where the signal crosses from "off" to "on"
    rising_edges = np.where(is_sound_on[1:] & (~is_sound_on[:-1]))[0] + 1
    if is_sound_on[0]:
        rising_edges = np.r_[0, rising_edges]

    detected_segments: list[tuple[int, int]] = []
    edge_index = 0
    while edge_index < len(rising_edges):
        start_sample = int(rising_edges[edge_index])
        end_sample = start_sample
        while end_sample < len(is_sound_on) and is_sound_on[end_sample]:
            end_sample += 1

        duration_ms = (end_sample - start_sample) * 1000.0 / sample_rate_hz
        if duration_ms >= minimum_on_milliseconds:
            center_sample = (start_sample + end_sample) // 2
            half_window_samples = int(0.6 * sample_rate_hz)  # analyze ~1.2 s around center
            padded_start = max(0, center_sample - half_window_samples - int(prepad_milliseconds * sample_rate_hz / 1000.0))
            padded_end = min(len(audio_data), center_sample + half_window_samples + int(postpad_milliseconds * sample_rate_hz / 1000.0))
            detected_segments.append((padded_start, padded_end))

        edge_index += 1

    if len(detected_segments) >= expected_wave_count:
        return detected_segments[:expected_wave_count]

    # Fallback: equal slicing if envelope found fewer than expected regions
    slice_length = len(audio_data) // expected_wave_count
    return [(k * slice_length, (k + 1) * slice_length) for k in range(expected_wave_count)]


def choose_loudest_subwindow(
    segment_audio: np.ndarray, sample_rate_hz: int, subwindow_seconds: float = 0.8
) -> np.ndarray:
    """
    Slide a fixed window across the segment and keep the highest-RMS subwindow.
    This tends to land on the most stable part of the tone for period detection.
    """
    subwindow_length = int(sample_rate_hz * subwindow_seconds)
    if len(segment_audio) <= subwindow_length:
        return segment_audio

    slide_step = max(1, subwindow_length // 8)
    best_start_index = 0
    best_rms = -1.0
    for start_index in range(0, len(segment_audio) - subwindow_length, slide_step):
        test_window = segment_audio[start_index : start_index + subwindow_length]
        test_rms = float(np.sqrt(np.mean(test_window * test_window) + 1e-12))
        if test_rms > best_rms:
            best_rms = test_rms
            best_start_index = start_index

    return segment_audio[best_start_index : best_start_index + subwindow_length]


def estimate_period_via_autocorrelation(
    segment_audio: np.ndarray, sample_rate_hz: int, period_hint_samples: int
) -> int:
    """
    Use an autocorrelation search around 'period_hint_samples' to estimate the period.
    Keeps us from slipping to an octave/harmonic by constraining the search window.
    """
    center_index = len(segment_audio) // 2
    half_window_samples = int(0.4 * sample_rate_hz)
    local_start = max(0, center_index - half_window_samples)
    local_end = min(len(segment_audio), center_index + half_window_samples)

    local_signal = segment_audio[local_start:local_end] - np.mean(segment_audio[local_start:local_end])
    max_lag = int(period_hint_samples * 1.5)
    min_lag = int(period_hint_samples * 0.5)

    full_corr = np.correlate(local_signal, local_signal, mode="full")
    positive_lag_corr = full_corr[len(local_signal) - 1 : len(local_signal) - 1 + max_lag + 1]
    positive_lag_corr[0] = 0  # ignore the zero lag

    best_lag = int(np.argmax(positive_lag_corr[min_lag : max_lag + 1]) + min_lag)
    return best_lag


def extract_single_cycle(
    segment_audio: np.ndarray,
    sample_rate_hz: int,
    period_hint_samples: int,
    output_cycle_length: int,
    target_peak: float,
    enforce_zero_cross: bool,
    force_hard_zero_edges: bool,
    edge_fade_samples: int,
) -> tuple[np.ndarray, int]:
    """
    Extract a single period from 'segment_audio' and resample it to 'output_cycle_length'.
    Returns (single_cycle_waveform, detected_period_samples).
    """
    detected_period = estimate_period_via_autocorrelation(segment_audio, sample_rate_hz, period_hint_samples)

    # Find a start cut near the middle: align to a zero crossing closest to the expected phase
    middle_index = len(segment_audio) // 2
    zero_crossings = np.where(np.diff(np.signbit(segment_audio)))[0]
    target_index = middle_index - detected_period // 2
    if len(zero_crossings) == 0:
        start_index = max(0, target_index)
    else:
        start_index = int(zero_crossings[np.argmin(np.abs(zero_crossings - target_index))])

    single_period = segment_audio[start_index : start_index + detected_period]
    if len(single_period) < detected_period:
        # Wrap-pad if we ran off the end
        single_period = np.pad(single_period, (0, detected_period - len(single_period)), mode="wrap")

    # Normalize to a fixed number of samples (e.g., 65,536) for export
    resampled_cycle = resample(single_period, output_cycle_length)
    resampled_cycle = remove_dc_and_normalize_peak(resampled_cycle, target_peak=target_peak)

    # Make loop point clickless (phase align, optional micro-fade)
    if enforce_zero_cross:
        resampled_cycle = enforce_clickless_loop(
            resampled_cycle,
            force_hard_zero_edges=force_hard_zero_edges,
            edge_fade_samples=edge_fade_samples,
        )

    return resampled_cycle, detected_period


# ==========
# CLI Setup
# ==========
def main() -> None:
    parser = argparse.ArgumentParser(description="Generic single-cycle extractor from synth sweeps.")
    # Required inputs/outputs
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Input long recording (.wav)")
    parser.add_argument("--out", dest="output_dir", type=Path, required=True, help="Output base folder")

    # Audio basics
    parser.add_argument("--fs", dest="target_sample_rate_hz", type=float, default=48000.0, help="Target sample rate (Hz)")
    parser.add_argument("--n-out", dest="output_cycle_length", type=int, default=65536, help="Samples per exported cycle")
    parser.add_argument("--amp", dest="target_peak", type=float, default=0.999, help="Peak normalization per file")
    parser.add_argument("--freq-c5", dest="assumed_c5_frequency_hz", type=float, default=523.2511306011972,
                        help="Nominal base frequency used by the period finder (Hz)")
    parser.add_argument("--num-waves", dest="expected_wave_count", type=int, default=64,
                        help="Expected number of distinct tones in the sweep")
    parser.add_argument("--count", dest="waves_to_export", type=int, default=None,
                        help="How many to actually export (default: expected count)")

    # Segmentation settings
    parser.add_argument("--silence-db", dest="silence_threshold_db", type=float, default=-40.0,
                        help="RMS threshold (dBFS) for 'sound-on'")
    parser.add_argument("--min-on-ms", dest="minimum_on_milliseconds", type=int, default=1200,
                        help="Minimum tone length to accept (ms)")
    parser.add_argument("--pre-ms", dest="prepad_milliseconds", type=int, default=120,
                        help="Padding before center (ms)")
    parser.add_argument("--post-ms", dest="postpad_milliseconds", type=int, default=120,
                        help="Padding after center (ms)")
    parser.add_argument("--rms-win-ms", dest="rms_window_milliseconds", type=int, default=10,
                        help="RMS envelope window (ms)")

    # Optional mains de-hum (off by default — only enable if hum is biasing detection)
    parser.add_argument("--dehum", dest="enable_dehum", action="store_true", help="Enable 60/120/180/240 Hz notches")
    parser.add_argument("--dehum-q", dest="dehum_quality_factor", type=float, default=40.0, help="Notch Q (higher = narrower)")

    # CLEAN copy shaping
    parser.add_argument("--clean-strength", dest="clean_keep_fraction", type=float, default=0.6,
                        help="Fraction of spectrum kept before cosine roll-off (0..1)")

    # Loop boundary conditioning
    parser.add_argument("--no-zero-cross", dest="enforce_zero_cross", action="store_false",
                        help="Disable rotating to a positive-going zero crossing")
    parser.add_argument("--no-force-zero", dest="force_hard_zero_edges", action="store_false",
                        help="Do not force exact zeros at start/end; just equalize edges")
    parser.add_argument("--edge-fade", dest="edge_fade_samples", type=int, default=8,
                        help="Cosine micro-fade samples at loop edges (0=off)")

    args = parser.parse_args()

    # Resolve arguments
    input_path: Path = args.input_path
    output_dir: Path = args.output_dir
    target_sample_rate_hz: int = int(args.target_sample_rate_hz)
    output_cycle_length: int = int(args.output_cycle_length)
    target_peak: float = float(args.target_peak)
    assumed_c5_frequency_hz: float = float(args.assumed_c5_frequency_hz)
    expected_wave_count: int = int(args.expected_wave_count)
    waves_to_export: int = int(args.waves_to_export if args.waves_to_export else expected_wave_count)
    silence_threshold_db: float = float(args.silence_threshold_db)
    minimum_on_milliseconds: int = int(args.minimum_on_milliseconds)
    prepad_milliseconds: int = int(args.prepad_milliseconds)
    postpad_milliseconds: int = int(args.postpad_milliseconds)
    rms_window_milliseconds: int = int(args.rms_window_milliseconds)
    enable_dehum: bool = bool(args.enable_dehum)
    dehum_quality_factor: float = float(args.dehum_quality_factor)
    clean_keep_fraction: float = float(args.clean_keep_fraction)
    enforce_zero_cross: bool = bool(args.enforce_zero_cross)  # default True
    force_hard_zero_edges: bool = bool(args.force_hard_zero_edges)  # default True
    edge_fade_samples: int = int(args.edge_fade_samples)

    # Filenames inside output_dir
    dirty_dir = output_dir / "DIRTY"
    clean_dir = output_dir / "CLEAN"
    preview_pdf_path = output_dir / "Preview.pdf"
    summary_csv_path = output_dir / "Summary.csv"
    zip_path = output_dir.with_suffix(".zip")

    # === Load and resample input ===
    print("Step 1/6  Loading input WAV…")
    audio_data, input_sample_rate_hz = sf.read(str(input_path), always_2d=False)
    if audio_data.ndim == 2:
        audio_data = audio_data[:, 0]  # use left channel if stereo
    audio_data = audio_data.astype(np.float32)

    if input_sample_rate_hz != target_sample_rate_hz:
        print(f"          Resampling {input_sample_rate_hz} → {target_sample_rate_hz} Hz…")
        from math import gcd as greatest_common_divisor
        common = greatest_common_divisor(int(input_sample_rate_hz), int(target_sample_rate_hz))
        up_factor = int(target_sample_rate_hz) // common
        down_factor = int(input_sample_rate_hz) // common
        audio_data = resample_poly(audio_data, up_factor, down_factor)
        working_sample_rate_hz = int(target_sample_rate_hz)
    else:
        working_sample_rate_hz = int(input_sample_rate_hz)

    # Optional de-hum (off by default)
    if enable_dehum:
        print("Step 2/6  De-hum at 60/120/180/240 Hz…")
        audio_data = apply_series_notch_filters(
            audio_data, working_sample_rate_hz, [60, 120, 180, 240], quality_factor=dehum_quality_factor
        )
        next_step = 3
    else:
        next_step = 2

    # === Segment into waves ===
    print(f"Step {next_step}/6  Envelope-based segmentation…")
    segments = segment_sweep_by_envelope(
        audio_data=audio_data,
        sample_rate_hz=working_sample_rate_hz,
        expected_wave_count=expected_wave_count,
        minimum_on_milliseconds=minimum_on_milliseconds,
        prepad_milliseconds=prepad_milliseconds,
        postpad_milliseconds=postpad_milliseconds,
        rms_window_milliseconds=rms_window_milliseconds,
        silence_threshold_db=silence_threshold_db,
    )
    segments = segments[:waves_to_export]

    # === Extract single cycles ===
    print(f"Step {next_step+1}/6  Extracting single cycles…")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dirty_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # C5 period hint (in samples): guides the autocorrelation search window
    period_hint_samples = int(round(working_sample_rate_hz / assumed_c5_frequency_hz))

    csv_rows = [["name", "peak_abs", "dc", "period_samples"]]

    for wave_index, (seg_start, seg_end) in enumerate(segments, start=1):
        segment_audio = audio_data[seg_start:seg_end]
        loudest_window = choose_loudest_subwindow(segment_audio, working_sample_rate_hz, subwindow_seconds=0.8)

        single_cycle, detected_period_samples = extract_single_cycle(
            segment_audio=loudest_window,
            sample_rate_hz=working_sample_rate_hz,
            period_hint_samples=period_hint_samples,
            output_cycle_length=output_cycle_length,
            target_peak=target_peak,
            enforce_zero_cross=enforce_zero_cross,
            force_hard_zero_edges=force_hard_zero_edges,
            edge_fade_samples=edge_fade_samples,
        )

        # File naming kept ≤8 chars for AR displays
        file_stem = f"WAVE{wave_index:02d}"

        # Write DIRTY (raw) file
        write_wav_with_loop(
            output_path=dirty_dir / f"{file_stem}.wav",
            mono_float_audio=single_cycle,
            sample_rate_hz=working_sample_rate_hz,
            loop_start_sample=0,
            loop_end_sample=len(single_cycle) - 1,
        )

        # Make a CLEAN copy (mildly band-limited) with the same loop conditioning
        clean_cycle = remove_dc_and_normalize_peak(
            make_clean_bandlimited_copy(single_cycle, keep_fraction=clean_keep_fraction),
            target_peak=target_peak,
        )
        if enforce_zero_cross:
            clean_cycle = enforce_clickless_loop(
                clean_cycle,
                force_hard_zero_edges=force_hard_zero_edges,
                edge_fade_samples=edge_fade_samples,
            )

        write_wav_with_loop(
            output_path=clean_dir / f"{file_stem}.wav",
            mono_float_audio=clean_cycle,
            sample_rate_hz=working_sample_rate_hz,
            loop_start_sample=0,
            loop_end_sample=len(clean_cycle) - 1,
        )

        csv_rows.append(
            [file_stem, f"{float(np.max(np.abs(single_cycle))):.6f}", f"{float(np.mean(single_cycle)):.6e}", str(detected_period_samples)]
        )
        show_progress("          Rendering waves", wave_index, waves_to_export)

    # === Reports (PDF, CSV, ZIP) ===
    print(f"Step {next_step+2}/6  Building PDF preview…")
    with PdfPages(str(preview_pdf_path)) as pdf_handle:
        for wave_index in range(waves_to_export):
            file_stem = f"WAVE{wave_index+1:02d}"
            preview_audio, _ = sf.read(str(dirty_dir / f"{file_stem}.wav"), always_2d=False)
            if preview_audio.ndim == 2:
                preview_audio = preview_audio[:, 0]
            figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
            # Time domain (first few thousand samples is enough to see shape)
            axes[0].plot(preview_audio[:2048])
            axes[0].set_title(f"{file_stem} (time)", fontsize=9)
            axes[0].set_xticks([]); axes[0].set_yticks([])
            # Spectrum
            spectrum_db = magnitude_spectrum_db(preview_audio)
            axes[1].plot(spectrum_db)
            axes[1].set_title(f"{file_stem} (spectrum dB)", fontsize=9)
            axes[1].set_xticks([]); axes[1].set_yticks([])
            figure.tight_layout()
            pdf_handle.savefig(figure)
            plt.close(figure)
            show_progress("          Drawing preview", wave_index + 1, waves_to_export)

    print(f"Step {next_step+3}/6  Writing CSV…")
    with open(summary_csv_path, "w", newline="") as csv_file:
        csv.writer(csv_file).writerows(csv_rows)

    print(f"Step {next_step+4}/6  Zipping output…")
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for subdir in [clean_dir, dirty_dir]:
            for wav_path in sorted(subdir.glob("*.wav")):
                zip_handle.write(wav_path, arcname=str(wav_path.relative_to(output_dir)))
        zip_handle.write(preview_pdf_path, arcname=preview_pdf_path.name)
        zip_handle.write(summary_csv_path, arcname=summary_csv_path.name)

    print(f"Done in {time.time() - t0:.1f}s → {output_dir}")
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
