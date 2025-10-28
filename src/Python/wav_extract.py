#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wav_extract.py — Generic single-cycle extractor for synth wave sweeps.

Extracts looping, Elektron-friendly single-cycle WAVs from a long synth sweep.
Features:
  • Envelope-based segmentation (auto-detect individual waves)
  • Loud-window recentering for stability
  • Period-locked single-cycle extraction (default assumes C5=523.251 Hz)
  • Optional zero-cross alignment and micro-fade for clickless loops
  • Optional mains de-hum (off by default)
  • Outputs DIRTY (raw) + CLEAN (band-limited) 48kHz/16-bit WAVs
  • Embedded loop points (0 → end)
  • Multi-page PDF preview, CSV summary, ZIP package
"""

import sys, math, struct, zipfile, shutil, time, csv, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample, resample_poly, iirnotch, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Progress ----------
def progress(prefix, i, n, width=28):
    pct = int((i / n) * 100)
    filled = int(width * i / n)
    bar = "█" * filled + "·" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {pct:3d}%")
    sys.stdout.flush()
    if i == n:
        sys.stdout.write("\n")

# ---------- WAV writer with loop chunk ----------
def write_wav_with_loop(path, data, sr, loop_start=0, loop_end=None):
    if loop_end is None:
        loop_end = len(data) - 1
    x = np.asarray(data, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()

    byte_rate = sr * 2
    block_align = 2
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, byte_rate, block_align, 16)
    data_chunk = b"data" + struct.pack("<I", len(pcm)) + pcm

    smpl_header = struct.pack(
        "<IIIIIIII", 0, 0, int(1e9 / sr), 60, 0, 0, 0, 1
    ) + struct.pack("<I", 0)
    smpl_loop = struct.pack("<IIIIII", 0, 0, int(loop_start), int(loop_end), 0, 0)
    smpl = b"smpl" + struct.pack("<I", len(smpl_header) + len(smpl_loop)) + smpl_header + smpl_loop

    riff_data = fmt + data_chunk + smpl
    riff = b"RIFF" + struct.pack("<I", 4 + len(riff_data)) + b"WAVE" + riff_data
    with open(path, "wb") as f:
        f.write(riff)

# ---------- DSP helpers ----------
def normalize_dc_free(w, peak=0.999):
    w = w - np.mean(w)
    p = np.max(np.abs(w))
    return (w / p) * peak if p > 1e-12 else w

def bandlimit_clean(w, strength=0.6):
    W = np.fft.rfft(w)
    n = len(W); start = int(n * strength)
    if start < n:
        W[start:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, n - start)))
    return np.fft.irfft(W, n=len(w))

def mag_spectrum_db(x):
    X = np.fft.rfft(x * np.hanning(len(x)))
    M = 20 * np.log10(np.maximum(1e-9, np.abs(X)))
    return M - np.max(M)

def notch_series(signal, sr, freqs, Q=40):
    y = signal
    for f0 in freqs:
        b, a = iirnotch(f0, Q, sr)
        y = filtfilt(b, a, y)
    return y

# ---------- Zero-cross utilities ----------
def find_pos_zero_cross(x):
    z = np.where(np.diff(np.signbit(x)))[0]
    if len(z) == 0:
        return int(np.argmin(np.abs(x[:-1])))
    pos = [i for i in z if x[i] <= 0 and x[i+1] > 0]
    cands = pos if pos else z.tolist()
    best = min(cands, key=lambda i: min(abs(x[i]), abs(x[i+1])))
    return best

def rotate_to_zero_cross(x):
    i = find_pos_zero_cross(x)
    start = i if abs(x[i]) <= abs(x[i+1]) else i+1
    return np.r_[x[start:], x[:start]]

def enforce_seamless_wrap(x, force_zero=True, edge=8):
    y = rotate_to_zero_cross(x).copy()
    if force_zero:
        y[0] = 0.0; y[-1] = 0.0
        if edge > 0:
            fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge, endpoint=True)))
            y[:edge] *= fade
            y[-edge:] *= fade[::-1]
    else:
        m = 0.5 * (y[0] + y[-1])
        y[0] = m; y[-1] = m
    return y

# ---------- Segmentation + extraction ----------
def auto_segments_from_envelope(x, sr, num, min_on_ms, pre_ms, post_ms, rms_win_ms, silence_db):
    nwin = max(1, int(sr * rms_win_ms / 1000.0))
    rms = np.sqrt(np.convolve(x * x, np.ones(nwin) / nwin, mode='same') + 1e-12)
    rms_db = 20 * np.log10(np.maximum(1e-12, rms))
    on = rms_db > silence_db
    edges = np.where(on[1:] & (~on[:-1]))[0] + 1
    if on[0]: edges = np.r_[0, edges]
    segments = []
    i = 0
    while i < len(edges):
        s = int(edges[i]); j = s
        while j < len(on) and on[j]: j += 1
        e = int(j)
        dur_ms = (e - s) * 1000.0 / sr
        if dur_ms >= min_on_ms:
            center = (s + e) // 2
            half = int(0.6 * sr)
            ss = max(0, center - half - int(pre_ms * sr / 1000.0))
            ee = min(len(x), center + half + int(post_ms * sr / 1000.0))
            segments.append((ss, ee))
        i += 1
    if len(segments) >= num:
        return segments[:num]
    seg_len = len(x) // num
    return [(k * seg_len, (k + 1) * seg_len) for k in range(num)]

def recentre_to_loud_window(seg, sr, win_s=0.8):
    win = int(sr * win_s)
    if len(seg) <= win:
        return seg
    step = max(1, win // 8)
    best_s, best_r = 0, -1.0
    for s in range(0, len(seg) - win, step):
        w = seg[s:s+win]
        r = float(np.sqrt(np.mean(w*w) + 1e-12))
        if r > best_r:
            best_r, best_s = r, s
    return seg[best_s:best_s+win]

def estimate_period(seg, sr, hint):
    mid = len(seg)//2
    half = int(0.4*sr)
    a = max(0, mid - half)
    b = min(len(seg), mid + half)
    z = seg[a:b] - np.mean(seg[a:b])
    max_lag = int(hint * 1.5)
    min_lag = int(hint * 0.5)
    ac = np.correlate(z, z, mode='full')[len(z)-1:len(z)-1+max_lag+1]
    ac[0] = 0
    return np.argmax(ac[min_lag:max_lag+1]) + min_lag

def extract_cycle(seg, sr, hint, n_out, peak_norm, zero_cross, force_zero, edge_fade):
    lag = estimate_period(seg, sr, hint)
    center = len(seg)//2
    zc = np.where(np.diff(np.signbit(seg)))[0]
    target = center - lag//2
    start = int(zc[np.argmin(np.abs(zc - target))]) if len(zc) else max(0, target)
    cyc = seg[start:start+lag]
    if len(cyc) < lag:
        cyc = np.pad(cyc, (0, lag - len(cyc)), mode='wrap')
    cyc = resample(cyc, n_out)
    cyc = normalize_dc_free(cyc, peak=peak_norm)
    if zero_cross:
        cyc = enforce_seamless_wrap(cyc, force_zero=force_zero, edge=edge_fade)
    return cyc, lag

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generic single-cycle extractor from synth sweeps.")
    # Required I/O
    ap.add_argument("--in", dest="input", type=Path, required=True, help="Input long recording (.wav)")
    ap.add_argument("--out", dest="out_dir", type=Path, required=True, help="Output folder base")
    # Audio basics
    ap.add_argument("--fs", type=float, default=48000.0, help="Target sample rate (Hz)")
    ap.add_argument("--n-out", type=int, default=65536, help="Samples per output cycle")
    ap.add_argument("--amp", type=float, default=0.999, help="Peak normalization")
    ap.add_argument("--freq-c5", type=float, default=523.2511306011972, help="Nominal C5 frequency (Hz)")
    ap.add_argument("--num-waves", type=int, default=64, help="Expected number of waves")
    ap.add_argument("--count", type=int, default=None, help="How many to export (default=num-waves)")
    # Segmentation
    ap.add_argument("--silence-db", type=float, default=-40.0, help="Silence threshold (dBFS)")
    ap.add_argument("--min-on-ms", type=int, default=1200, help="Minimum valid tone length (ms)")
    ap.add_argument("--pre-ms", type=int, default=120, help="Padding before center (ms)")
    ap.add_argument("--post-ms", type=int, default=120, help="Padding after center (ms)")
    ap.add_argument("--rms-win-ms", type=int, default=10, help="RMS window size (ms)")
    # De-hum
    ap.add_argument("--dehum", action="store_true", help="Enable notch filters (60/120/180/240Hz)")
    ap.add_argument("--dehum-q", type=float, default=40.0, help="Dehum notch Q")
    # CLEAN bandlimit
    ap.add_argument("--clean-strength", type=float, default=0.6, help="CLEAN rolloff strength (0–1)")
    # Zero-cross handling
    ap.add_argument("--no-zero-cross", action="store_false", dest="zero_cross", help="Disable zero-cross enforcement")
    ap.add_argument("--no-force-zero", action="store_false", dest="force_zero", help="Disable hard zero endpoints")
    ap.add_argument("--edge-fade", type=int, default=8, help="Cosine microfade samples at edges")
    args = ap.parse_args()

    SRC = args.input
    OUT_BASE = args.out_dir
    DIRTY = OUT_BASE / "DIRTY"
    CLEAN = OUT_BASE / "CLEAN"
    PDF_PATH = OUT_BASE / "Preview.pdf"
    CSV_PATH = OUT_BASE / "Summary.csv"
    ZIP_PATH = OUT_BASE.with_suffix(".zip")

    count = args.count if args.count else args.num_waves
    count = max(1, min(args.num_waves, int(count)))

    t0 = time.time()
    print("Step 1/6  Loading…")
    x, sr_in = sf.read(str(SRC), always_2d=False)
    if x.ndim == 2: x = x[:, 0]
    x = x.astype(np.float32)
    if sr_in != args.fs:
        from math import gcd
        g = gcd(int(sr_in), int(args.fs))
        up = int(args.fs)//g; down = int(sr_in)//g
        x = resample_poly(x, up, down)
        sr = int(args.fs)
    else:
        sr = int(sr_in)

    if args.dehum:
        print("Step 2/6  Dehum 60/120/180/240Hz…")
        x = notch_series(x, sr, [60,120,180,240], Q=args.dehum_q)
        step_idx = 3
    else:
        step_idx = 2

    print(f"Step {step_idx}/6  Segmenting…")
    segments_all = auto_segments_from_envelope(
        x, sr, num=args.num_waves,
        min_on_ms=args.min_on_ms, pre_ms=args.pre_ms, post_ms=args.post_ms,
        rms_win_ms=args.rms_win_ms, silence_db=args.silence_db
    )
    segments = segments_all[:count]

    print(f"Step {step_idx+1}/6  Extracting cycles…")
    if OUT_BASE.exists(): shutil.rmtree(OUT_BASE)
    DIRTY.mkdir(parents=True, exist_ok=True); CLEAN.mkdir(parents=True, exist_ok=True)

    period_hint = int(round(sr / args.freq_c5))
    rows = [["name","peak_abs","dc","period_samples"]]

    for i,(s,e) in enumerate(segments, start=1):
        seg = recentre_to_loud_window(x[s:e], sr)
        w, lag = extract_cycle(seg, sr, period_hint, args.n_out, args.amp,
                               args.zero_cross, args.force_zero, args.edge_fade)
        nm = f"WAVE{i:02d}"
        write_wav_with_loop(DIRTY/f"{nm}.wav", w, sr)
        wc = normalize_dc_free(bandlimit_clean(w, strength=args.clean_strength), peak=args.amp)
        if args.zero_cross: wc = enforce_seamless_wrap(wc, force_zero=args.force_zero, edge=args.edge_fade)
        write_wav_with_loop(CLEAN/f"{nm}.wav", wc, sr)
        rows.append([nm,f"{np.max(np.abs(w)):.6f}",f"{np.mean(w):.6e}",str(lag)])
        progress("          Rendering", i, count)

    print("Step 5/6  PDF + CSV + ZIP…")
    with PdfPages(str(PDF_PATH)) as pdf:
        for i in range(count):
            nm = f"WAVE{i+1:02d}"
            y,_=sf.read(str(DIRTY/f"{nm}.wav"),always_2d=False)
            if y.ndim==2:y=y[:,0]
            fig,axs=plt.subplots(1,2,figsize=(7.2,2.0))
            axs[0].plot(y[:2048]); axs[0].set_title(f"{nm} (time)",fontsize=9)
            axs[0].set_xticks([]); axs[0].set_yticks([])
            M=mag_spectrum_db(y); axs[1].plot(M)
            axs[1].set_title(f"{nm} (spectrum dB)",fontsize=9)
            axs[1].set_xticks([]); axs[1].set_yticks([])
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            progress("          Drawing preview",i+1,count)
    with open(CSV_PATH,"w",newline="") as f:
        csv.writer(f).writerows(rows)
    with zipfile.ZipFile(str(ZIP_PATH),"w",compression=zipfile.ZIP_DEFLATED) as z:
        for sub in [CLEAN,DIRTY]:
            for fp in sorted(sub.glob("*.wav")):
                z.write(fp,arcname=str(fp.relative_to(OUT_BASE)))
        z.write(PDF_PATH,arcname=PDF_PATH.name)
        z.write(CSV_PATH,arcname=CSV_PATH.name)
    print(f"Done in {time.time()-t0:.1f}s → {OUT_BASE}")
    print(f"ZIP: {ZIP_PATH}")

if __name__ == "__main__":
    main()
