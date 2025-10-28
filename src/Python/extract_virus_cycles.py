#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exact extraction of Access Virus waves → single-cycle pack (NO de-hum).
• 64 WAVs × (DIRTY + CLEAN), 48 kHz, 16-bit, 65,536 samples, loop points
• Envelope segmentation (no drift), loud-window recenter, C5 period lock
• Progress bars, multi-page PDF, CSV summary, ZIP
"""

import sys, math, struct, zipfile, shutil, time, csv, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample, resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SR_TARGET   = 48000
N_OUT       = 65536
AMP         = 0.999
FREQ_C5     = 523.2511306011972
NUM_WAVES   = 64
PDF_NAME    = "AccessVirus_Exact_Envelope_Preview.pdf"
CSV_NAME    = "AccessVirus_Exact_Envelope_Summary.csv"
ZIP_NAME    = "AccessVirus_Exact_Envelope.zip"

def progress(prefix, i, n, width=28):
    pct = int((i/n)*100); filled = int(width*i/n)
    bar = "█"*filled + "·"*(width-filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {pct:3d}%"); sys.stdout.flush()
    if i==n: sys.stdout.write("\n")

def write_wav_with_loop(path, data, sr=SR_TARGET, loop_start=0, loop_end=None):
    if loop_end is None: loop_end = len(data)-1
    x = np.asarray(data, dtype=np.float32); x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    fmt  = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16)
    data = b"data" + struct.pack("<I", len(pcm)) + pcm
    smpl_header = struct.pack("<IIIIIIII", 0, 0, int(1e9/sr), 60, 0, 0, 0, 1) + struct.pack("<I", 0)
    smpl_loop   = struct.pack("<IIIIII", 0, 0, int(loop_start), int(loop_end), 0, 0)
    smpl  = b"smpl" + struct.pack("<I", len(smpl_header)+len(smpl_loop)) + smpl_header + smpl_loop
    riffd = fmt + data + smpl
    riff  = b"RIFF" + struct.pack("<I", 4+len(riffd)) + b"WAVE" + riffd
    with open(path, "wb") as f: f.write(riff)

def normalize_dc_free(w):
    w = w - np.mean(w); p = np.max(np.abs(w))
    return (w/p)*AMP if p>1e-12 else w

def bandlimit_clean(w, strength=0.6):
    W = np.fft.rfft(w); n=len(W); start=int(n*strength)
    if start < n:
        W[start:] *= 0.5*(1+np.cos(np.linspace(0, np.pi, n-start)))
    return np.fft.irfft(W, n=len(w))

def mag_spectrum_db(x):
    X = np.fft.rfft(x * np.hanning(len(x)))
    M = 20*np.log10(np.maximum(1e-9, np.abs(X))); return M-np.max(M)

def auto_segments_from_envelope(x, sr, num=64, min_on_ms=1200, pre_ms=120, post_ms=120,
                                rms_win_ms=10, silence_db=-40):
    nwin = max(1, int(sr*rms_win_ms/1000.0))
    rms = np.sqrt(np.convolve(x*x, np.ones(nwin)/nwin, mode='same') + 1e-12)
    rms_db = 20*np.log10(np.maximum(1e-12, rms))
    on = rms_db > silence_db
    edges = np.where(on[1:] & (~on[:-1]))[0] + 1
    if on[0]: edges = np.r_[0, edges]
    segs = []
    i=0
    while i < len(edges):
        s = int(edges[i]); j = s
        while j < len(on) and on[j]: j += 1
        e = int(j)
        if (e-s)*1000.0/sr >= min_on_ms:
            c = (s+e)//2; half = int(0.6*sr)
            ss = max(0, c - half - int(pre_ms*sr/1000.0))
            ee = min(len(x), c + half + int(post_ms*sr/1000.0))
            segs.append((ss, ee))
        i += 1
    if len(segs) >= num: return segs[:num]
    L = len(x)//num; return [(k*L, (k+1)*L) for k in range(num)]

def recentre_to_loud_window(seg, sr, win_s=0.8):
    win = int(sr*win_s)
    if len(seg) <= win: return seg
    step = max(1, win//8)
    best_s, best_r = 0, -1.0
    for s in range(0, len(seg)-win, step):
        w = seg[s:s+win]; r = float(np.sqrt(np.mean(w*w) + 1e-12))
        if r > best_r: best_r, best_s = r, s
    return seg[best_s:best_s+win]

def estimate_period(seg, sr, hint):
    mid = len(seg)//2; half = int(0.4*sr)
    a = max(0, mid-half); b = min(len(seg), mid+half)
    z = seg[a:b] - np.mean(seg[a:b])
    max_lag = int(hint*1.5); min_lag = int(hint*0.5)
    ac = np.correlate(z, z, mode='full')[len(z)-1:len(z)-1+max_lag+1]; ac[0]=0
    return np.argmax(ac[min_lag:max_lag+1]) + min_lag

def extract_cycle(seg, sr, hint):
    lag = estimate_period(seg, sr, hint)
    center = len(seg)//2
    zc = np.where(np.diff(np.signbit(seg)))[0]
    target = center - lag//2
    start = int(zc[np.argmin(np.abs(zc - target))]) if len(zc) else max(0, target)
    cyc = seg[start:start+lag]
    if len(cyc) < lag: cyc = np.pad(cyc, (0, lag-len(cyc)), mode='wrap')
    cyc = resample(cyc, N_OUT)
    cyc = normalize_dc_free(cyc)
    cyc = enforce_seamless_wrap(cyc, force_zero=True, edge=8)  # edge=8–16 is plenty
    return cyc, lag

def find_pos_zero_cross(x):
    """Return index i where x[i] <= 0 and x[i+1] > 0; pick the one closest to zero."""
    z = np.where(np.diff(np.signbit(x)))[0]
    if len(z) == 0:
        # fallback: sample with smallest |value|
        return int(np.argmin(np.abs(x[:-1])))
    # prefer positive-going crossings
    pos = [i for i in z if x[i] <= 0 and x[i+1] > 0]
    cands = pos if pos else z.tolist()
    # choose the pair with the smaller abs value at the crossing
    best = min(cands, key=lambda i: min(abs(x[i]), abs(x[i+1])))
    return best

def rotate_to_zero_cross(x):
    """Circularly rotate so the file starts on a positive-going zero-cross."""
    i = find_pos_zero_cross(x)
    # choose the closer of the two samples at the crossing as start
    start = i if abs(x[i]) <= abs(x[i+1]) else i+1
    return np.r_[x[start:], x[:start]]

def enforce_seamless_wrap(x, force_zero=True, edge=8):
    """
    Make loop from end→start seamless.
    - If force_zero: start at a pos-going zero-cross *and* set exact zeros at both ends,
      with tiny cosine fades to keep slope sane (great for CV/LFO).
    - Else: just force end == start to kill round-off clicks.
    """
    y = rotate_to_zero_cross(x).copy()
    if force_zero:
        # set exact zeros at ends + micro cosine fades to avoid micro-kink
        y0 = y.copy()
        y[0] = 0.0; y[-1] = 0.0
        if edge > 0:
            fade = 0.5*(1-np.cos(np.linspace(0, np.pi, edge, endpoint=True)))
            y[:edge] = y[:edge] * fade
            y[-edge:] = y[-edge:] * fade[::-1]
    else:
        # enforce wrap-equality (common “math perfect loop” trick)
        m = 0.5*(y[0] + y[-1])
        y[0] = m; y[-1] = m
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("AccessVirus_AllWaves.wav"))
    ap.add_argument("--out", type=Path, default=Path("AccessVirus_Exact_Envelope"))
    ap.add_argument("--count", type=int, default=NUM_WAVES)
    ap.add_argument("--silence-db", type=float, default=-40.0)
    ap.add_argument("--min-on-ms", type=int, default=1200)
    args = ap.parse_args()

    SRC=args.src; OUT_BASE=args.out
    DIRTY=OUT_BASE/"DIRTY"; CLEAN=OUT_BASE/"CLEAN"
    PDF_PATH=OUT_BASE/ PDF_NAME; CSV_PATH=OUT_BASE/ CSV_NAME
    ZIP_PATH=OUT_BASE.parent/ ZIP_NAME
    count = max(1, min(NUM_WAVES, int(args.count)))

    print("Step 1/6  Loading…")
    x, sr = sf.read(str(SRC), always_2d=False)
    if x.ndim==2: x=x[:,0]
    x = x.astype(np.float32)

    if sr != SR_TARGET:
        print(f"          Resampling {sr} → {SR_TARGET} Hz…")
        from math import gcd
        g = gcd(sr, SR_TARGET); up = SR_TARGET//g; down = sr//g
        x = resample_poly(x, up, down); sr = SR_TARGET

    # *** NO DE-HUM FILTERING HERE ***

    print("Step 2/6  Envelope-based segmentation…")
    segments_all = auto_segments_from_envelope(
        x, sr, num=NUM_WAVES,
        min_on_ms=args.min_on_ms, pre_ms=120, post_ms=120,
        rms_win_ms=10, silence_db=args.silence_db
    )
    segments = segments_all[:count]

    print("Step 3/6  Extracting single cycles (C5 period lock)…")
    if OUT_BASE.exists(): shutil.rmtree(OUT_BASE)
    DIRTY.mkdir(parents=True, exist_ok=True); CLEAN.mkdir(parents=True, exist_ok=True)

    period_hint = int(round(sr / FREQ_C5))
    rows = [["name","peak_abs","dc","period_samples"]]

    for i, (s,e) in enumerate(segments, start=1):
        seg = x[s:e]
        seg = recentre_to_loud_window(seg, sr, win_s=0.8)
        w, lag = extract_cycle(seg, sr, period_hint)
        nm = f"VIRUS{i:02d}"
        write_wav_with_loop(DIRTY/f"{nm}.wav", w, loop_end=N_OUT-1)
        wc = normalize_dc_free(bandlimit_clean(w, strength=0.6))
        write_wav_with_loop(CLEAN/f"{nm}.wav", wc, loop_end=N_OUT-1)
        rows.append([nm, f"{np.max(np.abs(w)):.6f}", f"{np.mean(w):.6e}", str(lag)])
        progress("          Rendering waves", i, count)

    print("Step 4/6  PDF preview…")
    with PdfPages(str(PDF_PATH)) as pdf:
        for i in range(count):
            nm = f"VIRUS{i+1:02d}"
            y, _ = sf.read(str(DIRTY/f"{nm}.wav"), always_2d=False)
            if y.ndim==2: y=y[:,0]
            fig, axs = plt.subplots(1,2, figsize=(7.2,2.0))
            axs[0].plot(y[:2048]); axs[0].set_title(f"{nm} (time)", fontsize=9)
            axs[0].set_xticks([]); axs[0].set_yticks([])
            M = mag_spectrum_db(y); axs[1].plot(M)
            axs[1].set_title(f"{nm} (spectrum dB)", fontsize=9)
            axs[1].set_xticks([]); axs[1].set_yticks([])
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
            progress("          Drawing preview", i+1, count)

    print("Step 5/6  CSV + ZIP…")
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for sub in [CLEAN, DIRTY]:
            for fp in sorted(sub.glob("*.wav")):
                z.write(fp, arcname=str(fp.relative_to(OUT_BASE)))
        z.write(PDF_PATH, arcname=PDF_PATH.name)
        z.write(CSV_PATH, arcname=CSV_PATH.name)

    print(f"Step 6/6  Done → {ZIP_PATH.name}")

if __name__ == "__main__":
    main()
