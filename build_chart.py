"""Build fixed NORMAL/HARD charts for Star Trace on a 175 BPM grid.

usage: python build_chart.py <audio> <out.json> [--start SEC] [--end SEC] [--hyper-bar N]

MV:    --end 205 --hyper-bar 104   (album audio: --hyper-bar 104 as well)

Star Trace sits at 175 BPM (beat 0.342857 s). Onset autocorrelation peaks at the
175 BPM eighth (0.171 s) and is flat at the 116.67 BPM eighth, so 116.67 (2/3 of
175) is only a sub-pulse. Notes snap to the 175 BPM eighth-note grid.
"""
import argparse
import json

import librosa
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--bpm", type=float, default=175.0)
ap.add_argument("--start", type=float, default=1.5, help="no notes before this time (s)")
ap.add_argument("--end", type=float, default=None, help="no notes after this time (s)")
ap.add_argument("--hyper-bar", type=int, default=None, help="bar index where the x2 HYPERDRIVE section starts")
args = ap.parse_args()

y, sr = librosa.load(args.audio, sr=22050, mono=True)
dur = len(y) / sr
BEAT = 60.0 / args.bpm
STEP = BEAT / 2  # eighth note

# beat phase from low-band (kick) onsets, fitted over the whole track
hop = 64
fr = sr / hop
low = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))[:46].sum(0)  # < ~500 Hz
kick = np.maximum(0, np.diff(low, prepend=low[0]))


def comb(sig, phase, grid, t0=0.0, t1=None):
    t1 = t1 or dur
    ts = np.arange(phase + np.ceil((t0 - phase) / grid) * grid, t1, grid)
    idx = np.round(ts * fr).astype(int)
    idx = idx[(idx >= 0) & (idx < len(sig))]
    return sig[idx].mean()


phases = np.arange(0, BEAT, 0.001)
ph = phases[int(np.argmax([comb(kick, p, BEAT) for p in phases]))]
bar_ph = int(np.argmax([comb(kick, ph + j * BEAT, 4 * BEAT) for j in range(4)]))
b0 = (ph + bar_ph * BEAT) % (4 * BEAT)
print(f"bpm {args.bpm}  beat {BEAT:.6f}  first bar line {b0:.4f}s")

# onset strength (percussive + harmonic), brightness, loudness
hop2 = 256
yh, yp = librosa.effects.hpss(y)
op = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop2)
oh = librosa.onset.onset_strength(y=yh, sr=sr, hop_length=hop2)
op /= np.percentile(op, 99)
oh /= np.percentile(oh, 99)
env = np.maximum(op, 0.75 * oh)
S = np.abs(librosa.stft(y, hop_length=hop2))
cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
rms = librosa.feature.rms(S=S)[0]
ft = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop2)
rms_at = lambda t: rms[min(len(rms) - 1, int(t * sr / hop2))]


def peak(arr, t, w=0.025):
    m = (ft >= t - w) & (ft <= t + w)
    return arr[m].max() if m.any() else 0.0


nsteps = int((dur - b0) / STEP)
grid = b0 + np.arange(nsteps) * STEP
strength = np.array([peak(env, t) for t in grid])
bright = np.array([np.median(cent[(ft >= t - 0.02) & (ft <= t + 0.06)]) if t < dur - 0.1 else 0 for t in grid])
PER_BAR = 8
nbars = nsteps // PER_BAR
bar_e = np.array([rms[(ft >= grid[b * PER_BAR]) & (ft < grid[b * PER_BAR] + 4 * BEAT)].mean() for b in range(nbars)])
e_norm = np.clip((bar_e - np.percentile(bar_e, 10)) / (np.percentile(bar_e, 95) - np.percentile(bar_e, 10)), 0, 1)
end = args.end if args.end is not None else dur - 3


# HYPERDRIVE: from the bar after "Go" (EDM drop) to the end, notes scroll twice as fast and get denser.
hyper_t = b0 + args.hyper_bar * 4 * BEAT if args.hyper_bar is not None else None
s16 = np.array([peak(env, t) for t in b0 + np.arange(nsteps * 2) * STEP / 2])  # sixteenth-note strengths


def build(hard):
    """Every beat carries a note (continuous stream); offbeats and sixteenths are added by onset strength."""
    picks = []  # (time, kind) where kind is 'q' quarter, 'e' eighth, 's' sixteenth
    for b in range(nbars):
        s0 = b * PER_BAR
        if grid[s0] < args.start or grid[s0] > end:
            continue
        hyper = hyper_t is not None and grid[s0] >= hyper_t - 1e-3
        offs = [strength[s] for s in range(s0 + 1, s0 + PER_BAR, 2)]
        # share of offbeats that get a note: NORMAL 0 / 25%, HARD 50% / 100% (normal / hyper section)
        share = (1.0 if hyper else 0.5) if hard else (0.5 if hyper else 0.0)
        cut = np.quantile(offs, 1 - share) if share > 0 else np.inf
        for s in range(s0, s0 + PER_BAR):
            if grid[s] < args.start or grid[s] > end:
                continue
            if s % 2 == 0:
                if strength[s] > 0.03 or rms_at(grid[s]) > 0.01:
                    picks.append((grid[s], "q", s))
            elif strength[s] >= cut and strength[s] > 0.05:
                picks.append((grid[s], "e", s))
        if hard and hyper:  # sixteenth pickups before strong beats
            for s in range(s0, s0 + PER_BAR, 2):
                k = 2 * s + 3  # the sixteenth just before the next beat
                if k < len(s16) and s16[k] > np.quantile(s16[2 * s0:2 * s0 + 16], 0.7):
                    picks.append((b0 + k * STEP / 2, "s", s))
    picks.sort()
    out = []
    for i, (t, kind, s) in enumerate(picks):
        if kind == "s" and out:
            lane = out[-1][1] + (1 if out[-1][1] < 2 else -1)  # roll from the previous note
        else:
            win = np.sort([bright[n] for _, _, n in picks[max(0, i - 24):i + 24]])
            lane = min(3, int(np.searchsorted(win, bright[s]) / max(1, len(win)) * 4))
            if out and lane == out[-1][1] and t - out[-1][0] < 0.36:
                lane = lane + 1 if lane < 2 else lane - 1  # no fast jacks
            if len(out) >= 2 and lane == out[-1][1] == out[-2][1]:
                lane = (lane + 2) % 4
        out.append([round(float(t), 3), int(lane)])
        bar = s // PER_BAR
        hyper = hyper_t is not None and t >= hyper_t - 1e-3
        if kind == "q" and s % PER_BAR == 0 and (hyper or (hard and bar % 2 == 0 and e_norm[bar] > 0.7)):
            out.append([round(float(t), 3), (3 - lane) if 3 - lane != lane else (lane + 2) % 4])
    return out


normal, hard = build(False), build(True)
for name, c in (("normal", normal), ("hard", hard)):
    ts = sorted({n[0] for n in c})
    print(f"{name}: {len(c)} notes, {len(c) / (ts[-1] - ts[0]):.2f} nps, {ts[0]:.2f}-{ts[-1]:.2f}s, lanes {np.bincount([n[1] for n in c])}")
json.dump({"bpm": args.bpm, "beat": round(BEAT, 6), "b0": round(float(b0), 4), "dur": round(dur, 3),
           "hyper": round(float(hyper_t), 3) if hyper_t is not None else None,
           "normal": normal, "hard": hard}, open(args.out, "w"), separators=(",", ":"))
