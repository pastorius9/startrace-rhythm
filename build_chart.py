"""Build fixed NORMAL/HARD charts for Star Trace on a 175 BPM grid.

usage: python build_chart.py <audio> <out.json> [--start SEC] [--end SEC] [--hyper-bar N]

MV:    --end 205 --hyper-bar 104 --rush-from 100 --go-early 1 --dance-bars 48:56 --vocal-range 120.5:141.1 --finish 206
album: --end 176 --hyper-bar 104 --rush-from 100 --go-early 1 --dance-bars 48:56 --vocal-range 120.58:141.18
(3:26 is when the car crosses the line in the MV; sung long notes run 2:01-2:21: "woah", then "This is the time of our lives")

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
ap.add_argument("--finish", type=float, default=None, help="time (s) the finish line crosses the judge line")
ap.add_argument("--go-early", type=float, default=0, help="beats before the hyper bar where the sung \"Go\" lands (x2 starts there)")
ap.add_argument("--rush-from", type=int, default=None, help="bar index where the pre-Go build-up rush starts")
ap.add_argument("--vocal-range", default=None, help="START:END seconds where long notes follow the sung notes (the 2:01 woah section)")
ap.add_argument("--dance-bars", default=None, help="START:END bar indices of the x1.5 dance break, e.g. 48:56")
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
go_t = hyper_t - args.go_early * BEAT if hyper_t is not None else None  # the sung "Go": chord + x2 start
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
    return hand_split(add_rush(add_vocal(add_holds(out, hard), hard), hard))


def vocal_notes(v0, v1):
    """Sustained sung notes (pyin on the harmonic part) between v0 and v1, snapped to the eighth grid."""
    a, b = int(v0 * sr), int(v1 * sr)
    yh, _ = librosa.effects.hpss(y[a:b], margin=3.0)
    f0, voiced, _ = librosa.pyin(yh, fmin=150, fmax=900, sr=sr, frame_length=2048, hop_length=256)
    tt = librosa.times_like(f0, sr=sr, hop_length=256) + v0
    segs, cur = [], None
    for t, f, v in zip(tt, f0, voiced):
        if v and not np.isnan(f):
            m = float(librosa.hz_to_midi(f))
            if cur and abs(m - cur["m"]) < 1.6 and t - cur["e"] < 0.06:
                cur["e"] = t; cur["ms"].append(m); cur["m"] = float(np.median(cur["ms"][-8:]))
                continue
            if cur: segs.append(cur)
            cur = {"s": t, "e": t, "m": m, "ms": [m]}
        elif cur and t - cur["e"] > 0.06:
            segs.append(cur); cur = None
    if cur: segs.append(cur)
    snap = lambda t: b0 + round((t - b0) / STEP) * STEP
    notes, last_end = [], -1
    for sg in segs:
        pitch = float(np.median(sg["ms"]))
        if sg["e"] - sg["s"] < 0.3 or pitch < 60:   # skip short blips and low instruments
            continue
        st, en = max(snap(sg["s"]), last_end + STEP), min(snap(sg["e"]), v1 - STEP)
        if en - st < 2 * STEP - 1e-3:
            continue
        notes.append((st, en, pitch)); last_end = en
    return notes


def add_vocal(out, hard):
    """The woah section: long notes as long as each sung note, lane by pitch (higher = further right).
    Beats between them stay as taps; on HARD a tap also lands on every beat during a long note."""
    if not args.vocal_range:
        return out
    v0, v1 = (float(x) for x in args.vocal_range.split(":"))
    sung = vocal_notes(v0, v1)
    out = [n for n in out if not (v0 <= n[0] < v1) and not (len(n) > 2 and n[0] < v1 and n[0] + n[2] > v0)]
    lo, hi = min(p for *_, p in sung), max(p for *_, p in sung)
    prev = None
    for st, en, pitch in sung:
        lane = int(round((pitch - lo) / max(1e-6, hi - lo) * 3))
        if prev and lane == prev[1] and st - prev[2] < 0.35:
            lane = lane + 1 if lane < 3 else lane - 1
        out.append([round(float(st), 3), lane, round(float(en - st), 3)])
        prev = (st, lane, en)
    k0 = int(np.ceil((v0 - b0) / BEAT))
    i = 0
    for k in range(k0, int((v1 - b0) / BEAT) + 1):
        tb = b0 + k * BEAT
        if tb >= v1 - 1e-3:
            break
        held = [(st, en, l) for st, en, l in ((n[0], n[0] + n[2], n[1]) for n in out if len(n) > 2) if st - 0.05 <= tb <= en + 0.05]
        if held:
            if hard and abs(tb - held[0][0]) > 1e-3:   # the free thumb alternates its two lanes
                free = (0, 1) if held[0][2] >= 2 else (2, 3)
                out.append([round(float(tb), 3), free[k % 2]])
        else:
            out.append([round(float(tb), 3), 1 + i % 2]); i += 1
    return sorted(out, key=lambda n: (n[0], n[1]))


THUMB_GAP = 0.15  # one thumb can't tap faster than this (s)


def hand_split(out):
    """Two-thumb playability (mobile): lanes 0-1 are the left thumb, 2-3 the right.
    - notes at the same moment go one per thumb,
    - while a thumb holds a long note, other notes go to the other thumb,
    - one thumb never gets two notes closer than THUMB_GAP.
    A note that breaks a rule moves to the other thumb's matching lane, or is dropped if that thumb is busy."""
    hand = lambda l: l // 2
    mirror = lambda l: l + 2 if l < 2 else l - 2
    free_at = [-9.0, -9.0]   # time each thumb can take its next note
    used_at = [None, None]   # time of the thumb's latest note (to keep chords one per thumb)
    res = []
    # long notes first at equal times, so a chord's tap is placed around the hold
    for n in sorted(out, key=lambda n: (n[0], 0 if len(n) > 2 else 1, n[1])):
        t = n[0]
        ok = lambda h: t >= free_at[h] - 1e-6 and used_at[h] != t
        h = hand(n[1])
        lane = n[1]
        if not ok(h):
            h, lane = 1 - h, mirror(n[1])
            if not ok(h):
                continue
        if len(n) > 2:
            res.append([t, lane, n[2]])
            free_at[h] = t + n[2] + 0.12      # release, then a moment before the next tap
        else:
            res.append([t, lane])
            free_at[h] = t + THUMB_GAP
        used_at[h] = t
    return res


def add_rush(out, hard):
    """Build-up before "Go": fast, simple alternating taps on the two middle lanes, then a two-lane
    chord on the outer lanes exactly when the x2 section starts."""
    if args.rush_from is None or hyper_t is None:
        return out
    r0 = b0 + args.rush_from * 4 * BEAT
    # clear the rush window (up to, not including, the drop downbeat) and any long note reaching into it
    out = [n for n in out if not (r0 - 1e-3 <= n[0] < hyper_t - 1e-3)]
    holds = [n for n in out if len(n) > 2]
    t, i, j = r0, 0, 0
    while t < go_t - 0.15 - 1e-3:            # leave both thumbs free for the Go chord
        held = next((h for h in holds if h[0] - 1e-3 <= t <= h[0] + h[2] + 0.12), None)
        if held:   # one thumb holds the sung note: the other alternates its two lanes in eighths
            free = (0, 1) if held[1] >= 2 else (2, 3)
            out.append([round(float(t), 3), free[j % 2]]); j += 1
            t += STEP
            continue
        out.append([round(float(t), 3), 1 + i % 2])
        fast = hard and t >= r0 + 4 * BEAT - 1e-3   # HARD: sixteenths (thumbs alternate) after the first bar
        t += STEP / 2 if fast else STEP
        i += 1
    out += [[round(float(go_t), 3), 0], [round(float(go_t), 3), 3]]
    return sorted(out, key=lambda n: (n[0], n[1]))


CALM = 0.3  # bars quieter than this (normalised loudness) get a long note across the bar


def add_holds(out, hard):
    """Long notes: every other bar of calm stretches (intro, verses, bridge), plus the end of each 8-bar phrase.

    A long note is [t, lane, dur]. Other notes in its span move off its lane; on NORMAL they are
    dropped so the player only has to hold."""
    bar_of = lambda t: int(np.floor((t - b0) / (4 * BEAT) + 1e-6))
    holds, prev_lane = [], 3
    for bar in sorted({bar_of(t) for t, *_ in out}):
        t0 = b0 + bar * 4 * BEAT
        if hyper_t is not None and t0 >= hyper_t - 1e-3:
            continue
        calm = e_norm[bar] < CALM
        if calm and (bar % 2 == 1 or e_norm[bar - 1] >= CALM):
            continue  # calm stretches alternate: one bar held, one bar tapped
        if calm:
            start, length = t0, (3.5 if hard else 3.0) * BEAT
        elif bar % 8 == 7:
            start, length = t0 + 2 * BEAT, (1.5 if hard else 2.0) * BEAT
        else:
            continue
        here = [n for n in out if abs(n[0] - start) < 1e-3]
        if not here:
            continue
        lane = here[0][1] if here[0][1] != prev_lane else (here[0][1] + 2) % 4
        holds.append((start, lane, length))
        prev_lane = lane
    res = [n for n in out]
    for start, lane, length in holds:
        end_t = start + length
        keep = []
        for n in res:
            inside = start - 1e-3 <= n[0] <= end_t + 1e-3
            if not inside:
                keep.append(n)
            elif abs(n[0] - start) < 1e-3:
                continue                      # replaced by the long note
            elif hard and n[0] < end_t - 1e-3:
                keep.append([n[0], n[1] if n[1] != lane else (lane + 1 + int(n[0] * 7) % 3) % 4])
            elif not hard and n[0] > end_t + BEAT * 0.5 - 1e-3:
                keep.append(n)
        keep.append([round(float(start), 3), int(lane), round(float(length), 3)])
        res = keep
    res.sort(key=lambda n: (n[0], n[1]))
    # drop accidental duplicates (same time, same lane)
    seen, final = set(), []
    for n in res:
        key = (n[0], n[1])
        if key not in seen:
            seen.add(key); final.append(n)
    return final


# speed zones the game reads: notes scroll mul times faster between t0 and t1
zones = []
if args.dance_bars:
    a_, b_ = (int(x) for x in args.dance_bars.split(":"))
    zones.append({"t0": round(float(b0 + a_ * 4 * BEAT), 3), "t1": round(float(b0 + b_ * 4 * BEAT), 3), "mul": 1.5, "kind": "dance"})
if hyper_t is not None:
    zones.append({"t0": round(float(go_t), 3), "t1": round(dur + 5, 3), "mul": 2, "kind": "hyper"})

normal, hard = build(False), build(True)
for name, c in (("normal", normal), ("hard", hard)):
    ts = sorted({n[0] for n in c})
    nh = sum(1 for n in c if len(n) > 2)
    print(f"{name}: {len(c)} notes ({nh} long), {len(c) / (ts[-1] - ts[0]):.2f} nps, {ts[0]:.2f}-{ts[-1]:.2f}s, lanes {np.bincount([n[1] for n in c])}")
json.dump({"bpm": args.bpm, "beat": round(BEAT, 6), "b0": round(float(b0), 4), "dur": round(dur, 3),
           "hyper": round(float(go_t), 3) if go_t is not None else None,
           "zones": zones,
           "finish": args.finish,
           "normal": normal, "hard": hard}, open(args.out, "w"), separators=(",", ":"))
