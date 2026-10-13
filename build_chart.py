"""Build fixed EASY/NORMAL/HYPERDRIVE charts for Star Trace on a 175 BPM grid.

usage: python build_chart.py <audio> <out.json> [--start SEC] [--end SEC] [--hyper-bar N]

MV:    --end 205 --hyper-bar 104 --rush-from 100 --go-early 1 --dance-bars 48:56 --vocal-range 120.4:141.1 --finish 206
album: --end 176 --hyper-bar 104 --rush-from 100 --go-early 1 --dance-bars 48:56 --vocal-range 120.48:141.18 --vocal-shift 0.077
       (both: --drums <drum track> --drum-bars 29:32,48:56,69:72,149:150; the drum track is separated with an MDX-Net drum model)
(3:26 is when the car crosses the line in the MV; sung long notes run 2:00.5-2:21: "woah", then "This is the time of our lives")

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
ap.add_argument("--vocal-range", default=None, help="START:END seconds where long notes follow the sung syllables (the 2:00.5 woah section)")
ap.add_argument("--vocal-shift", type=float, default=0.0, help="seconds this audio runs behind the MV audio (album: 0.077)")
ap.add_argument("--dance-bars", default=None, help="START:END bar indices of the x1.5 dance break, e.g. 48:56")
ap.add_argument("--drums", default=None, help="drum track separated from the same audio (same timing), for --drum-bars")
ap.add_argument("--drum-bars", default="", help="bar ranges whose notes follow the drum hits, e.g. 29:32,48:56,69:72,149:150 "
                "(the --rush-from bars before Go always do when --drums is given)")
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


EASY, NORMAL, HYPER = 0, 1, 2
LEVELS = (("easy", EASY), ("normal", NORMAL), ("hyperdrive", HYPER))


def build(lv):
    """Every beat carries a note (continuous stream); offbeats and sixteenths are added by onset strength.
    lv: EASY (beats only), NORMAL (+ offbeats in the x2 section), HYPER (+ offbeats everywhere, sixteenths in x2)."""
    picks = []  # (time, kind) where kind is 'q' quarter, 'e' eighth, 's' sixteenth
    for b in range(nbars):
        s0 = b * PER_BAR
        if grid[s0] < args.start or grid[s0] > end:
            continue
        hyper = hyper_t is not None and grid[s0] >= hyper_t - 1e-3
        offs = [strength[s] for s in range(s0 + 1, s0 + PER_BAR, 2)]
        # share of offbeats that get a note (normal / x2 section): EASY 0 / 0, NORMAL 0 / 50%, HYPER 50% / 100%
        share = ((0.0, 0.0), (0.0, 0.5), (0.5, 1.0))[lv][int(hyper)]
        cut = np.quantile(offs, 1 - share) if share > 0 else np.inf
        for s in range(s0, s0 + PER_BAR):
            if grid[s] < args.start or grid[s] > end:
                continue
            if s % 2 == 0:
                if strength[s] > 0.03 or rms_at(grid[s]) > 0.01:
                    picks.append((grid[s], "q", s))
            elif strength[s] >= cut and strength[s] > 0.05:
                picks.append((grid[s], "e", s))
        if lv == HYPER and hyper:  # sixteenth pickups before strong beats
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
        double = (hyper and (lv > EASY or bar % 2 == 0)) or (lv == HYPER and bar % 2 == 0 and e_norm[bar] > 0.7)
        if kind == "q" and s % PER_BAR == 0 and double:
            out.append([round(float(t), 3), (3 - lane) if 3 - lane != lane else (lane + 2) % 4])
    return hand_split(add_rush(add_vocal(add_drum_bars(add_holds(out, lv), lv), lv), lv))


# Sung syllables in MV time (s): (start, end, midi). end None = short syllable -> tap.
# Measured on the vocal alone (MDX-Net vocal separation, see vocal_onsets.py): start = middle of the
# loudness rise at the syllable's first sound (the consonant or scoop, not where the pitch settles),
# a syllable sung on without a break starts where the pitch steps. Ends are where the voice stops or the next one starts.
SYLLABLES = [
    # woah (2:00.5)
    (120.50, 121.88, 67), (121.92, 122.90, 66), (122.95, None, 67), (123.28, 123.80, 64), (123.87, None, 62),
    # woah (2:06.0)
    (125.99, 127.38, 67), (127.42, 128.38, 66), (128.42, None, 67), (128.74, 129.50, 64),
    # This / is / the / time / of / our / lives (2:10.9)
    (130.90, 131.55, 67), (131.67, 132.80, 69), (132.88, None, 69), (133.14, 134.04, 71),
    (134.20, 135.46, 73), (135.82, 136.38, 71), (136.45, 141.55, 71),
]


def add_vocal(out, lv):
    """Woah and "This is the time of our lives": a long note starts on each sung syllable and lasts as long
    as it is sung; very short syllables are taps. Syllables alternate thumbs (a phrase starts on the left),
    the higher of a thumb's two lanes takes the higher notes.
    Other thumb: beats with nothing sung are taps on every level; while a syllable is held the free thumb
    taps every beat on HYPER, every other beat on NORMAL, never on EASY."""
    if not args.vocal_range:
        return out
    v0, v1 = (float(x) for x in args.vocal_range.split(":"))
    sh = args.vocal_shift
    out = [n for n in out if not (v0 <= n[0] < v1) and not (len(n) > 2 and n[0] < v1 and n[0] + n[2] > v0)]
    phrases, last = [], -9.0                          # a break of over 0.5 s starts a new phrase
    for st, en, midi in SYLLABLES:
        if st - last > 0.5:
            phrases.append([])
        phrases[-1].append((st, en, midi))
        last = en if en is not None else st
    sung, prev_t = [], -9.0
    for st, en, midi, hand, med in ((st, en, m, j % 2, np.median([x[2] for x in ph]))
                                    for ph in phrases for j, (st, en, m) in enumerate(ph)):
        st += sh
        en = en + sh if en is not None else None
        if not (v0 <= st < v1):
            continue
        lane = hand * 2 + (1 if midi >= med else 0)
        sung.append((st, en if en is not None else st, lane))
        out.append([round(st, 3), lane] + ([round(en - st, 3)] if en is not None else []))
    k0 = int(np.ceil((v0 - b0) / BEAT))
    i = 0
    for k in range(k0, int((v1 - b0) / BEAT) + 1):
        tb = b0 + k * BEAT
        if tb >= v1 - 1e-3:
            break
        near = [(st, l) for st, en, l in sung if -0.3 < st - tb < 0.3]
        if any(abs(st - tb) < 0.12 for st, _ in near):
            continue                                      # no flam right next to a syllable
        held = [l for st, en, l in sung if st <= tb <= en + 0.12 and en > st]
        if not held:
            if any(abs(st - tb) < 0.2 for st, _ in near):
                continue                                  # the syllable itself is the note here
            out.append([round(float(tb), 3), 1 + i % 2]); i += 1
            continue
        if lv == EASY or (lv == NORMAL and k % 2) or len({l // 2 for l in held}) > 1:
            continue
        free = 1 - held[0] // 2
        if any(l // 2 == free and -0.15 < st - tb < 0.3 for st, l in near):
            continue                                  # keep that thumb ready for its next syllable
        out.append([round(float(tb), 3), free * 2 + k % 2])
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


# ---------- notes that follow the drums (fills, rolls, the dance break, the build-up before Go)
DRUM_HIT = 9.0   # dB: a drum hit on a sixteenth = the loudness jump of its band right there
BIG_HIT = 20.0   # dB: kick and snare both this loud on a beat -> a two-thumb chord
_drum = None


def drum_hits(t0, t1):
    """Sixteenth-grid drum hits in [t0, t1) from the separated drum track: (time, step in bar, kind, dB jump, big).
    kind 'K' = kick (lows lead) -> outer lanes, 'S' = snare / hats -> inner lanes."""
    global _drum
    if _drum is None:
        yd, _ = librosa.load(args.drums, sr=sr, mono=True)
        Sd = np.abs(librosa.stft(yd, n_fft=512, hop_length=32))
        fd = librosa.fft_frequencies(sr=sr, n_fft=512)
        bands = [20 * np.log10(Sd[(fd >= lo) & (fd < hi)].sum(0) + 1e-6) for lo, hi in ((30, 150), (200, 3000), (5000, 11000))]
        _drum = (np.arange(Sd.shape[1]) * 32 / sr, bands)
    td, bands = _drum
    S16 = BEAT / 4
    hits = []
    for k in range(int(np.ceil((t0 - b0) / S16 - 1e-6)), int(np.ceil((t1 - b0) / S16 - 1e-6))):
        t = b0 + k * S16
        before, at = (td >= t - 0.045) & (td < t - 0.005), (td >= t - 0.01) & (td < t + 0.035)
        if not at.any() or not before.any():
            continue
        jump = [e[at].max() - e[before].min() for e in bands]
        if max(jump) >= DRUM_HIT:
            hits.append((t, k % 16, "K" if jump[0] >= jump[1] else "S", max(jump), min(jump[0], jump[1]) >= BIG_HIT))
    return hits


def drum_notes(lv, t0, t1, busy=None):
    """Notes on the drum hits in [t0, t1).
    HYPER plays every hit (sixteenths, rolls alternate thumbs), NORMAL the hits on eighths,
    EASY the hits on beats (eighths in a roll bar unless a thumb is holding). busy = (thumb, until): that thumb is
    holding a long note. Lanes: kick on the outer lane, snare/hats inner; in roll bars the thumbs trill
    outer lanes one beat, inner lanes the next; elsewhere each thumb swaps lanes every hit."""
    hits = drum_hits(t0, t1)
    per_bar = {}
    bar_of = lambda t: int(np.floor((t - b0) / (4 * BEAT) + 1e-6))
    for t, *_ in hits:
        per_bar[bar_of(t)] = per_bar.get(bar_of(t), 0) + 1
    picked = []
    for t, st, kind, db, big in hits:
        roll = per_bar[bar_of(t)] >= 12
        if lv == HYPER or (lv == NORMAL and st % 2 == 0) or (lv == EASY and st % (2 if roll and not busy else 4) == 0):
            picked.append((t, st, kind, big, roll))
    last = [-9.0, -9.0]
    if busy:
        last[busy[0]] = busy[1] - THUMB_GAP           # that thumb is free again at busy[1]
    prev_hand, prev_t, lane_at, notes = 1, -9.0, {}, []
    for i, (t, st, kind, big, roll) in enumerate(picked):
        free = [h for h in (0, 1) if t - last[h] >= THUMB_GAP - 1e-6]
        if not free:
            continue
        gap_next = picked[i + 1][0] - t if i + 1 < len(picked) else 9.0
        if big and lv > EASY and st % 4 == 0 and len(free) == 2 and t - prev_t >= 0.17 and gap_next >= 0.17:
            pair = (0, 3) if kind == "K" else (1, 2)
            notes += [[t, pair[0]], [t, pair[1]]]
            last = [t, t]; prev_t = t
            continue
        want = 1 - prev_hand if t - prev_t < 0.3 else (0 if kind == "K" else 1)
        h = want if want in free else free[0]
        outer = kind == "K"
        if roll and not busy:
            outer = int(round((t - b0) / BEAT - st % 4 / 4)) % 2 == 0
        elif lane_at.get(h):
            outer = lane_at[h][1] in (1, 2)            # swap to the thumb's other lane
        lane = (0 if outer else 1) if h == 0 else (3 if outer else 2)
        notes.append([t, lane])
        last[h] = t; prev_hand, prev_t = h, t; lane_at[h] = (t, lane)
    return [[round(float(t), 3), int(l)] for t, l in notes]


def bar_t(b):
    return b0 + b * 4 * BEAT


def add_drum_bars(out, lv):
    """Fills, rolls and the dance break (--drum-bars): the notes there are the drum hits."""
    if not args.drums or not args.drum_bars:
        return out
    for rng in args.drum_bars.split(","):
        a_, b_ = (int(x) for x in rng.split(":"))
        t0, t1 = bar_t(a_), min(bar_t(b_), end + 1e-3)
        if t0 >= t1:
            continue
        out = [n for n in out if not (t0 - 1e-3 <= n[0] < t1 - 1e-3) and not (len(n) > 2 and n[0] < t1 and n[0] + n[2] > t0)]
        out += drum_notes(lv, t0, t1)
    return sorted(out, key=lambda n: (n[0], n[1]))


def add_rush(out, lv):
    """Build-up before "Go": the snare roll and fill, note for note on HYPER (the sung "lives~" long note
    is cut where the roll starts so both thumbs are free); on NORMAL/EASY "lives~" stays held and the
    other thumb plays the hits on eighths/beats. Then a two-lane chord on the outer lanes when the x2 section starts."""
    if args.rush_from is None or hyper_t is None:
        return out
    r0 = b0 + args.rush_from * 4 * BEAT
    # clear the rush window (up to, not including, the drop downbeat)
    out = [n for n in out if not (r0 - 1e-3 <= n[0] < hyper_t - 1e-3)]
    busy = None
    for n in out:
        if len(n) > 2 and n[0] < r0 < n[0] + n[2] + 0.12:
            if lv == HYPER and args.drums:
                n[2] = round(max(STEP, r0 - THUMB_GAP - n[0]), 3)
            else:
                busy = (n[1] // 2, n[0] + n[2] + 0.12)
    if args.drums:
        out += drum_notes(lv, r0, go_t - THUMB_GAP + 1e-3, busy)
    else:                                              # no drum track: plain alternating eighths
        out += [[round(float(t), 3), 1 + i % 2] for i, t in enumerate(np.arange(r0, go_t - THUMB_GAP, STEP))]
    out += [[round(float(go_t), 3), 0], [round(float(go_t), 3), 3]]
    return sorted(out, key=lambda n: (n[0], n[1]))


CALM = 0.3  # bars quieter than this (normalised loudness) get a long note across the bar


def add_holds(out, lv):
    """Long notes: every other bar of calm stretches (intro, verses, bridge), plus the end of each 8-bar phrase.

    A long note is [t, lane, dur]. Other notes in its span move off its lane; on NORMAL they are
    dropped so the player only has to hold."""
    hard = lv == HYPER
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

charts = {name: build(lv) for name, lv in LEVELS}
for name, c in charts.items():
    ts = sorted({n[0] for n in c})
    nh = sum(1 for n in c if len(n) > 2)
    print(f"{name}: {len(c)} notes ({nh} long), {len(c) / (ts[-1] - ts[0]):.2f} nps, {ts[0]:.2f}-{ts[-1]:.2f}s, lanes {np.bincount([n[1] for n in c])}")
json.dump({"bpm": args.bpm, "beat": round(BEAT, 6), "b0": round(float(b0), 4), "dur": round(dur, 3),
           "hyper": round(float(go_t), 3) if go_t is not None else None,
           "zones": zones,
           "finish": args.finish,
           **charts}, open(args.out, "w"), separators=(",", ":"))
