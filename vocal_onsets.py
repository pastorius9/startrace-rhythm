"""List sung syllable starts in a song window, measured on the vocal alone.

usage: python vocal_onsets.py <model.onnx> <audio> <T0> <T1>

1. Separates the vocal with an MDX-Net vocal model (UVR "Kim_Vocal_2.onnx", from the
   TRvlvr/model_repo GitHub releases), so drums and synths don't add false onsets.
2. Prints every loudness rise of the vocal (start = middle of the rise) with its pitch, and the
   pitch segments, so syllables sung on without a break show up where the pitch steps.
The SYLLABLES table in build_chart.py was picked from this output (needs librosa, onnxruntime).
"""
import sys

import librosa
import numpy as np
import onnxruntime as ort
import scipy.signal as ss

model, src, T0, T1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
pad0 = min(T0, 6.0)                                   # separate a little extra on both sides
y, sr = librosa.load(src, sr=44100, mono=False, offset=T0 - pad0, duration=T1 - T0 + pad0 + 6)
y = np.stack([y, y]) if y.ndim == 1 else y

# --- MDX-Net vocal separation (n_fft 7680, 3072 bins, 256 frames per chunk)
n_fft, dim_f, dim_t, hop = 7680, 3072, 256, 1024
chunk = hop * (dim_t - 1); trim = n_fft // 2; gen = chunk - 2 * trim
n = y.shape[1]
mix = np.concatenate([np.zeros((2, trim)), y, np.zeros((2, gen - n % gen + trim))], 1)
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
win = np.hanning(n_fft + 1)[:-1]
parts = []
for i in range(0, mix.shape[1] - 2 * trim, gen):
    w = mix[:, i:i + chunk]
    X = np.stack([librosa.stft(w[c], n_fft=n_fft, hop_length=hop, window=win, pad_mode="reflect")[:dim_f] for c in range(2)])
    o = sess.run(None, {sess.get_inputs()[0].name: np.stack([X[0].real, X[0].imag, X[1].real, X[1].imag])[None].astype(np.float32)})[0][0]
    Z = np.zeros((2, n_fft // 2 + 1, dim_t), complex)
    Z[0, :dim_f], Z[1, :dim_f] = o[0] + 1j * o[1], o[2] + 1j * o[3]
    parts.append(np.stack([librosa.istft(Z[c], n_fft=n_fft, hop_length=hop, window=win, length=chunk) for c in range(2)])[:, trim:-trim])
voc = librosa.resample(np.concatenate(parts, 1)[:, :n].mean(0), orig_sr=sr, target_sr=22050)
sr = 22050
t_of = lambda frames, h: frames * h / sr + T0 - pad0

# --- loudness rises of the vocal
h = 64
S = np.abs(librosa.stft(voc, n_fft=1024, hop_length=h))
fr = librosa.fft_frequencies(sr=sr, n_fft=1024)
e = np.convolve(np.sqrt((S[(fr > 120) & (fr < 6000)] ** 2).sum(0)), np.ones(5) / 5, "same")
db = 20 * np.log10(e + 1e-6); db -= db.max()
L = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=64, fmin=120, fmax=6000))
flux = np.convolve(np.maximum(0, np.diff(L, axis=1, prepend=L[:, :1])).mean(0), np.ones(3) / 3, "same")
f0, _, _ = librosa.pyin(voc, fmin=120, fmax=1000, sr=sr, frame_length=2048, hop_length=256)
midi = librosa.hz_to_midi(f0)
print("rises (start, dB rise, level, pitch):")
for p in ss.find_peaks(flux, height=np.percentile(flux, 80), distance=int(0.1 * sr / h))[0]:
    b = max(0, p - int(0.08 * sr / h)); b += int(np.argmin(db[b:p + 1]))
    t = t_of((b + p) / 2, h)
    if not T0 <= t < T1:
        continue
    k = int((p * h + 0.05 * sr) / 256)            # pitch 50 ms into the syllable
    m = np.nanmedian(midi[k:k + 6]) if np.any(~np.isnan(midi[k:k + 6])) else np.nan
    print(f"  {t:8.3f}  {db[p:p + int(0.06 * sr / h)].max() - db[b]:5.1f} dB  {db[p]:6.1f}  {librosa.midi_to_note(m) if m == m else '-'}")

print("pitch segments:")
rms = librosa.feature.rms(y=voc, frame_length=1024, hop_length=256)[0]
lv = 20 * np.log10(rms + 1e-6); lv -= lv.max()
mm = np.array([np.nanmedian(w) if np.any(~np.isnan(w)) else np.nan
               for w in (np.where(lv > -30, midi, np.nan)[max(0, i - 3):i + 4] for i in range(len(midi)))])
cur = None
for i, v in enumerate(list(mm) + [np.nan]):
    t = t_of(i, 256)
    if cur and (np.isnan(v) or abs(v - cur[2]) >= 0.7):
        if cur[1] - cur[0] >= 0.06 and T0 <= cur[0] < T1:
            print(f"  {cur[0]:8.3f}-{cur[1]:8.3f}  {librosa.midi_to_note(round(cur[2]))}")
        cur = None
    if not np.isnan(v):
        cur = [t, t, v] if cur is None else [cur[0], t, cur[2]]
