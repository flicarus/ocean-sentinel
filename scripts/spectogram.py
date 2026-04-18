import librosa
import numpy as np

y, sr = librosa.load('data/mbari/sample_60s.wav', sr=None)
print(f"Sample rate: {sr} Hz")
print(f"Długość: {len(y) / sr:.1f} sekund")
print(f"Min/Max amplitude: {y.min():.4f} / {y.max():.4f}")

import matplotlib.pyplot as plt
S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128,
    fmax=1000,
    hop_length=512
)

S_db = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(14, 5))
plt.figure(figsize=(14, 5))
librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', fmax=1000)
plt.colorbar(format='%+2.0f dB')
plt.title('MBARI Monterey Bay — 2024-01-01 00:00 UTC — mel spectrogram (0-1kHz)')
plt.tight_layout()
plt.savefig('data/spectrograms/mbari_sample_mel.png', dpi=150)
print("Saved: data/spectrograms/mbari_sample_mel.png")


# STFT spectrogram — linear frequency scale, better resolution for ship noise band
