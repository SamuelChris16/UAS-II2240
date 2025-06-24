import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# -------------------------------------
# Parameter Dasar
# -------------------------------------
fs = 48000               # Sample rate (48 kHz)
duration = 2.0           # Durasi sinyal (detik)
f = 440                  # Frekuensi sinusoidal (Hz)
seed = 1211              # Seed dari tanggal lahir 
weights = [0.01, 0.1]    # Dua bobot watermark

# -------------------------------------
# Generate Sinyal Sinusoidal Asli
# -------------------------------------
np.random.seed(seed)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
original_signal = 0.5 * np.sin(2 * np.pi * f * t)

# -------------------------------------
# Generate Watermark: Spread Spectrum Noise
# -------------------------------------
spread_code = np.random.choice([-1, 1], size=original_signal.shape)

# -------------------------------------
# Proses Embedding Watermark
# -------------------------------------
watermarked_signals = []
for weight in weights:
    watermarked = original_signal + weight * spread_code
    watermarked_signals.append(watermarked)

# -------------------------------------
# Simpan ke File Audio
# -------------------------------------
write("original.wav", fs, original_signal.astype(np.float32))
write("watermarked_001.wav", fs, watermarked_signals[0].astype(np.float32))
write("watermarked_01.wav", fs, watermarked_signals[1].astype(np.float32))

# -------------------------------------
# Tampilkan Grafik 1000 Sampel Pertama
# -------------------------------------
plt.figure(figsize=(14, 6))
plt.subplot(3, 1, 1)
plt.plot(t[:1000], original_signal[:1000])
plt.title("Sinyal Asli (440 Hz)")

plt.subplot(3, 1, 2)
plt.plot(t[:1000], watermarked_signals[0][:1000])
plt.title("Watermarked Signal (Weight = 0.01)")

plt.subplot(3, 1, 3)
plt.plot(t[:1000], watermarked_signals[1][:1000])
plt.title("Watermarked Signal (Weight = 0.1)")

plt.tight_layout()
plt.show()

# -------------------------------------
# Deteksi Watermark dengan Korelasi
# -------------------------------------
for i, wm in enumerate(watermarked_signals):
    corr = np.dot(wm, spread_code)
    print(f"Bobot watermark {weights[i]} → Korelasi deteksi = {corr:.2f}")
