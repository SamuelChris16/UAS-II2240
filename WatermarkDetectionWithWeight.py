import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Parameter dasar
# -----------------------------------
fs = 48000          # sample rate
duration = 2.0      # durasi sinyal
f = 440             # frekuensi sinyal sinus
seed = 1211         # seed berdasarkan tgl lahir
threshold = 1000    # ambang korelasi deteksi

# -----------------------------------
# Inisialisasi sinyal
# -----------------------------------
np.random.seed(seed)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
original_signal = 0.5 * np.sin(2 * np.pi * f * t)

# Pseudorandom spread code
spread_code = np.random.choice([-1, 1], size=original_signal.shape)

# -----------------------------------
# Variasi bobot watermark
# -----------------------------------
weights = [0.001, 0.01, 0.05, 0.1, 0.2]
correlations = []

# -----------------------------------
# Proses embedding + deteksi
# -----------------------------------
for w in weights:
    watermarked = original_signal + w * spread_code
    correlation = np.dot(watermarked, spread_code)
    correlations.append(correlation)
    status = "TERDETEKSI ✅" if abs(correlation) > threshold else "TIDAK TERDETEKSI ❌"
    print(f"Bobot: {w:.3f} → Korelasi: {correlation:.2f} → {status}")

# -----------------------------------
# Visualisasi Hasil Deteksi
# -----------------------------------
plt.figure(figsize=(10, 5))
plt.plot(weights, correlations, marker='o')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold Deteksi')
plt.title("Efek Bobot terhadap Nilai Korelasi Deteksi Watermark")
plt.xlabel("Bobot Watermark")
plt.ylabel("Nilai Korelasi")
plt.grid(True)
plt.legend()
plt.show()
