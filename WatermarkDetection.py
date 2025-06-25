import numpy as np
from scipy.io.wavfile import read
import os

# ----------------------------------------
# Parameter sesuai saat embedding
# ----------------------------------------
fs = 48000           # sample rate
duration = 2.0       # durasi (detik)
seed = 1211          # seed dari tanggal & bulan lahir (misal: 23 Juli -> 2307)
threshold = 1000     # ambang korelasi untuk menyatakan watermark terdeteksi

# ----------------------------------------
# Fungsi deteksi watermark
# -------------------   ---------------------
def detect_watermark(filename, seed, threshold=1000):
    # Baca file audio
    sr, signal = read(filename)
    signal = signal.astype(np.float32)

    # Regenerate pseudorandom spread code (harus sama seperti saat embedding)
    np.random.seed(seed)
    spread_code = np.random.choice([-1, 1], size=signal.shape)

    # Korelasi (dot product)
    correlation = np.dot(signal, spread_code)

    # Deteksi
    detected = abs(correlation) > threshold
    print(f"File: {os.path.basename(filename)}")
    print(f"Nilai Korelasi: {correlation:.2f}")
    print(f"Status Watermark: {'TERDETEKSI ✅' if detected else 'TIDAK TERDETEKSI ❌'}\n")
    return correlation, detected

# ----------------------------------------
# Uji pada file audio
# ----------------------------------------
detect_watermark("original.wav", seed)
detect_watermark("result/watermarked_001.wav", seed)
detect_watermark("result/watermarked_01.wav", seed)
