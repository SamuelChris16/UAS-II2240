import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ==== 1. Konversi WAV ke MP3 ====
def convert_wav_to_mp3(wav_path, mp3_path, bitrate="128k"):
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, mp3_path], check=True)

# ==== 2. Konversi MP3 ke WAV ====
def convert_mp3_to_wav(mp3_path, wav_path):
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)

# ==== 3. Fungsi visualisasi ====
def analyze_audio(wav_path, title, color='blue'):
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    # Normalisasi data
    data = data / np.max(np.abs(data))

    time = np.linspace(0, len(data)/rate, num=len(data))
    freq = np.fft.rfftfreq(len(data), 1/rate)
    spectrum = np.abs(np.fft.rfft(data))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Plot waktu
    axs[0].plot(time, data, color=color, linewidth=0.8)
    axs[0].set_title(f"{title} - Gelombang Waktu")
    axs[0].set_xlabel("Waktu (detik)")
    axs[0].set_ylabel("Amplitudo")
    axs[0].set_ylim(-1.1, 1.1)  # batas tetap agar bisa dibandingkan

    # Plot spektrum
    axs[1].plot(freq, spectrum, color=color, linewidth=0.8)
    axs[1].set_xlim(0, 20000)  # fokus hanya sampai 20 kHz
    axs[1].set_yscale("log")   # pakai log scale agar perbedaan kecil kelihatan
    axs[1].set_title(f"{title} - Spektrum Frekuensi (Log Scale)")
    axs[1].set_xlabel("Frekuensi (Hz)")
    axs[1].set_ylabel("Magnitudo (log)")

    plt.tight_layout()
    plt.show()

# ==== 4. Eksekusi ====
wav_input = "sample_audio.wav"
mp3_output = "compressed.mp3"
wav_from_mp3 = "compressed_back.wav"

convert_wav_to_mp3(wav_input, mp3_output)
convert_mp3_to_wav(mp3_output, wav_from_mp3)

analyze_audio(wav_input, "Original (WAV)", color='blue')
analyze_audio(wav_from_mp3, "Compressed (MP3 → WAV)", color='red')

# ==== 5. Bandingkan ukuran ====
size_wav = os.path.getsize(wav_input) / 1024
size_mp3 = os.path.getsize(mp3_output) / 1024

print(f"Ukuran WAV : {size_wav:.2f} KB")
print(f"Ukuran MP3 : {size_mp3:.2f} KB")
print(f"Rasio Kompresi : {size_wav / size_mp3:.2f}x")
    