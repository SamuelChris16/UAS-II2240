import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import io

# === 1. Load File WAV Asli ===
wav_path = "sample_audio.wav"  
rate_wav, data_wav = wavfile.read(wav_path)
if data_wav.ndim > 1:  # stereo ke mono
    data_wav = data_wav.mean(axis=1).astype(data_wav.dtype)

# === 2. Kompresi ke MP3 ===
audio_segment = AudioSegment.from_wav(wav_path)
mp3_path = "compressed.mp3"
audio_segment.export(mp3_path, format="mp3", bitrate="128k")

# === 3. Load Kembali MP3 (konversi ke WAV di memori) ===
mp3_audio = AudioSegment.from_mp3(mp3_path)
wav_buffer = io.BytesIO()
mp3_audio.export(wav_buffer, format="wav")
wav_buffer.seek(0)
rate_mp3, data_mp3 = wavfile.read(wav_buffer)
if data_mp3.ndim > 1:
    data_mp3 = data_mp3.mean(axis=1).astype(data_mp3.dtype)

# === 4. Visualisasi Gelombang dan Spektrum ===
def plot_wave_and_spectrum(data, rate, title):
    duration = len(data) / rate
    time = np.linspace(0., duration, len(data))
    freq = np.fft.rfftfreq(len(data), 1/rate)
    spectrum = np.abs(np.fft.rfft(data))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(time, data)
    axs[0].set_title(f"{title} - Gelombang Waktu")
    axs[0].set_xlabel("Waktu (detik)")
    axs[0].set_ylabel("Amplitudo")

    axs[1].plot(freq, spectrum)
    axs[1].set_title(f"{title} - Spektrum Frekuensi")
    axs[1].set_xlabel("Frekuensi (Hz)")
    axs[1].set_ylabel("Magnitudo")

    plt.tight_layout()
    plt.show()

plot_wave_and_spectrum(data_wav, rate_wav, "Asli (WAV)")
plot_wave_and_spectrum(data_mp3, rate_mp3, "Terkonversi (MP3)")

# === 5. Bandingkan Ukuran File ===
size_wav = os.path.getsize(wav_path) / 1024
size_mp3 = os.path.getsize(mp3_path) / 1024
print(f"Ukuran file WAV : {size_wav:.2f} KB")
print(f"Ukuran file MP3 : {size_mp3:.2f} KB")
print(f"Rasio kompresi   : {size_wav / size_mp3:.2f}x")
