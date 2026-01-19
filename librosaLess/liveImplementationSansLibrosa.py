import sounddevice as sd
import numpy as np
#import Jetson.GPIO as GPIO
import tensorflow as tf
import time
#from IPython.display import clear_output
from scipy.signal import stft
from scipy.io import wavfile
import matplotlib.pyplot as plt


import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib_local'))
# import librosa

# --- Configuration ---
# FORMAT = pyaudio.paInt16
RATE = 44100
CHANNELS = 1
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 150
SAMPLES_PER_CHUNK = N_FFT + (TARGET_FRAMES - 1) * HOP_LENGTH
TEMP_WAV = "temp_recording/live_chunk.wav"

# Spectrogram settings (128x259)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
T_SAMP = 259

# --- GPIO SETUP ---
LED_CLAPS = 11      # Board Pin 11
LED_KEYBOARD = 13   # Board Pin 13
LED_KEYS = 15       # Board Pin 15

MODEL_PATH = "Best Models Folder/Yvette_1.h5"

CATEGORIES = ['Claps', 'Keyboard', 'Keys']

# Library-less alternative
# Convert frequency in Hz to Mel scale

# Library-less alternative
# Convert frequency in Hz to Mel scale
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

# Convert Mel scale back to frequency in Hz
def mel_to_hz(m):
    return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_filterbank(n_mels, n_fft, sample_rate, fmax):
    """
    Create a Mel filterbank matrix to convert FFT bins to Mel bins.
    
    Parameters:
        n_mels: number of Mel bands
        n_fft: FFT size
        sample_rate: sampling rate of audio
        fmax: max frequency to include in Mel scale
    
    Returns:
        filterbank: shape (n_mels, n_fft//2+1)
    """
    # Convert min and max frequencies to Mel scale
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(fmax)

    # Equally spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert Mel points back to Hz
    hz_points = mel_to_hz(mel_points)

    # Map Hz points to nearest FFT bin indices
    bin_points = np.round((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Initialize filterbank matrix
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    # Create triangular filters
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]     # left edge of triangle
        f_center = bin_points[m]       # center (peak) of triangle
        f_right = bin_points[m + 1]    # right edge of triangle

        # Avoid zero-width triangles
        if f_center == f_left:
            f_center += 1
        if f_right == f_center:
            f_right += 1

        # Rising slope
        for k in range(f_left, f_center):
            filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)

        # Falling slope
        for k in range(f_center, f_right):
            filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank

def generate_mel_spectrogram(
    audio_signal,  # path to WAV file
    sample_rate,   # expected sample rate
    n_mels,        # number of Mel bins
    fmax,          # max frequency for Mel scale
    n_fft=2048,    # FFT window size
    hop_length=256 # hop length between frames
):
    # --- Load WAV file ---
    sr, signal = wavfile.read(audio_signal)

    # Check that WAV matches expected sample rate
    if sr != sample_rate:
        raise ValueError("Provided sample_rate does not match WAV file")

    # If stereo, convert to mono by averaging channels
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    # Ensure signal is float32 for processing
    signal = signal.astype(np.float32)

    # --- Compute STFT ---
    # Returns frequency bins (f), time bins (t), and complex STFT matrix (Zxx)
    _, _, Zxx = stft(
        signal,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        padded=False,
        boundary=None
    )

    # --- Compute power spectrogram ---
    # Take magnitude squared of STFT
    power_spec = np.abs(Zxx) ** 2

    # --- Apply Mel filterbank ---
    mel_fb = mel_filterbank(n_mels, n_fft, sample_rate, fmax)
    # Multiply filterbank with power spectrogram
    mel_spec = np.dot(mel_fb, power_spec)

    # --- Convert to decibels (dB) ---
    mel_spec = 10.0 * np.log10(mel_spec + 1e-10)  # avoid log(0)

    # --- Normalize to [0, 1] ---
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max() + 1e-10

    # --- Add channel dimension for ML ---
    mel_spec = mel_spec[:, :, np.newaxis]

    return mel_spec

def save_wav_file():
    # Record EXACT number of samples
    audio_float = sd.rec(
        frames=SAMPLES_PER_CHUNK,
        samplerate=RATE,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()
    # Convert float32 → int16 (pyaudio.paInt16 equivalent)
    audio_int16 = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    # Save WAV
    wavfile.write(TEMP_WAV, RATE, audio_int16)

    print(f"Saved spectrogram-aligned chunk to {TEMP_WAV}")
    
print("Available audio devices:\n")
print(sd.query_devices())

default_input = sd.default.device[0]
device_info = sd.query_devices(default_input)

print(f"Using microphone device #{default_input}:")
print(f"  Name: {device_info['name']}")
print(f"  Channels: {device_info['max_input_channels']}")
print(f"  Default samplerate: {device_info['default_samplerate']}")

# --- Load Model ---
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# --- LED trigger function ---
def trigger_led(category):
    mapping = {'Claps': LED_CLAPS, 'Keyboard': LED_KEYBOARD, 'Keys': LED_KEYS}
    pin = mapping.get(category)
    
    if pin:
        # GPIO.output(pin, GPIO.HIGH)
        time.sleep(4)
        # GPIO.output(pin, GPIO.LOW)
    else:
        print(f"No LED configured for: {category}")

# --- Continuous prediction loop ---
print("\n--- Starting Continuous Prediction Loop ---")

try:
    while True:
        print("Press enter to record")
        input("Ready...")

        print("Recording...", end="\r")

        # --- 1. Record audio from microphone ---
        save_wav_file()

        # --- 2. Generate Mel spectrogram ---
        mel_spec = generate_mel_spectrogram(
            audio_signal=TEMP_WAV,
            sample_rate=RATE,
            n_mels=N_MELS,
            fmax=8000,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # --- Pad or truncate to T_SAMP ---
        time_steps = mel_spec.shape[1]
        if time_steps < T_SAMP:
            pad_width = T_SAMP - time_steps
            mel_spec = np.pad(mel_spec, ((0,0),(0,pad_width),(0,0)), mode='constant')
        elif time_steps > T_SAMP:
            mel_spec = mel_spec[:, :T_SAMP, :]

        plt.figure(figsize=(6, 4))

        # mel_spec shape: (128, 150, 1)
        plt.imshow(
            mel_spec[:, :, 0],
            aspect='auto',
            origin='lower',
            interpolation='nearest'
        )

        print(mel_spec.shape)

        plt.title("Live Mel Spectrogram (Model Input)")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.colorbar(label="Normalized energy")

        plt.tight_layout()
        plt.show()

        mel_spec = np.expand_dims(mel_spec, axis=0)
        print(mel_spec.shape)

        # --- 4. Predict ---
        prediction = model.predict(mel_spec, verbose=1)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        class_name = CATEGORIES[predicted_index]

        print(f"Prediction: Class {class_name} ({confidence:.2%} confidence)")

        # --- 5. Trigger LED ---
        trigger_led(class_name)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    print("done")
    # GPIO.cleanup()