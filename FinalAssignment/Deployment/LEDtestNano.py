import os
import time
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
import Jetson.GPIO as GPIO

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_audio_model.h5")
OUTPUT_FILENAME = os.path.join(BASE_DIR, "live_recording.wav")

# STRICT TIMING SETTINGS
RECORD_SECONDS = 3       # Exactly 3 seconds
SAMPLE_RATE = 44100      # Must match training data
CATEGORIES = ['Claps', 'Keyboard', 'Keys']
N_MELS = 128

# --- GPIO SETUP ---
LED_CLAPS = 11      # Board Pin 11
LED_KEYBOARD = 13   # Board Pin 13
LED_KEYS = 15       # Board Pin 15

def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_CLAPS, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LED_KEYBOARD, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LED_KEYS, GPIO.OUT, initial=GPIO.LOW)

def trigger_led(category):
    mapping = {'Claps': LED_CLAPS, 'Keyboard': LED_KEYBOARD, 'Keys': LED_KEYS}
    pin = mapping.get(category)
    
    if pin:
        print(f"--> Lighting LED for {category} (Pin {pin})")
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(4) 
        GPIO.output(pin, GPIO.LOW)
    else:
        print(f"No LED configured for: {category}")

# --- STRICT 3S RECORDING ---
def record_audio_fixed_duration():
    print(f"\n[INFO] Get ready...")
    time.sleep(1) # Give user a split second to prepare
    
    print(f"[REC]  Recording for {RECORD_SECONDS}s NOW!")
    
    # Calculate exact number of frames for 3 seconds
    frames = int(RECORD_SECONDS * SAMPLE_RATE)
    
    # Non-blocking recording start
    recording = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1)
    
    # Wait exactly for the duration
    sd.wait() 
    
    print("[STOP] Recording complete.")
    
    # Save (Overwrite)
    write(OUTPUT_FILENAME, SAMPLE_RATE, recording)

def preprocess_audio(file_path):
    # Load with strict duration enforcement
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=RECORD_SECONDS)
    
    # Pad if shorter than 3s (rare, but good for safety)
    target_length = int(RECORD_SECONDS * SAMPLE_RATE)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
        
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, fmax=8000)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
    mel_spec = mel_spec[..., np.newaxis]
    return mel_spec[np.newaxis, ...]

# --- MAIN ---
def main():
    setup_gpio()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file missing.")

    print("[INFO] Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print ("[INFO] Model loaded successfully.")

    try:
        # 1. Capture 3s
        record_audio_fixed_duration()
        
        # 2. Process
        spectrogram = preprocess_audio(OUTPUT_FILENAME)
        
        # 3. Predict
        prediction = model.predict(spectrogram)
        class_index = np.argmax(prediction)
        result_label = CATEGORIES[class_index]
        
        print(f"\n>>> RESULT: {result_label} <<<\n")
        
        # 4. LED
        trigger_led(result_label)
        
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()