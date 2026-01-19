import pyaudio
import numpy as np

import tensorflow as tf
import time
from IPython.display import clear_output

sys.path.insert(0, "./librosa")
import librosa

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3

# Spectrogram settings (128x259)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

MODEL_PATH = "models/model_2026-01-16_11-13-50.keras"

# --- Load Model ---
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Audio Setup ---
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

chunks_per_cycle = int(RATE / CHUNK * RECORD_SECONDS)

print("\n--- Starting Continuous Prediction Loop (0-1 Normalized) ---")

try:
    while True:
        print("Listening...", end="\r")
        
        # 1. Capture
        frames = []
        for i in range(0, chunks_per_cycle):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        # 2. Process to Float
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # 3. Generate Mel Spectrogram (in dB)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_float, sr=RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # --- 4. NORMALIZE TO [0, 1] ---
        # This scales the data so the quietest part is 0 and loudest is 1
        spec_min = np.min(mel_spec_db)
        spec_max = np.max(mel_spec_db)
        
        if spec_max - spec_min > 0:
            mel_spec_norm = (mel_spec_db - spec_min) / (spec_max - spec_min)
        else:
            mel_spec_norm = mel_spec_db # Handle silence
            
        # 5. Shape Enforcement (128x259)
        target_width = 259
        current_width = mel_spec_norm.shape[1]
        
        if current_width < target_width:
            padding = target_width - current_width
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, padding)))
        elif current_width > target_width:
            mel_spec_norm = mel_spec_norm[:, :target_width]
            
        # 6. Reshape and Predict
        # Input shape: (1, 128, 259, 1)
        input_data = np.expand_dims(mel_spec_norm, axis=0) 
        input_data = np.expand_dims(input_data, axis=-1) 
        
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        class_name = CATEGORIES[predicted_index]
        clear_output(wait=True)
        print(f"Prediction: Class {class_name} ({confidence:.2%} confidence)")

        plt.clf() # Clear previous plot
        
        # Visualize the normalized data entering the model
        librosa.display.specshow(mel_spec_norm, 
                                 sr=RATE, 
                                 hop_length=HOP_LENGTH, 
                                 x_axis='time', 
                                 y_axis='mel')
        
        plt.colorbar(format='%+2.2f')
        plt.title(f"Model Input: {class_name} ({confidence:.0%})")
        plt.tight_layout()
        plt.show()
        plt.close()
        

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()