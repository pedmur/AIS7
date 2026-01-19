import pyaudio
import matplotlib as plt
import numpy as np
import soundevice
import librosa
import tensorflow as tf
import time
from IPython.display import clear_output
from scipy.ndimage import gaussian_filter
import scipy.ndimage

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1

CATEGORIES = ['Claps', 'Keyboard', 'Keys']

RECORD_SECONDS = 3 #seconds     # X resolution of the spectrograms (basically image width) depends on sampling rate
SRATE = 44100
CHUNK = 1024
N_MELS = 128                    # Y resolution of the spectrograms (basically image height)
HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000
BLUR_SIGMA = 0.3

MODEL_PATH = "Best Models Folder/Yvette_1.keras"  #Model needs to be rebuilt
#MODEL_PATH = "best_audio_ model.keras"

# --- Spectrogram function ---
def convert_to_spectrogram(audio_signal, sample_rate, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX, target_width=259, blur_sigma=BLUR_SIGMA):
    """
    Unified pre-processing function.
    Returns:
        model_input: 4D tensor (1, 128, 259, 1) for the model
        plot_data: 2D array (128, 259) for visualization
    """
    
    # 1. Generate Mel Spectrogram
    # explicit params ensure consistency between Training and Live
    mel_spec = librosa.feature.melspectrogram(
        y=audio_signal, 
        sr=sample_rate, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length,
        fmax=fmax 
    )
    
    # 2. Convert to Decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Normalize to [0, 1]
    # (Matches your training function logic)
    spec_min = mel_spec_db.min()
    spec_max = mel_spec_db.max()
    
    if spec_max - spec_min > 0:
        mel_spec_norm = (mel_spec_db - spec_min) / (spec_max - spec_min)
    else:
        mel_spec_norm = mel_spec_db # Handle silence/errors
        
    # 4. Shape Enforcement (Pad or Trim to target_width)
    # This was missing in your training func but is CRITICAL for the model input shape
    current_width = mel_spec_norm.shape[1]
    
    if current_width < target_width:
        padding = target_width - current_width
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, padding)))
    elif current_width > target_width:
        mel_spec_norm = mel_spec_norm[:, :target_width]
        
    # 5. Optional Blur (Matches your training function)
    if blur_sigma > 0:
        mel_spec_norm = gaussian_filter(mel_spec_norm, sigma=BLUR_SIGMA)
        
    # 6. Prepare Final Output
    # Create 2D version for plotting
    plot_data = mel_spec_norm
    
    # Create 4D version for Model (Batch, Height, Width, Channels)
    # (128, 259) -> (1, 128, 259, 1)
    model_input = mel_spec_norm[np.newaxis, ..., np.newaxis]
    
    return model_input, plot_data



# --- Load Model ---
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {model}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Audio Setup ---
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=SRATE, input=True,
                    frames_per_buffer=CHUNK)

chunks_per_cycle = int(SRATE / CHUNK * RECORD_SECONDS)

print("\n--- Starting Continuous Prediction Loop---")

try:
    while True:
        # 1. Capture Audio
        frames = []
        for i in range(0, chunks_per_cycle):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError:
                continue
        
        if len(frames) < chunks_per_cycle:
            continue # Skip incomplete reads

        # 2. Process to Float
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # We get the prediction tensor AND the plot data in one go
        input_tensor, plot_data = convert_to_spectrogram(
            audio_signal=audio_float,
            sample_rate=SRATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            fmax=FMAX,
            target_width=259,
            blur_sigma=BLUR_SIGMA
        )
        
        # Training images (origin='lower') are read by Keras top-down (High->Low).
        # Live data is generated Low->High. We must flip it to match the model's training.
        
        # Flip the input tensor (Batch, Freq, Time, Channels) along axis 1 (Freq)
        input_tensor = np.flip(input_tensor, axis=1)
        
        # Flip the plot data so the visual matches what the model sees
        plot_data = np.flipud(plot_data)
        
        # 4. Predict
        prediction = model.predict(input_tensor, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if predicted_index < len(CATEGORIES):
            class_name = CATEGORIES[predicted_index]
        else:
            class_name = "Unknown"
            
        # 5. Visualization
        clear_output(wait=True)
        print(f"Prediction: {class_name} ({confidence:.2%} confidence)")

        plt.figure(figsize=(10, 4))
        plt.imshow(plot_data, 
                   origin='upper', # Matches Keras top-down reading
                   aspect='auto', 
                   cmap='gray')
        
        plt.colorbar(format='%+2.2f')
        plt.title(f"Model Input (Flipped): {class_name} ({confidence:.0%})")
        plt.tight_layout()
        plt.show()
        plt.close()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()