#INITIALIZATION
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import stft
import pyaudio
import tensorflow as tf
from IPython.display import clear_output

MODEL_PATH = "Best Models Folder/Yvette_1.keras"
CATEGORIES = ['Claps', 'Keyboard', 'Keys']

RECORD_SECONDS = 3 #seconds     # X resolution of the spectrograms (basically image width) depends on sampling rate
SRATE = 44100
CHUNK = 1024
N_MELS = 128                    # Y resolution of the spectrograms (basically image height)
HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000
BLUR_SIGMA = 0.3

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
T_SAMP = 259

#--------------------------------------------------------------------------------------

# Functions for librosaless live testing
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

def librosaless_spectrogram(audio_signal, sample_rate, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX, target_width=259, blur_sigma=BLUR_SIGMA):
    """
    Unified pre-processing function.
    Returns:
        model_input: 4D tensor (1, 128, 259, 1) for the model
        plot_data: 2D array (128, 259) for visualization
    """
    
    # 1. Generate Mel Spectrogram
    # --- Compute STFT ---
    # Returns frequency bins (f), time bins (t), and complex STFT matrix (Zxx)
    _, _, Zxx = stft(
        audio_signal,
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
    mel_spec_db = 10.0 * np.log10(mel_spec + 1e-10)  # avoid log(0)
    
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



# ----- LIVE IMPLEMENTATION LOOP -----

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
        
        # --- 2. Generate Mel spectrogram ---
        input_tensor, plot_data = librosaless_spectrogram(
            audio_signal=audio_float,
            sample_rate=SRATE,
            n_mels=N_MELS,
            fmax=8000,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
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