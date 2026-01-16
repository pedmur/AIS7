import os
import numpy as np
import librosa
import soundfile as sf

def split_wavs_to_fixed_length(
    input_dir,
    output_dir,
    target_frames=150,
    n_fft=2048,
    hop_length=512,
    sr=None
):
    """
    Splits WAV files into fixed-length chunks that produce
    exactly `target_frames` mel-spectrogram frames.

    Output WAVs are safe to use for LOSO or file-level splitting.
    """

    os.makedirs(output_dir, exist_ok=True)

    samples_per_chunk = n_fft + (target_frames - 1) * hop_length
    print(f"Samples per chunk: {samples_per_chunk}")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".wav"):
            continue

        in_path = os.path.join(input_dir, fname)
        audio, sr_loaded = librosa.load(in_path, sr=sr)

        sr_use = sr_loaded
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / samples_per_chunk))

        base = os.path.splitext(fname)[0]

        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = audio[start:end]

            if len(chunk) < samples_per_chunk:
                chunk = np.pad(chunk, (0, samples_per_chunk - len(chunk)))

            out_name = f"{base}_chunk{i:03d}.wav"
            out_path = os.path.join(output_dir, out_name)

            sf.write(out_path, chunk, sr_use)

        print(f"{fname}: {num_chunks} chunks")

split_wavs_to_fixed_length(
    input_dir="FinalAssignment/sound_database/Claps/",
    output_dir="FinalAssignment/dataset/Claps/"
)

split_wavs_to_fixed_length(
    input_dir="FinalAssignment/sound_database/Keyboard/",
    output_dir="FinalAssignment/dataset/Keyboard/"
)

split_wavs_to_fixed_length(
    input_dir="FinalAssignment/sound_database/Keys/",
    output_dir="FinalAssignment/dataset/Keys/"
)