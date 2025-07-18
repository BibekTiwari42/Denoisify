import numpy as np
import os
import wave

# WAV I/O utils

def read_wav(filename):
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        dtype = np.int16 if sampwidth == 2 else np.uint8
        samples = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            samples = samples[::n_channels]
        samples = samples.astype(np.float32)
        # Normalize to [-1, 1] for 16-bit PCM
        if sampwidth == 2:
            samples = samples / 32768.0
        return framerate, samples

def write_wav(filename, sr, data):
    data = np.clip(data, -1.0, 1.0)
    data_int16 = np.int16(data * 32767)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data_int16.tobytes())


def frame_signal(signal, frame_length, hop_length):
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.stack([signal[i*hop_length:i*hop_length+frame_length] for i in range(num_frames)], axis=1)
    return frames


def overlap_add(frames, hop_length):
    frame_length, num_frames = frames.shape
    signal_length = (num_frames - 1) * hop_length + frame_length
    signal = np.zeros(signal_length)
    for i in range(num_frames):
        signal[i*hop_length:i*hop_length+frame_length] += frames[:, i]
    return signal


def ssbse_denoise(
    input_path,
    output_path,
    frame_length=512,
    hop_length=256,
    noise_frames=6,
    noise_threshold=0.9
):
    # Read the noisy audio
    sr, signal = read_wav(input_path)

    # Frame the signal (shape: [frame_length, num_frames])
    frames = frame_signal(signal, frame_length, hop_length)

    # Estimate noise covariance from initial frames
    noise_matrix = frames[:, :noise_frames]
    Rn = np.cov(noise_matrix)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(Rn)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Find rank for noise subspace using energy threshold
    cumulative_energy = np.cumsum(eigvals) / np.sum(eigvals)
    rank = np.searchsorted(cumulative_energy, noise_threshold) + 1

    # Create projection matrix for signal subspace
    E_noise = eigvecs[:, :rank]
    P_noise = E_noise @ E_noise.T
    P_signal = np.eye(frame_length) - P_noise

    # Enhance each frame by projecting onto signal subspace
    enhanced_frames = P_signal @ frames

    # Reconstruct the signal using overlap-add
    enhanced_signal = overlap_add(enhanced_frames, hop_length)

    # Normalize
    enhanced_signal = enhanced_signal / np.max(np.abs(enhanced_signal))

    # Write the denoised audio
    write_wav(output_path, sr, enhanced_signal)

    print(f"✅ Denoised audio saved at: {output_path}")
    return enhanced_signal


# Example usage:
if __name__ == "__main__":
    input_file = "media/input/noisy_sample.wav"
    output_file = "media/output/ssbse_cleaned.wav"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    ssbse_denoise(input_file, output_file)
