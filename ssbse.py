import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import os


def read_wav(filename):
    sr, data = wav.read(filename)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768  # Convert to float in range [-1, 1]
    return sr, data


def write_wav(filename, sr, data):
    # Normalize to int16 for saving
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    wav.write(filename, sr, data)


def frame_signal(signal, frame_length, hop_length):
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.stack([signal[i * hop_length:i * hop_length + frame_length]
                       for i in range(num_frames)])
    return frames.T  # shape: (frame_length, num_frames)


def overlap_add(frames, hop_length):
    frame_length, num_frames = frames.shape
    signal = np.zeros(num_frames * hop_length + frame_length)
    for i in range(num_frames):
        signal[i * hop_length:i * hop_length + frame_length] += frames[:, i]
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
