# denoiser/model_loader.py

import torch
import numpy as np
import os
import wave
from model_training.model import WaveUNet

CHECKPOINT_PATH = os.path.join("model_training", "checkpoints", "unet_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if sampwidth == 2:
            samples = samples / 32768.0
        return samples, framerate

def write_wav(filename, samples, framerate):
    samples = samples.astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(samples.tobytes())

# Load model once
model = WaveUNet(in_ch=1, out_ch=1, depth=5, base_ch=24).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

def denoise_audio(input_path, output_path, use_ssbse_only=False):
    """
    Denoise audio using WaveUNet model - clean working version
    """
    # Load audio
    data, sr = read_wav(input_path)
    # Convert to mono if stereo (already handled in read_wav)
    waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # shape: (1, L)
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16384
    OVERLAP = 4096  # 25% overlap
    STEP = SEGMENT_LENGTH - OVERLAP
    # Resample if needed
    if sr != SAMPLE_RATE:
        duration = waveform.shape[1] / sr
        new_length = int(duration * SAMPLE_RATE)
        waveform = torch.from_numpy(np.interp(
            np.linspace(0, waveform.shape[1], new_length, endpoint=False),
            np.arange(waveform.shape[1]),
            waveform.squeeze(0).numpy()
        ).astype(np.float32)).unsqueeze(0)
        sr = SAMPLE_RATE
    original_len = waveform.shape[1]
    # Pad audio
    num_chunks = (original_len - OVERLAP + STEP - 1) // STEP
    pad_len = max(0, num_chunks * STEP + OVERLAP - original_len)
    waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    # Overlap-add inference
    window = torch.hann_window(SEGMENT_LENGTH).to(DEVICE)
    denoised_audio = torch.zeros_like(waveform)
    normalization = torch.zeros_like(waveform)
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * STEP
            end = start + SEGMENT_LENGTH
            chunk = waveform[:, start:end].to(DEVICE)
            input_tensor = chunk.unsqueeze(0)  # shape: (1, 1, SEGMENT_LENGTH)
            output = model(input_tensor).squeeze(0).squeeze(0)  # shape: (SEGMENT_LENGTH,)
            denoised_audio[:, start:end] += (output * window).unsqueeze(0).cpu()
            normalization[:, start:end] += window.unsqueeze(0).cpu()
    # Normalize overlap-add
    denoised_audio /= normalization.clamp(min=1e-8)
    denoised_audio = denoised_audio[:, :original_len]
    # Match original volume
    max_amp_input = waveform[:, :original_len].abs().max()
    max_amp_output = denoised_audio.abs().max()
    if max_amp_output > 0:
        denoised_audio = denoised_audio * (max_amp_input / max_amp_output)
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save output
    write_wav(output_path, denoised_audio.squeeze(0).numpy(), sr)
    return denoised_audio.squeeze(0).numpy(), sr

def autocorrelation_pitch(waveform, sr, fmin=80, fmax=400):
    """
    Estimate the pitch (Hz) of a waveform using autocorrelation.
    """
    waveform = waveform - np.mean(waveform)
    corr = np.correlate(waveform, waveform, mode='full')
    corr = corr[len(corr)//2:]
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    corr[:min_lag] = 0
    peak = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch = sr / peak
    return pitch

def naive_pitch_shift(waveform, sr, semitones):
    """
    Naively shift the pitch by resampling (changes both pitch and speed).
    """
    factor = 2 ** (semitones / 12)
    n_samples = int(len(waveform) / factor)
    shifted = np.interp(
        np.linspace(0, len(waveform), n_samples, endpoint=False),
        np.arange(len(waveform)),
        waveform
    )
    return shifted

def pitch_normalize_pure_python(waveform, sample_rate, target_pitch_hz=220.0):
    """
    Normalize the pitch of the input waveform to the target pitch (in Hz) using only pure Python and numpy.
    Returns the pitch-normalized waveform (numpy array).
    """
    est_pitch = autocorrelation_pitch(waveform, sample_rate)
    print(f"Estimated pitch: {est_pitch:.2f} Hz")
    n_steps = 12 * np.log2(target_pitch_hz / est_pitch)
    print(f"Shifting pitch by {n_steps:.2f} semitones to reach {target_pitch_hz} Hz")
    y_shifted = naive_pitch_shift(waveform, sample_rate, n_steps)
    return y_shifted
