# denoiser/model_loader.py

import torch
import numpy as np
import os
import wave
from model_training.model import WaveUNet
import matplotlib.pyplot as plt
from denoiser.mmse_stsa import mmse_stsa

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
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32767).astype(np.int16)
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

def plot_spectrogram(wave, sr, title, filename):
    plt.figure(figsize=(10, 4))
    plt.specgram(wave, NFFT=1024, Fs=sr, noverlap=512, cmap='magma')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_waveform(audio, sr, title, filename, color='blue'):
    """Plot waveform with time axis in seconds and amplitude statistics"""
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, audio, color=color, linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time [seconds]', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.1, 1.1)
    
    # Add amplitude statistics
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    plt.text(0.02, 0.95, f'Max: {max_amp:.3f}\nRMS: {rms:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def denoise_audio(input_path, output_path):
    """
    Denoise audio using WaveUNet model - clean working version
    """
    # Load audio
    data, sr = read_wav(input_path)
    print(f"Input audio stats - Max: {np.max(np.abs(data)):.3f}, RMS: {np.sqrt(np.mean(data**2)):.3f}")
    
    # Plot input waveform and spectrogram
    plot_waveform(data, sr, 'Input Audio Waveform', 'input_waveform.png', 'red')
    plot_spectrogram(data, sr, 'Input Audio', 'input_spectrogram.png')
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
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_wav(output_path, denoised_audio.squeeze(0).numpy(), sr)
    
    # Plot denoised output waveform and spectrogram
    denoised_np = denoised_audio.squeeze(0).numpy()
    print(f"Denoised audio stats - Max: {np.max(np.abs(denoised_np)):.3f}, RMS: {np.sqrt(np.mean(denoised_np**2)):.3f}")
    plot_waveform(denoised_np, sr, 'WaveUNet Denoised Output', 'denoised_waveform.png', 'green')
    plot_spectrogram(denoised_np, sr, 'Denoised Output', 'denoised_output_spectrogram.png')
    # MMSE-STSA post-processing with improved parameters
    postprocessed_audio = mmse_stsa(
        denoised_audio.squeeze(0).numpy(),
        sr,
        Gmin=0.7,  # More signal preserved
        alpha=0.95,
        beta=0.95
    )
    if np.max(np.abs(postprocessed_audio)) > 0:
        postprocessed_audio = postprocessed_audio / np.max(np.abs(postprocessed_audio))
    postprocessed_audio = postprocessed_audio * 0.9  # scale to 90% of full range
    
    print(f"Post-processed audio stats - Max: {np.max(np.abs(postprocessed_audio)):.3f}, RMS: {np.sqrt(np.mean(postprocessed_audio**2)):.3f}")
    
    write_wav('postprocessed_output_mmse_stsa.wav', postprocessed_audio, sr)
    plot_waveform(postprocessed_audio, sr, 'MMSE-STSA Post-Processed Output', 'postprocessed_waveform.png', 'blue')
    plot_spectrogram(postprocessed_audio, sr, 'MMSE-STSA Postprocessed Output', 'postprocessed_output_mmse_stsa_spectrogram.png')
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
