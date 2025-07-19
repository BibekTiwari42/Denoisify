import torch
import numpy as np
import os
import wave
import matplotlib.pyplot as plt
import sys
sys.path.append('..')  # Add parent directory to path
from model_training.model import WaveUNet
from denoiser.mmse_stsa import mmse_stsa

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../model_training/checkpoints/unet_best.pth"
INPUT_AUDIO_PATH = r"D:\denoisify\Data\test\noisy_testset\p232_007.wav"

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 16384
OVERLAP = 4096  # 25% overlap
STEP = SEGMENT_LENGTH - OVERLAP

# Extract base filename for output naming
INPUT_FILENAME = os.path.splitext(os.path.basename(INPUT_AUDIO_PATH))[0]
print(f"Processing audio file: {INPUT_FILENAME}")

# Create separate folder for each audio file
IMAGES_FOLDER = f"images/{INPUT_FILENAME}"
os.makedirs(IMAGES_FOLDER, exist_ok=True)
print(f"Output folder created: {IMAGES_FOLDER}")

# -------- WAV I/O UTILS --------
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
        # Normalize to [-1, 1] range
        if sampwidth == 2:  # 16-bit audio
            samples = samples / 32768.0
        return samples, framerate

def write_wav(filename, samples, framerate):
    # Scale float audio (-1 to 1) to int16 range (-32768 to 32767)
    samples = np.clip(samples, -1.0, 1.0)  # Clip to prevent overflow
    samples = (samples * 32767.0).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(samples.tobytes())

def plot_waveform(audio, sr, title, filename, color='blue'):
    """Plot waveform with time axis in seconds"""
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

def plot_spectrogram(wave, sr, title, filename):
    plt.figure(figsize=(12, 4))
    plt.specgram(wave, NFFT=1024, Fs=sr, noverlap=512, cmap='magma')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Frequency [Hz]', fontsize=12)
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# -------- LOAD MODEL --------
print("Loading WaveUNet model...")
model = WaveUNet(in_ch=1, out_ch=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# -------- LOAD & PREPROCESS AUDIO --------
print(f"Loading input audio: {INPUT_AUDIO_PATH}")
data, sr = read_wav(INPUT_AUDIO_PATH)
print(f"Input audio stats - Max: {np.max(np.abs(data)):.3f}, RMS: {np.sqrt(np.mean(data**2)):.3f}")

# Plot input waveform
plot_waveform(data, sr, f'Input Audio Waveform - {INPUT_FILENAME}', f'{IMAGES_FOLDER}/input_waveform.png', 'red')

waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # shape: (1, L)

if sr != SAMPLE_RATE:
    print(f"Resampling from {sr} to {SAMPLE_RATE} Hz...")
    duration = waveform.shape[1] / sr
    new_length = int(duration * SAMPLE_RATE)
    waveform = torch.from_numpy(np.interp(
        np.linspace(0, waveform.shape[1], new_length, endpoint=False),
        np.arange(waveform.shape[1]),
        waveform.squeeze(0).numpy()
    ).astype(np.float32)).unsqueeze(0)
    sr = SAMPLE_RATE

original_len = waveform.shape[1]

# -------- PAD AUDIO --------
num_chunks = (original_len - OVERLAP + STEP - 1) // STEP
pad_len = max(0, num_chunks * STEP + OVERLAP - original_len)
waveform = torch.nn.functional.pad(waveform, (0, pad_len))
print(f"Processing {num_chunks} chunks with overlap-add...")

# -------- INFERENCE WITH OVERLAP-ADD --------
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
        # Apply window to output and overlap-add
        denoised_audio[:, start:end] += (output * window).unsqueeze(0).cpu()
        normalization[:, start:end] += window.unsqueeze(0).cpu()

# -------- FINALIZE DENOISED OUTPUT --------
denoised_audio /= normalization.clamp(min=1e-8)
denoised_audio = denoised_audio[:, :original_len]

# Volume matching
max_amp_input = waveform[:, :original_len].abs().max()
max_amp_output = denoised_audio.abs().max()
if max_amp_output > 0:
    denoised_audio = denoised_audio * (max_amp_input / max_amp_output)

denoised_np = denoised_audio.squeeze(0).numpy()
print(f"Denoised audio stats - Max: {np.max(np.abs(denoised_np)):.3f}, RMS: {np.sqrt(np.mean(denoised_np**2)):.3f}")

# Plot denoised waveform
plot_waveform(denoised_np, SAMPLE_RATE, f'WaveUNet Denoised Output - {INPUT_FILENAME}', f'{IMAGES_FOLDER}/denoised_waveform.png', 'green')

# -------- POST-PROCESSING --------
print("Applying MMSE-STSA post-processing...")
postprocessed_audio = mmse_stsa(
    denoised_np,
    SAMPLE_RATE,
    Gmin=0.7,  # More signal preserved
    alpha=0.95,
    beta=0.95
)

# Normalize post-processed audio
if np.max(np.abs(postprocessed_audio)) > 0:
    postprocessed_audio = postprocessed_audio / np.max(np.abs(postprocessed_audio))
postprocessed_audio = postprocessed_audio * 0.9  # scale to 90% of full range

print(f"Post-processed audio stats - Max: {np.max(np.abs(postprocessed_audio)):.3f}, RMS: {np.sqrt(np.mean(postprocessed_audio**2)):.3f}")

# Plot post-processed waveform
plot_waveform(postprocessed_audio, SAMPLE_RATE, f'MMSE-STSA Post-Processed - {INPUT_FILENAME}', f'{IMAGES_FOLDER}/postprocessed_waveform.png', 'blue')

# -------- SAVE AUDIO FILES --------
print("Saving audio files...")
write_wav(f'{IMAGES_FOLDER}/denoised_output.wav', denoised_np, SAMPLE_RATE)
write_wav(f'{IMAGES_FOLDER}/postprocessed_output.wav', postprocessed_audio, SAMPLE_RATE)

# -------- CREATE COMPARISON PLOT --------
print("Creating comparison plots...")
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Input waveform
duration = len(data) / sr
time_axis = np.linspace(0, duration, len(data))
axes[0].plot(time_axis, data, color='red', linewidth=0.5)
axes[0].set_title(f'Input Audio (Noisy) - {INPUT_FILENAME}', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1.1, 1.1)

# Denoised waveform
duration = len(denoised_np) / SAMPLE_RATE
time_axis = np.linspace(0, duration, len(denoised_np))
axes[1].plot(time_axis, denoised_np, color='green', linewidth=0.5)
axes[1].set_title(f'WaveUNet Denoised Output - {INPUT_FILENAME}', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Amplitude', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-1.1, 1.1)

# Post-processed waveform
duration = len(postprocessed_audio) / SAMPLE_RATE
time_axis = np.linspace(0, duration, len(postprocessed_audio))
axes[2].plot(time_axis, postprocessed_audio, color='blue', linewidth=0.5)
axes[2].set_title(f'MMSE-STSA Post-Processed Output - {INPUT_FILENAME}', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time [seconds]', fontsize=12)
axes[2].set_ylabel('Amplitude', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig(f'{IMAGES_FOLDER}/waveform_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# -------- CREATE SPECTROGRAM COMPARISON --------
print("Creating spectrogram comparison...")
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Input spectrogram
axes[0].specgram(data, NFFT=1024, Fs=sr, noverlap=512, cmap='magma')
axes[0].set_title(f'Input Audio Spectrogram (Noisy) - {INPUT_FILENAME}', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Frequency [Hz]', fontsize=12)

# Denoised spectrogram
axes[1].specgram(denoised_np, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512, cmap='magma')
axes[1].set_title(f'WaveUNet Denoised Spectrogram - {INPUT_FILENAME}', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency [Hz]', fontsize=12)

# Post-processed spectrogram
axes[2].specgram(postprocessed_audio, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512, cmap='magma')
axes[2].set_title(f'MMSE-STSA Post-Processed Spectrogram - {INPUT_FILENAME}', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time [s]', fontsize=12)
axes[2].set_ylabel('Frequency [Hz]', fontsize=12)

plt.tight_layout()
plt.savefig(f'{IMAGES_FOLDER}/spectrogram_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ All plots and audio files generated successfully!")
print(f"\nGenerated files in '{IMAGES_FOLDER}' folder:")
print("- input_waveform.png")
print("- denoised_waveform.png") 
print("- postprocessed_waveform.png")
print("- waveform_comparison.png")
print("- spectrogram_comparison.png")
print("- denoised_output.wav")
print("- postprocessed_output.wav") 