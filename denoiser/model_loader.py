# denoiser/model_loader.py

import torch
import numpy as np
import os
import wave
from model_training.model import WaveUNet
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors
import matplotlib.pyplot as plt
from denoiser.voice_preserving_mmse import mmse_stsa_voice_preserving, mmse_stsa_minimal, no_postprocessing
from denoiser.mmse_stsa_fixed import mmse_stsa_conservative

CHECKPOINT_PATH = os.path.join("model_training", "checkpoints", "unet_version2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WAV I/O utils

def read_wav(filename):
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        
        # Preserve original audio properties
        original_params = {
            'channels': n_channels,
            'sampwidth': sampwidth,
            'framerate': framerate
        }
        
        # Handle different bit depths properly
        if sampwidth == 1:
            dtype = np.uint8
            samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            samples = (samples - 128) / 128.0  # Convert to -1 to 1 range
        elif sampwidth == 2:
            dtype = np.int16
            samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            samples = samples / 32768.0  # Convert to -1 to 1 range
        elif sampwidth == 3:
            # 24-bit audio (3 bytes per sample)
            dtype = np.uint8
            raw_bytes = np.frombuffer(frames, dtype=dtype)
            # Convert 24-bit to 32-bit integers
            samples = np.zeros(len(raw_bytes) // 3, dtype=np.int32)
            for i in range(len(samples)):
                samples[i] = (raw_bytes[i*3] | (raw_bytes[i*3+1] << 8) | (raw_bytes[i*3+2] << 16))
                if samples[i] >= 0x800000:  # Handle negative values
                    samples[i] -= 0x1000000
            samples = samples.astype(np.float32) / 8388608.0  # Convert to -1 to 1 range
        else:
            # 32-bit float or other formats
            dtype = np.float32
            samples = np.frombuffer(frames, dtype=dtype)
        
        # Handle multi-channel audio (convert to mono by averaging channels)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)
            samples = np.mean(samples, axis=1)  # Average channels instead of just taking first
            
        return samples, original_params

def write_wav(filename, samples, original_params):
    """Write WAV file preserving original quality parameters"""
    samples = np.clip(samples, -1.0, 1.0)
    
    # Preserve original bit depth and sample rate
    sampwidth = original_params.get('sampwidth', 2)
    framerate = original_params.get('framerate', 44100)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Keep mono for now, but preserve quality
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        
        # Convert samples based on original bit depth
        if sampwidth == 1:
            # 8-bit unsigned
            samples_int = ((samples * 128.0) + 128).astype(np.uint8)
        elif sampwidth == 2:
            # 16-bit signed
            samples_int = (samples * 32767).astype(np.int16)
        elif sampwidth == 3:
            # 24-bit signed
            samples_int = (samples * 8388607).astype(np.int32)
            # Convert to 3-byte format
            bytes_array = np.zeros(len(samples_int) * 3, dtype=np.uint8)
            for i, sample in enumerate(samples_int):
                if sample < 0:
                    sample += 0x1000000
                bytes_array[i*3] = sample & 0xFF
                bytes_array[i*3+1] = (sample >> 8) & 0xFF
                bytes_array[i*3+2] = (sample >> 16) & 0xFF
            wf.writeframes(bytes_array.tobytes())
            return
        else:
            # Default to 16-bit if unknown
            samples_int = (samples * 32767).astype(np.int16)
            
        wf.writeframes(samples_int.tobytes())

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

def denoise_audio(input_path, output_path, processing_method='voice_preserving'):
    """
    Denoise audio using WaveUNet model with selectable post-processing options
    
    Args:
        input_path: Path to input noisy audio file
        output_path: Path to save denoised audio
        processing_method: One of 'voice_preserving', 'standard', 'aggressive', 'waveunet_only'
    """
    # Load audio
    data, original_params = read_wav(input_path)
    sr = original_params['framerate']
    print(f"Input audio stats - Max: {np.max(np.abs(data)):.3f}, RMS: {np.sqrt(np.mean(data**2)):.3f}")
    print(f"Original audio quality: {original_params['sampwidth']*8}-bit, {sr}Hz, {original_params['channels']} channel(s)")
    
    # Plot input waveform and spectrogram
    import os
    current_dir = os.getcwd()
    plot_waveform(data, sr, 'Input Audio Waveform', os.path.join(current_dir, 'input_waveform.png'), 'red')
    plot_spectrogram(data, sr, 'Input Audio', os.path.join(current_dir, 'input_spectrogram.png'))
    # Convert to mono if stereo (already handled in read_wav)
    waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # shape: (1, L)
    
    # Model was trained at 16kHz - resample for processing but preserve original for output
    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16384
    OVERLAP = 4096  # 25% overlap
    STEP = SEGMENT_LENGTH - OVERLAP
    
    # Store original sample rate for final output
    original_sr = sr
    
    # Resample to 16kHz for model processing (high quality resampling)
    if sr != SAMPLE_RATE:
        print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz for model processing")
        duration = waveform.shape[1] / sr
        new_length = int(duration * SAMPLE_RATE)
        # Use high-quality linear interpolation for resampling
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
    
    # Resample back to original sample rate if needed
    if original_sr != SAMPLE_RATE:
        print(f"Resampling back from {SAMPLE_RATE}Hz to {original_sr}Hz")
        duration = denoised_audio.shape[1] / SAMPLE_RATE
        new_length = int(duration * original_sr)
        denoised_resampled = torch.from_numpy(np.interp(
            np.linspace(0, denoised_audio.shape[1], new_length, endpoint=False),
            np.arange(denoised_audio.shape[1]),
            denoised_audio.squeeze(0).numpy()
        ).astype(np.float32)).unsqueeze(0)
        denoised_audio = denoised_resampled
        sr = original_sr
    
    # Save output preserving original sample rate and bit depth
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_wav(output_path, denoised_audio.squeeze(0).numpy(), original_params)
    
    # Plot denoised output waveform and spectrogram
    denoised_np = denoised_audio.squeeze(0).numpy()
    print(f"Denoised audio stats - Max: {np.max(np.abs(denoised_np)):.3f}, RMS: {np.sqrt(np.mean(denoised_np**2)):.3f}")
    plot_waveform(denoised_np, original_sr, 'WaveUNet Denoised Output', os.path.join(current_dir, 'denoised_waveform.png'), 'green')
    plot_spectrogram(denoised_np, original_sr, 'Denoised Output', os.path.join(current_dir, 'denoised_output_spectrogram.png'))
    
    # Apply post-processing based on selected method using original sample rate
    if processing_method == 'voice_preserving':
        print("Applying voice-preserving MMSE-STSA...")
        postprocessed_audio = mmse_stsa_voice_preserving(
            denoised_audio.squeeze(0).numpy(),
            original_sr,  # Use original sample rate for post-processing
            Gmin=0.3,  # Higher minimum gain to preserve voice characteristics
            alpha=0.98,
            beta=0.98
        )
    elif processing_method == 'standard':
        print("Applying standard MMSE-STSA...")
        postprocessed_audio = mmse_stsa_conservative(
            denoised_audio.squeeze(0).numpy(),
            original_sr,
            Gmin=0.1,  # Balanced noise reduction
            alpha=0.98,
            beta=0.98
        )
    elif processing_method == 'aggressive':
        print("Applying aggressive MMSE-STSA...")
        postprocessed_audio = mmse_stsa_conservative(
            denoised_audio.squeeze(0).numpy(),
            original_sr,
            Gmin=0.05,  # Very aggressive noise reduction
            alpha=0.99,
            beta=0.99
        )
    elif processing_method == 'waveunet_only':
        print("Using WaveUNet output only (no post-processing)...")
        postprocessed_audio = no_postprocessing(
            denoised_audio.squeeze(0).numpy(),
            original_sr
        )
    else:
        # Default to voice-preserving
        print("Using default voice-preserving MMSE-STSA...")
        postprocessed_audio = mmse_stsa_voice_preserving(
            denoised_audio.squeeze(0).numpy(),
            original_sr,
            Gmin=0.3,
            alpha=0.98,
            beta=0.98
        )
    
    # Post-processing preserves proper levels - no additional normalization needed
    
    print(f"Post-processed audio stats - Max: {np.max(np.abs(postprocessed_audio)):.3f}, RMS: {np.sqrt(np.mean(postprocessed_audio**2)):.3f}")
    
    write_wav(os.path.join(current_dir, 'postprocessed_output_mmse_stsa.wav'), postprocessed_audio, original_params)
    plot_waveform(postprocessed_audio, original_sr, 'MMSE-STSA Post-Processed Output', os.path.join(current_dir, 'postprocessed_waveform.png'), 'blue')
    plot_spectrogram(postprocessed_audio, original_sr, 'MMSE-STSA Postprocessed Output', os.path.join(current_dir, 'postprocessed_output_mmse_stsa_spectrogram.png'))
    return denoised_audio.squeeze(0).numpy(), original_sr
