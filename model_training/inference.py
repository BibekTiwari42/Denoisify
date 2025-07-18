import torch
import numpy as np
import os
import wave
import struct
from model import WaveUNet

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_training/checkpoints/unet_best.pth"
INPUT_AUDIO_PATH = r"D:\denoisify\Data\test\noisy_testset\p232_009.wav"
OUTPUT_AUDIO_PATH = "example_outputs/denoised_output.wav"

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 16384
OVERLAP = 4096  # 25% overlap
STEP = SEGMENT_LENGTH - OVERLAP

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
        return samples.astype(np.float32), framerate

def write_wav(filename, samples, framerate):
    samples = samples.astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(samples.tobytes())

# -------- WIENER FILTERING FUNCTION --------
# def wiener_post_process(audio, sample_rate, noise_estimate=None, nperseg=512):
#     """
#     Apply Wiener filtering to the denoised audio.
#     Args:
#         audio (torch.Tensor): Denoised audio (shape: [1, time])
#         sample_rate (int): Sampling rate of the audio
#         noise_estimate (np.ndarray, optional): Precomputed noise power spectrum
#         nperseg (int): STFT window size
#     Returns:
#         torch.Tensor: Post-processed denoised audio
#     """
#     audio_np = audio.detach().cpu().numpy().squeeze(0)  # Shape: (time,)
#     # Compute STFT (simple implementation)
#     def stft(x, n_fft, hop_length):
#         x = np.pad(x, (0, n_fft - len(x) % n_fft), mode='constant')
#         frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop_length]
#         window = np.hanning(n_fft)
#         return np.fft.rfft(frames * window, axis=1)
#     def istft(X, n_fft, hop_length):
#         window = np.hanning(n_fft)
#         time_len = (X.shape[0] - 1) * hop_length + n_fft
#         x = np.zeros(time_len)
#         for i, frame in enumerate(X):
#             x[i*hop_length:i*hop_length+n_fft] += np.fft.irfft(frame) * window
#         return x
#     n_fft = nperseg
#     hop_length = nperseg // 2
#     Zxx = stft(audio_np, n_fft, hop_length)
#     # Estimate noise power (placeholder: use first 100ms or precomputed profile)
#     if noise_estimate is None:
#         noise_frames = Zxx[:int(0.1 * sample_rate / hop_length)]
#         noise_power = np.mean(np.abs(noise_frames)**2, axis=0, keepdims=True)
#     else:
#         noise_power = noise_estimate
#     # Apply Wiener filter
#     signal_power = np.abs(Zxx)**2
#     wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
#     Zxx_denoised = Zxx * wiener_gain
#     # Reconstruct time-domain signal
#     denoised = istft(Zxx_denoised, n_fft, hop_length)
#     # Ensure output length matches input
#     if denoised.shape[-1] > audio.shape[1]:
#         denoised = denoised[:audio.shape[1]]
#     elif denoised.shape[-1] < audio.shape[1]:
#         denoised = np.pad(denoised, (0, audio.shape[1] - denoised.shape[-1]), mode='constant')
#     return torch.tensor(denoised[None, :], dtype=torch.float32)

# -------- LOAD MODEL --------
print(f"Loading model from {MODEL_PATH}...")
model = WaveUNet(in_ch=1, out_ch=1).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# -------- LOAD & PREPROCESS AUDIO --------
print(f"Loading audio from {INPUT_AUDIO_PATH}...")
if not os.path.exists(INPUT_AUDIO_PATH):
    raise FileNotFoundError(f"File not found: {INPUT_AUDIO_PATH}")
data, sr = read_wav(INPUT_AUDIO_PATH)
waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # shape: (1, L)

if sr != SAMPLE_RATE:
    print(f"Resampling from {sr} to {SAMPLE_RATE} Hz...")
    # Simple resampling using numpy
    duration = waveform.shape[1] / sr
    new_length = int(duration * SAMPLE_RATE)
    waveform = torch.from_numpy(np.interp(
        np.linspace(0, waveform.shape[1], new_length, endpoint=False),
        np.arange(waveform.shape[1]),
        waveform.squeeze(0).numpy()
    ).astype(np.float32)).unsqueeze(0)
    sr = SAMPLE_RATE

# Convert to mono if stereo (already handled in read_wav)
original_len = waveform.shape[1]

# -------- PAD AUDIO --------
num_chunks = (original_len - OVERLAP + STEP - 1) // STEP
pad_len = max(0, num_chunks * STEP + OVERLAP - original_len)
waveform = torch.nn.functional.pad(waveform, (0, pad_len))
print(f"Padded waveform to length: {waveform.shape[1]}")

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

# -------- FINALIZE OUTPUT --------
# Avoid division by zero
denoised_audio /= normalization.clamp(min=1e-8)
# Trim to original length
denoised_audio = denoised_audio[:, :original_len]
# Apply Wiener filtering
# print("Applying Wiener filtering...")
# denoised_audio = wiener_post_process(denoised_audio, sample_rate=SAMPLE_RATE)
# Optional: match original volume level
max_amp_input = waveform[:, :original_len].abs().max()
max_amp_output = denoised_audio.abs().max()
if max_amp_output > 0:
    denoised_audio = denoised_audio * (max_amp_input / max_amp_output)
# -------- SAVE OUTPUT --------
os.makedirs(os.path.dirname(OUTPUT_AUDIO_PATH), exist_ok=True)
write_wav(OUTPUT_AUDIO_PATH, denoised_audio.squeeze(0).numpy(), SAMPLE_RATE)
print(f"✅ Denoised audio saved to: {OUTPUT_AUDIO_PATH}")