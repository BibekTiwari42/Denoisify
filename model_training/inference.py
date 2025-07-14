import torch
import soundfile as sf
import torchaudio.transforms as T
import os
from model import WaveUNet
import numpy as np

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_training/checkpoints/unet_best.pth"
INPUT_AUDIO_PATH = r"D:\Backend\Data\test\noisy_testset\p232_002.wav"
OUTPUT_AUDIO_PATH = "example_outputs/denoised_output.wav"

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 16384
OVERLAP = 4096  # 25% overlap
STEP = SEGMENT_LENGTH - OVERLAP

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
data, sr = sf.read(INPUT_AUDIO_PATH)
waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # shape: (1, L)

if sr != SAMPLE_RATE:
    print(f"Resampling from {sr} to {SAMPLE_RATE} Hz...")
    resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resampler(waveform)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

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

# Optional: match original volume level
max_amp_input = waveform[:, :original_len].abs().max()
max_amp_output = denoised_audio.abs().max()
if max_amp_output > 0:
    denoised_audio = denoised_audio * (max_amp_input / max_amp_output)

# -------- SAVE OUTPUT --------
os.makedirs(os.path.dirname(OUTPUT_AUDIO_PATH), exist_ok=True)
sf.write(OUTPUT_AUDIO_PATH, denoised_audio.squeeze(0).numpy(), SAMPLE_RATE)
print(f"✅ Denoised audio saved to: {OUTPUT_AUDIO_PATH}")
