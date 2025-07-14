# denoiser/model_loader.py

import torch
import torchaudio
import numpy as np
from model_training.model import WaveUNet
from ssbse import ssbse_denoise
import os
import soundfile as sf
import torchaudio.transforms as T

CHECKPOINT_PATH = os.path.join("model_training", "checkpoints", "unet_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once
model = WaveUNet(in_ch=1, out_ch=1).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

def denoise_audio(input_path, output_path):
    # Load audio
    data, sr = sf.read(input_path)
    waveform = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # shape: (1, L)

    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 16384
    OVERLAP = 4096  # 25% overlap
    STEP = SEGMENT_LENGTH - OVERLAP

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

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
    sf.write(output_path, denoised_audio.squeeze(0).numpy(), sr)
