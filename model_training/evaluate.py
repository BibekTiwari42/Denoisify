import os
import glob
import numpy as np
import wave
import torch
from model import WaveUNet
from dataset import ssbse_enhance

# WAV I/O utils

def read_wav(filename, sr=16000):
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
        # Resample if needed
        if framerate != sr:
            duration = len(samples) / framerate
            new_length = int(duration * sr)
            samples = np.interp(
                np.linspace(0, len(samples), new_length, endpoint=False),
                np.arange(len(samples)),
                samples
            ).astype(np.float32)
        return samples, sr

def write_wav(filename, samples, framerate):
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(samples.tobytes())

# ---------------------- CONFIG ---------------------- #
CONFIG = {
    "noisy_dir": "data/test/noisy",
    "clean_dir": "data/test/clean",
    "model_path": "model_training/checkpoints/unet_best.pth",
    "sr": 16000,
    "segment_length": 16384,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_output": True,
    "output_dir": "results"
}

os.makedirs(str(CONFIG["output_dir"]), exist_ok=True)

# ---------------------- METRICS ---------------------- #
def compute_snr(clean, denoised):
    noise = clean - denoised
    return 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8))

def compute_segmental_snr(clean, denoised, frame_length=512):
    num_frames = len(clean) // frame_length
    segsnr = 0.0
    for i in range(num_frames):
        c = clean[i*frame_length:(i+1)*frame_length]
        d = denoised[i*frame_length:(i+1)*frame_length]
        segsnr += compute_snr(c, d)
    return segsnr / max(num_frames, 1)

# ---------------------- UTILITY ---------------------- #
def slice_audio(audio, segment_length):
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    return segments

def run_denoising(model, noisy):
    segments = slice_audio(noisy, CONFIG["segment_length"])
    outputs = []

    for segment in segments:
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(CONFIG["device"])
        with torch.no_grad():
            output = model(input_tensor).squeeze().cpu().numpy()
        outputs.append(output)

    return np.concatenate(outputs)

# ---------------------- EVALUATE ---------------------- #
def evaluate_all():
    model = WaveUNet(in_ch=1, out_ch=1).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.eval()

    noisy_files = sorted(glob.glob(os.path.join(CONFIG["noisy_dir"], "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(CONFIG["clean_dir"], "*.wav")))

    assert len(noisy_files) == len(clean_files), "Mismatch between clean and noisy test files."

    snr_list = []
    segsnr_list = []

    for idx, (noisy_path, clean_path) in enumerate(zip(noisy_files, clean_files)):
        # Load and preprocess
        noisy, _ = read_wav(noisy_path, sr=CONFIG["sr"])
        clean, _ = read_wav(clean_path, sr=CONFIG["sr"])

        noisy = ssbse_enhance(noisy)
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        denoised = run_denoising(model, noisy)
        denoised = denoised[:min_len]
        clean = clean[:min_len]

        # Metrics
        snr = compute_snr(clean, denoised)
        segsnr = compute_segmental_snr(clean, denoised)

        snr_list.append(snr)
        segsnr_list.append(segsnr)

        print(f"[{idx+1}] {os.path.basename(noisy_path)} | SNR: {snr:.2f} dB | SegSNR: {segsnr:.2f} dB")

        if CONFIG["save_output"]:
            out_path = os.path.join(CONFIG["output_dir"], f"denoised_{idx+1}.wav")
            write_wav(out_path, denoised, CONFIG["sr"])

    print("\n📊 Final Report")
    print(f"Average SNR: {np.mean(snr_list):.2f} dB")
    print(f"Average SegSNR: {np.mean(segsnr_list):.2f} dB")

# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":
    evaluate_all()
