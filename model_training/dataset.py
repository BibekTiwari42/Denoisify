import torch
from torch.utils.data import Dataset
import numpy as np
import wave

# Utility to read wav files using standard library and numpy

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
        if sampwidth == 2:
            samples = samples / 32768.0
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

class AudioWaveformDataset(Dataset):
    def __init__(self, noisy_files, clean_files, sr=16000, segment_length=16384):
        assert len(noisy_files) == len(clean_files), "Mismatch in number of noisy and clean files."
        self.noisy_files = noisy_files
        self.clean_files = clean_files
        self.sr = sr
        self.segment_length = segment_length

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy, _ = read_wav(self.noisy_files[idx], sr=self.sr)
        clean, _ = read_wav(self.clean_files[idx], sr=self.sr)

        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        if min_len < self.segment_length:
            pad_len = self.segment_length - min_len
            noisy = np.pad(noisy, (0, pad_len), mode='constant')
            clean = np.pad(clean, (0, pad_len), mode='constant')
        else:
            start = np.random.randint(0, min_len - self.segment_length + 1)
            noisy = noisy[start:start + self.segment_length]
            clean = clean[start:start + self.segment_length]

        noisy_tensor = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)

        return noisy_tensor, clean_tensor
