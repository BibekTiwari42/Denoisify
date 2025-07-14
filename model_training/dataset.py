import torch
from torch.utils.data import Dataset
import numpy as np
import librosa

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
        noisy, _ = librosa.load(self.noisy_files[idx], sr=self.sr)
        clean, _ = librosa.load(self.clean_files[idx], sr=self.sr)

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
