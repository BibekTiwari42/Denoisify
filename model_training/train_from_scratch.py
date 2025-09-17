import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import warnings
import numpy as np
from pesq import pesq
from pystoi import stoi
import soundfile as sf

# Suppress PyTorch save/load warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.save.*")

from model import WaveUNet
from dataset import AudioWaveformDataset

# ---------------- CONFIG ----------------
CONFIG = {
    "train_noisy_dir": "Data_new/train/noisy",
    "train_clean_dir": "Data_new/train/clean", 
    "valid_noisy_dir": "Data_new/test/noisy",
    "valid_clean_dir": "Data_new/test/clean",
    "sample_rate": 16000,
    "segment_length": 16384,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "checkpoint_path": "model_training/checkpoints/unet_version4.pth",  # Version4
    "log_path": "training_log_version4.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Ensure checkpoint folder exists (if path contains a directory)
checkpoint_dir = os.path.dirname(str(CONFIG["checkpoint_path"]))
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

# ---------------- Helpers ----------------
def get_file_pairs(noisy_dir, clean_dir):
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    print(f"Found {len(noisy_files)} noisy and {len(clean_files)} clean files.")
    assert len(noisy_files) == len(clean_files), "Mismatch between noisy and clean file counts."
    return noisy_files, clean_files

def compute_snr(clean, denoised):
    noise = clean - denoised
    snr = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8))
    return snr

# ---------------- Training ----------------
def train():
    print("\nStarting training with WaveUNet version4 (resume-enabled)\n")

    train_noisy, train_clean = get_file_pairs(CONFIG["train_noisy_dir"], CONFIG["train_clean_dir"])
    valid_noisy, valid_clean = get_file_pairs(CONFIG["valid_noisy_dir"], CONFIG["valid_clean_dir"])

    train_dataset = AudioWaveformDataset(train_noisy, train_clean, CONFIG["sample_rate"], CONFIG["segment_length"])
    valid_dataset = AudioWaveformDataset(valid_noisy, valid_clean, CONFIG["sample_rate"], CONFIG["segment_length"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=(CONFIG["device"]=="cuda"))
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, pin_memory=(CONFIG["device"]=="cuda"))

    device = CONFIG["device"]
    model = WaveUNet(in_ch=1, out_ch=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    # Resume support: load checkpoint if exists
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(CONFIG["checkpoint_path"]):
        try:
            checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
                start_epoch = int(checkpoint.get("epoch", start_epoch)) + 1
                print(f"Resuming from checkpoint: starting at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
            else:
                # older-style checkpoint that is just state_dict
                model.load_state_dict(checkpoint)
                print("Loaded model weights from checkpoint (no optimizer/epoch info). Starting from epoch 0.")
        except Exception as e:
            print(f"Warning: failed to load checkpoint '{CONFIG['checkpoint_path']}' ({e}). Starting from scratch.")

    # CSV logging: write header only if not exists or we're starting from epoch 0
    log_path = CONFIG["log_path"]
    log_header = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'pesq', 'stoi', 'snr']
    if (not os.path.exists(log_path)) or start_epoch == 0:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
    else:
        print(f"Appending to existing log: {log_path}")

    # Training loop
    for epoch in range(start_epoch, CONFIG["epochs"]):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        train_steps = 0

        for noisy, clean in train_loader:
            train_steps += 1
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            with torch.no_grad():
                target_var = torch.var(clean)
                norm_mse = loss.detach() / (target_var + 1e-8)
                batch_acc = torch.clamp(1.0 - norm_mse, 0.0, 1.0) * 100
                total_train_acc += float(batch_acc)

        avg_train_loss = total_train_loss / max(1, train_steps)
        avg_train_acc = total_train_acc / max(1, train_steps)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        pesq_scores, stoi_scores, snr_scores = [], [], []
        val_steps = 0

        with torch.no_grad():
            for noisy, clean in valid_loader:
                val_steps += 1
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                total_val_loss += loss.item()

                target_var = torch.var(clean)
                norm_mse = loss / (target_var + 1e-8)
                batch_acc = torch.clamp(1.0 - norm_mse, 0.0, 1.0) * 100
                total_val_acc += float(batch_acc)

                # Convert to numpy arrays for metrics
                clean_np = clean.squeeze().cpu().numpy()
                denoised_np = output.squeeze().cpu().numpy()

                # PESQ (optional) and STOI
                if pesq is not None:
                    try:
                        pesq_scores.append(pesq(CONFIG["sample_rate"], clean_np, denoised_np, 'wb'))
                    except Exception:
                        pesq_scores.append(np.nan)
                else:
                    pesq_scores.append(np.nan)

                try:
                    stoi_scores.append(stoi(clean_np, denoised_np, CONFIG["sample_rate"], extended=False))
                except Exception:
                    stoi_scores.append(np.nan)

                try:
                    snr_scores.append(compute_snr(clean_np, denoised_np))
                except Exception:
                    snr_scores.append(np.nan)

        avg_val_loss = total_val_loss / max(1, val_steps)
        avg_val_acc = total_val_acc / max(1, val_steps)
        avg_pesq = float(np.nanmean(pesq_scores)) if len(pesq_scores) > 0 else np.nan
        avg_stoi = float(np.nanmean(stoi_scores)) if len(stoi_scores) > 0 else np.nan
        avg_snr = float(np.nanmean(snr_scores)) if len(snr_scores) > 0 else np.nan

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | "
              f"Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | PESQ: {avg_pesq:.3f} | STOI: {avg_stoi:.3f} | SNR: {avg_snr:.2f} | Time: {time.time() - start_time:.2f}s")

        # Append CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_pesq, avg_stoi, avg_snr])

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save a full checkpoint dict for resuming
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CONFIG["checkpoint_path"])
            print("Best model saved.\n")

    print("Training version4 complete.")

# ---------------- Main ----------------
if __name__ == "__main__":
    train()