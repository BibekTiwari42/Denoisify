import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import WaveUNet
from dataset import AudioWaveformDataset

CONFIG = {
    "train_noisy_dir": "Data/train/noisy_trainset_28spk/noisy_trainset_28spk_wav",
    "train_clean_dir": "Data/train/clean_trainset_28spk/clean_trainset_28spk_wav",
    "valid_noisy_dir": "Data/test/noisy_testset",
    "valid_clean_dir": "Data/test/clean_testset",
    "sample_rate": 16000,
    "segment_length": 16384,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "checkpoint_path": "model_training/checkpoints/unet_best.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(os.path.dirname(str(CONFIG["checkpoint_path"])), exist_ok=True)

def get_file_pairs(noisy_dir, clean_dir):
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    print(f"Train noisy dir: {noisy_dir}")
    print(f"Train clean dir: {clean_dir}")
    print(f"Found {len(noisy_files)} noisy and {len(clean_files)} clean files.")
    assert len(noisy_files) > 0, f"No noisy files found in {noisy_dir}"
    assert len(noisy_files) == len(clean_files), "Mismatch between noisy and clean file counts."
    return noisy_files, clean_files

def train():
    print("\n🔁 Starting training with WaveUNet + SSBSE...\n")

    train_noisy, train_clean = get_file_pairs(CONFIG["train_noisy_dir"], CONFIG["train_clean_dir"])
    valid_noisy, valid_clean = get_file_pairs(CONFIG["valid_noisy_dir"], CONFIG["valid_clean_dir"])

    train_dataset = AudioWaveformDataset(train_noisy, train_clean, CONFIG["sample_rate"], CONFIG["segment_length"])
    valid_dataset = AudioWaveformDataset(valid_noisy, valid_clean, CONFIG["sample_rate"], CONFIG["segment_length"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=(CONFIG["device"] == "cuda"))
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, pin_memory=(CONFIG["device"] == "cuda"))

    model = WaveUNet(in_ch=1, out_ch=1).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    # 🔁 Resume checkpoint if available
    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(CONFIG["checkpoint_path"]):
        checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
        if "epoch" in checkpoint:  # Full checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            start_epoch = checkpoint['epoch'] + 1
            print(f"🔄 Resuming from epoch {start_epoch} with best val loss {best_val_loss:.5f}")
        else:  # Only state_dict saved (older format)
            model.load_state_dict(checkpoint)
            print("✅ Loaded model weights (no resume state).")

    # 🔂 Training loop
    for epoch in range(start_epoch, CONFIG["epochs"]):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0

        for noisy, clean in train_loader:
            noisy = noisy.to(CONFIG["device"])
            clean = clean.to(CONFIG["device"])
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in valid_loader:
                noisy = noisy.to(CONFIG["device"])
                clean = clean.to(CONFIG["device"])
                output = model(noisy)
                loss = criterion(output, clean)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Time: {time.time() - start_time:.2f}s")

        # Save if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CONFIG["checkpoint_path"])
            print("✅ Best model saved.\n")

    print("🎉 Training complete.")

if __name__ == "__main__":
    train()
