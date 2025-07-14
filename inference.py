import torch
import sys
import os

# Add model_training to path
sys.path.append('model_training')

from model import WaveUNet
from postprocess import waveunet_with_ssbse_postprocess, ssbse_only_postprocess

def load_model(model_path, device='cpu'):
    """Load trained WaveUNet model"""
    model = WaveUNet(in_ch=1, out_ch=1, depth=5, base_ch=24)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def denoise_audio(input_path, output_path, model_path=None, use_ssbse_only=False, device='cpu'):
    """
    Denoise audio using WaveUNet + SSBSE or SSBSE only
    """
    if use_ssbse_only:
        # Use only SSBSE post-processing
        return ssbse_only_postprocess(input_path, output_path)
    else:
        # Use WaveUNet + SSBSE pipeline
        if model_path is None:
            raise ValueError("Model path required for WaveUNet + SSBSE pipeline")
        
        model = load_model(model_path, device)
        return waveunet_with_ssbse_postprocess(model, input_path, output_path, device)

# Example usage
if __name__ == "__main__":
    # Set paths
    input_file = "media/input/noisy_sample.wav"
    output_file = "media/output/denoised_waveunet_ssbse.wav"
    model_path = "model_training/checkpoints/best_model.pth"  # Update with your model path
    
    # Choose processing method
    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'
    
    print(f"Using device: {device}")
    
    # Option 1: WaveUNet + SSBSE (recommended)
    denoise_audio(input_file, output_file, model_path, use_ssbse_only=False, device=device)
    
    # Option 2: SSBSE only (for comparison)
    # denoise_audio(input_file, "media/output/denoised_ssbse_only.wav", use_ssbse_only=True)