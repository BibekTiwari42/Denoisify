# denoiser/model_loader.py

import torch
import sys
import os

# Add model_training to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_training'))

from model import WaveUNet
from postprocess import waveunet_with_ssbse_postprocess, ssbse_only_postprocess

def denoise_audio(input_path, output_path, use_ssbse_only=False):
    """
    Denoise audio using WaveUNet + SSBSE or SSBSE only
    """
    # Use CPU device for web interface to avoid GPU memory issues
    device = 'cpu'
    
    if use_ssbse_only:
        # Use only SSBSE post-processing
        return ssbse_only_postprocess(input_path, output_path)
    else:
        # Use WaveUNet + SSBSE pipeline
        model_path = os.path.join("model_training", "checkpoints", "unet_best.pth")
        
        # Load model
        model = WaveUNet(in_ch=1, out_ch=1, depth=5, base_ch=24)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if checkpoint contains model_state_dict key (full checkpoint)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct model weights
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        return waveunet_with_ssbse_postprocess(model, input_path, output_path, device)
