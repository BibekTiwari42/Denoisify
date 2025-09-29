import numpy as np
import scipy.io.wavfile as wav
import torch
import os

def ssbse_enhance(signal, frame_length=512, hop_length=256, noise_frames=6, noise_threshold=0.8):
    """
    Memory-optimized SSBSE enhancement
    """
    # Reduce parameters for memory efficiency
    if len(signal) < frame_length * noise_frames:
        return signal
    
    # Limit signal length to prevent memory issues
    max_length = 160000  # ~10 seconds at 16kHz
    if len(signal) > max_length:
        print(f"Warning: Audio too long ({len(signal)} samples), processing first {max_length} samples")
        signal = signal[:max_length]
    
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.stack([
        signal[i * hop_length:i * hop_length + frame_length]
        for i in range(num_frames)
    ]).T

    noise_matrix = frames[:, :noise_frames]
    
    # Add regularization to avoid singular matrix
    Rn = np.cov(noise_matrix) + 1e-6 * np.eye(frame_length)

    eigvals, eigvecs = np.linalg.eigh(Rn)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Ensure we don't remove all signal components
    cumulative_energy = np.cumsum(eigvals) / np.sum(eigvals)
    rank = min(np.searchsorted(cumulative_energy, noise_threshold) + 1, frame_length - 1)

    E_noise = eigvecs[:, :rank]
    P_noise = E_noise @ E_noise.T
    P_signal = np.eye(frame_length) - P_noise

    enhanced_frames = P_signal @ frames
    
    # Proper overlap-add reconstruction
    enhanced_signal = np.zeros(len(signal))
    window_count = np.zeros(len(signal))
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(signal):
            enhanced_signal[start:end] += enhanced_frames[:, i]
            window_count[start:end] += 1
    
    # Normalize by overlap count to avoid amplification
    enhanced_signal = np.divide(enhanced_signal, window_count, 
                               out=np.zeros_like(enhanced_signal), 
                               where=window_count!=0)
    
    # Normalize amplitude
    if np.max(np.abs(enhanced_signal)) > 0:
        enhanced_signal = enhanced_signal / np.max(np.abs(enhanced_signal))
    
    return enhanced_signal

def read_wav(filename):
    """Read WAV file and convert to float32 with memory limits"""
    sr, data = wav.read(filename)
    
    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648
    
    # Limit audio length to prevent memory issues
    max_samples = 320000  # ~20 seconds at 16kHz
    if len(data) > max_samples:
        print(f"Warning: Audio too long, truncating to {max_samples} samples")
        data = data[:max_samples]
    
    return sr, data

def write_wav(filename, sr, data):
    """Write WAV file, converting from float32 to int16"""
    # Ensure data is in range [-1, 1]
    data = np.clip(data, -1.0, 1.0)
    # Convert to int16
    data_int16 = np.int16(data * 32767)
    wav.write(filename, sr, data_int16)

def process_single_chunk(model, audio, device):
    """Process a single audio chunk with WaveUNet - memory optimized"""
    # Limit chunk size to prevent memory issues
    max_chunk_size = 16384  # Reduced from potentially large sizes
    
    if len(audio) > max_chunk_size:
        audio = audio[:max_chunk_size]
        print(f"Warning: Chunk too large, truncating to {max_chunk_size} samples")
    
    try:
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(audio_tensor)
        
        result = output.squeeze().cpu().numpy()
        
        # Clear GPU memory if using CUDA
        if device != 'cpu':
            torch.cuda.empty_cache()
        
        return result
        
    except RuntimeError as e:
        if "out of memory" in str(e) or "not enough memory" in str(e):
            print(f"Memory error in process_single_chunk: {e}")
            print("Falling back to smaller chunk size...")
            
            # Try with even smaller chunk
            smaller_chunk = audio[:8192] if len(audio) > 8192 else audio
            audio_tensor = torch.FloatTensor(smaller_chunk).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(audio_tensor)
            
            result = output.squeeze().cpu().numpy()
            
            # Pad result to match original audio length if needed
            if len(result) < len(audio):
                result = np.pad(result, (0, len(audio) - len(result)), mode='constant')
            
            if device != 'cpu':
                torch.cuda.empty_cache()
            
            return result
        else:
            raise e

def process_long_audio(model, audio, chunk_size, overlap, device):
    """Process long audio files in overlapping chunks - memory optimized"""
    # Reduce chunk size and overlap to save memory
    safe_chunk_size = min(chunk_size, 8192)  # Maximum 8K samples per chunk
    safe_overlap = min(overlap, 1024)  # Maximum 1K overlap
    
    enhanced_audio = np.zeros_like(audio)
    window_count = np.zeros_like(audio)
    
    step = safe_chunk_size - safe_overlap
    
    for start in range(0, len(audio), step):
        end = min(start + safe_chunk_size, len(audio))
        chunk = audio[start:end]
        
        # Pad if necessary
        if len(chunk) < safe_chunk_size:
            chunk = np.pad(chunk, (0, safe_chunk_size - len(chunk)), mode='constant')
        
        try:
            # Process chunk
            enhanced_chunk = process_single_chunk(model, chunk, device)
            
            # Add to output with overlap handling
            actual_end = min(start + len(enhanced_chunk), len(audio))
            enhanced_audio[start:actual_end] += enhanced_chunk[:actual_end-start]
            window_count[start:actual_end] += 1
            
        except Exception as e:
            print(f"Error processing chunk {start}-{end}: {e}")
            # Skip this chunk and continue
            continue
    
    # Normalize by overlap count
    enhanced_audio = np.divide(enhanced_audio, window_count, 
                              out=np.zeros_like(enhanced_audio), 
                              where=window_count!=0)
    
    return enhanced_audio

def waveunet_with_ssbse_postprocess(
    model, 
    input_path, 
    output_path, 
    device='cpu',
    chunk_size=8192,  # Reduced default chunk size
    overlap=1024      # Reduced default overlap
):
    """
    Memory-optimized denoising pipeline: WaveUNet + SSBSE post-processing
    """
    try:
        # Load audio with memory limits
        sr, audio = read_wav(input_path)
        
        print(f"Processing audio: {len(audio)} samples, {sr} Hz")
        
        # Process in chunks for long audio files
        if len(audio) > chunk_size:
            print("Processing long audio in chunks...")
            enhanced_audio = process_long_audio(model, audio, chunk_size, overlap, device)
        else:
            print("Processing single chunk...")
            enhanced_audio = process_single_chunk(model, audio, device)
        
        # Apply SSBSE post-processing with reduced parameters
        print("Applying SSBSE post-processing...")
        final_result = ssbse_enhance(enhanced_audio, 
                                   frame_length=512, 
                                   hop_length=256,
                                   noise_frames=6,
                                   noise_threshold=0.8)
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_wav(output_path, sr, final_result)
        
        print(f"✅ WaveUNet + SSBSE processed audio saved at: {output_path}")
        return final_result
        
    except Exception as e:
        print(f"Error in waveunet_with_ssbse_postprocess: {e}")
        raise e

def ssbse_only_postprocess(input_path, output_path):
    """
    Apply only SSBSE post-processing to existing audio - memory optimized
    """
    try:
        sr, audio = read_wav(input_path)
        enhanced_audio = ssbse_enhance(audio, 
                                     frame_length=512, 
                                     hop_length=256,
                                     noise_frames=6,
                                     noise_threshold=0.8)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_wav(output_path, sr, enhanced_audio)
        
        print(f"✅ SSBSE processed audio saved at: {output_path}")
        return enhanced_audio
        
    except Exception as e:
        print(f"Error in ssbse_only_postprocess: {e}")
        raise e