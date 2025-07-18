import numpy as np
import torch
import os
import wave

# WAV I/O utils

def read_wav(filename):
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
        # Normalize to [-1, 1] for 16-bit PCM
        if sampwidth == 2:
            samples = samples / 32768.0
        return framerate, samples

def write_wav(filename, sr, data):
    data = np.clip(data, -1.0, 1.0)
    data_int16 = np.int16(data * 32767)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data_int16.tobytes())

# Custom STFT/ISTFT using numpy

def stft(x, nperseg=512, noverlap=256):
    x = np.pad(x, (0, nperseg - len(x) % nperseg), mode='constant')
    step = nperseg - noverlap
    frames = np.lib.stride_tricks.sliding_window_view(x, nperseg)[::step]
    window = np.hanning(nperseg)
    return np.fft.rfft(frames * window, axis=1)

def istft(X, nperseg=512, noverlap=256):
    step = nperseg - noverlap
    window = np.hanning(nperseg)
    time_len = (X.shape[0] - 1) * step + nperseg
    x = np.zeros(time_len)
    for i, frame in enumerate(X):
        x[i*step:i*step+nperseg] += np.fft.irfft(frame) * window
    return x

def ssbse_enhance_frequency_domain(signal, nperseg=512, noverlap=256, noise_frames=6, noise_threshold=0.8):
    """
    SSBSE enhancement in frequency domain using STFT
    """
    if len(signal) < nperseg * noise_frames:
        return signal
    
    # Limit signal length to prevent memory issues
    max_length = 160000  # ~10 seconds at 16kHz
    if len(signal) > max_length:
        print(f"Warning: Audio too long ({len(signal)} samples), processing first {max_length} samples")
        signal = signal[:max_length]
    
    # Convert to frequency domain using STFT
    frequencies, times, Zxx = stft(signal, nperseg=nperseg, noverlap=noverlap)
    
    # Get magnitude and phase
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Apply SSBSE algorithm in frequency domain
    enhanced_magnitude = np.zeros_like(magnitude)
    
    # Process each frequency bin separately
    for freq_idx in range(magnitude.shape[0]):
        freq_frames = magnitude[freq_idx, :]
        
        if len(freq_frames) < noise_frames:
            enhanced_magnitude[freq_idx, :] = freq_frames
            continue
            
        # Estimate noise from first few frames
        noise_spectrum = freq_frames[:noise_frames]
        
        # Create noise covariance matrix for this frequency
        if len(noise_spectrum) > 1:
            noise_frames_matrix = np.column_stack([
                freq_frames[i:i+noise_frames] 
                for i in range(len(freq_frames) - noise_frames + 1)
            ])
            
            if noise_frames_matrix.shape[1] > 0:
                # Covariance matrix with regularization
                Rn = np.cov(noise_frames_matrix) + 1e-6 * np.eye(noise_frames_matrix.shape[0])
                
                # Eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(Rn)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                  # Determine noise subspace rank (corrected logic)
                cumulative_energy = np.cumsum(eigvals) / np.sum(eigvals)
                # Use smaller eigenvalues for noise subspace (not larger ones)
                noise_rank = min(np.searchsorted(cumulative_energy, noise_threshold) + 1, len(eigvals) - 2)
                
                # Signal subspace projection (corrected)
                # Keep the signal subspace (larger eigenvalues), remove noise subspace (smaller eigenvalues)
                E_signal = eigvecs[:, :len(eigvals) - noise_rank]  # Signal subspace
                P_signal = E_signal @ E_signal.T  # Project onto signal subspace
                
                # Apply projection to overlapping frames
                enhanced_magnitude[freq_idx, :] = np.zeros_like(freq_frames)
                overlap_count = np.zeros_like(freq_frames)
                
                for i in range(len(freq_frames) - noise_frames + 1):
                    frame_segment = freq_frames[i:i+noise_frames]
                    enhanced_segment = P_signal @ frame_segment
                    
                    # Overlap-add reconstruction
                    for j, val in enumerate(enhanced_segment):
                        if i + j < len(freq_frames):
                            enhanced_magnitude[freq_idx, i + j] += val
                            overlap_count[i + j] += 1
                
                # Normalize by overlap count
                enhanced_magnitude[freq_idx, :] = np.divide(
                    enhanced_magnitude[freq_idx, :], overlap_count,
                    out=freq_frames.copy(), where=overlap_count > 0
                )
            else:
                enhanced_magnitude[freq_idx, :] = freq_frames
        else:
            enhanced_magnitude[freq_idx, :] = freq_frames
    
    # Reconstruct complex spectrum with enhanced magnitude and original phase
    enhanced_Zxx = enhanced_magnitude * np.exp(1j * phase)
    
    # Convert back to time domain using inverse STFT
    _, enhanced_signal = istft(enhanced_Zxx, nperseg=nperseg, noverlap=noverlap)
    
    # Ensure output length matches input
    if len(enhanced_signal) > len(signal):
        enhanced_signal = enhanced_signal[:len(signal)]
    elif len(enhanced_signal) < len(signal):
        enhanced_signal = np.pad(enhanced_signal, (0, len(signal) - len(enhanced_signal)), mode='constant')
    
    # Normalize amplitude
    if np.max(np.abs(enhanced_signal)) > 0:
        enhanced_signal = enhanced_signal / np.max(np.abs(enhanced_signal))
    
    return enhanced_signal

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
          # Apply SSBSE post-processing in frequency domain
        print("Applying SSBSE post-processing in frequency domain...")
        final_result = ssbse_enhance_frequency_domain(enhanced_audio, 
                                                     nperseg=512, 
                                                     noverlap=256,
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
        enhanced_audio = ssbse_enhance_frequency_domain(audio, 
                                                       nperseg=512, 
                                                       noverlap=256,
                                                       noise_frames=6,
                                                       noise_threshold=0.8)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_wav(output_path, sr, enhanced_audio)
        
        print(f"✅ SSBSE processed audio saved at: {output_path}")
        return enhanced_audio
        
    except Exception as e:
        print(f"Error in ssbse_only_postprocess: {e}")
        raise e