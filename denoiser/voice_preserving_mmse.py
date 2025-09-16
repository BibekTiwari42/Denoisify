# Voice-preserving MMSE-STSA implementations
# Optimized to maintain natural voice characteristics
import numpy as np

def mmse_stsa_voice_preserving(x, fs, frame_len=0.032, frame_shift=0.008, alpha=0.98, beta=0.98, Gmin=0.3):
    """
    Voice-preserving MMSE-STSA that maintains natural voice characteristics.
    - Higher Gmin to preserve fundamental frequencies
    - Gentle processing to avoid pitch artifacts
    """
    # Store original characteristics for preservation
    input_rms = np.sqrt(np.mean(x**2))
    
    # Parameters optimized for voice preservation
    nFFT = int(2 ** np.ceil(np.log2(frame_len * fs)))
    nwin = int(frame_len * fs)
    nshift = int(frame_shift * fs)
    win = np.hanning(nwin)
    x = np.append(np.zeros(nwin//2), x)
    x = np.append(x, np.zeros(nwin//2))
    nframes = int((len(x) - nwin) / nshift) + 1
    y = np.zeros(len(x))
    noise_ps = None
    prev_post_snr = None
    prev_prio_snr = None
    
    for i in range(nframes):
        start = i * nshift
        frame = x[start:start+nwin] * win
        X = np.fft.rfft(frame, nFFT)
        X_mag = np.abs(X)
        X_phase = np.angle(X)
        
        # Noise estimation (more frames for better estimation)
        if i < 8:  # Use more frames for stable noise estimation
            if noise_ps is None:
                noise_ps = X_mag ** 2
            else:
                noise_ps = (noise_ps * i + X_mag ** 2) / (i + 1)
            y[start:start+nwin] += frame * win
            continue
            
        # Posteriori SNR
        post_snr = X_mag ** 2 / (noise_ps + 1e-12)
        
        # A priori SNR with more conservative smoothing
        if prev_prio_snr is None:
            prio_snr = np.maximum(post_snr - 1, 0)
        else:
            prio_snr = alpha * prev_prio_snr + (1 - alpha) * np.maximum(post_snr - 1, 0)
        prev_prio_snr = prio_snr
        
        # Voice-preserving gain function
        v = prio_snr * post_snr / (1 + prio_snr)
        G = (prio_snr / (1 + prio_snr)) * np.exp(-0.5 * v)
        
        # Higher minimum gain to preserve voice characteristics
        G = np.maximum(G, Gmin)
        
        # Frequency-dependent processing to preserve speech fundamentals
        freqs = np.fft.rfftfreq(nFFT, 1/fs)
        
        # Preserve fundamental frequency range (80-300 Hz) more aggressively
        fundamental_mask = (freqs >= 80) & (freqs <= 300)
        G[fundamental_mask] = np.maximum(G[fundamental_mask], 0.6)
        
        # Preserve formant regions (300-3000 Hz)
        formant_mask = (freqs >= 300) & (freqs <= 3000)
        G[formant_mask] = np.maximum(G[formant_mask], 0.4)
        
        # Apply gain
        Y = G * X_mag * np.exp(1j * X_phase)
        y_frame = np.fft.irfft(Y, nFFT)[:nwin]
        y[start:start+nwin] += y_frame * win
        
        # More conservative noise update to avoid tracking speech as noise
        noise_ps = beta * noise_ps + (1 - beta) * (G * X_mag) ** 2
    
    # Remove padding
    y = y[nwin//2:-(nwin//2)]
    
    # Preserve RMS level with gentle limiting
    if input_rms > 0:
        output_rms = np.sqrt(np.mean(y**2))
        if output_rms > 0:
            scale_factor = input_rms / output_rms
            y = y * scale_factor
            
            # Gentle limiting to preserve dynamics
            if np.max(np.abs(y)) > 1.0:
                y = y / (1.1 * np.max(np.abs(y)))
    
    return y

def mmse_stsa_minimal(x, fs, frame_len=0.032, frame_shift=0.008, alpha=0.95, beta=0.95, Gmin=0.5):
    """
    Minimal MMSE-STSA processing for maximum voice preservation.
    - Very gentle processing
    - High minimum gain
    - Conservative noise reduction
    """
    input_rms = np.sqrt(np.mean(x**2))
    
    nFFT = int(2 ** np.ceil(np.log2(frame_len * fs)))
    nwin = int(frame_len * fs)
    nshift = int(frame_shift * fs)
    win = np.hanning(nwin)
    x = np.append(np.zeros(nwin//2), x)
    x = np.append(x, np.zeros(nwin//2))
    nframes = int((len(x) - nwin) / nshift) + 1
    y = np.zeros(len(x))
    noise_ps = None
    prev_prio_snr = None
    
    for i in range(nframes):
        start = i * nshift
        frame = x[start:start+nwin] * win
        X = np.fft.rfft(frame, nFFT)
        X_mag = np.abs(X)
        X_phase = np.angle(X)
        
        # Minimal noise estimation (only first 3 frames)
        if i < 3:
            if noise_ps is None:
                noise_ps = X_mag ** 2
            else:
                noise_ps = (noise_ps * i + X_mag ** 2) / (i + 1)
            y[start:start+nwin] += frame * win
            continue
            
        # Conservative SNR estimation
        post_snr = X_mag ** 2 / (noise_ps + 1e-12)
        
        if prev_prio_snr is None:
            prio_snr = np.maximum(post_snr - 1, 0)
        else:
            prio_snr = alpha * prev_prio_snr + (1 - alpha) * np.maximum(post_snr - 1, 0)
        prev_prio_snr = prio_snr
        
        # Very conservative gain function
        v = prio_snr * post_snr / (1 + prio_snr)
        G = (prio_snr / (1 + prio_snr)) * np.exp(-0.5 * v)
        
        # High minimum gain for voice preservation
        G = np.maximum(G, Gmin)
        
        # Apply gain
        Y = G * X_mag * np.exp(1j * X_phase)
        y_frame = np.fft.irfft(Y, nFFT)[:nwin]
        y[start:start+nwin] += y_frame * win
        
        # Very gentle noise update
        noise_ps = beta * noise_ps + (1 - beta) * (G * X_mag) ** 2
    
    # Remove padding
    y = y[nwin//2:-(nwin//2)]
    
    # Preserve original level exactly
    if input_rms > 0:
        output_rms = np.sqrt(np.mean(y**2))
        if output_rms > 0:
            y = y * (input_rms / output_rms)
    
    return y

def no_postprocessing(x, fs, **kwargs):
    """
    Skip post-processing entirely - return WaveUNet output as-is.
    This preserves voice characteristics best but may leave some noise.
    """
    return x
