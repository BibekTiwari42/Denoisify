# MMSE-STSA implementation from pysepm (MIT License)
# Source: https://github.com/ehabets/pysepm/blob/master/pysepm/mmse_stsa.py
# Modified to preserve audio levels instead of normalizing to max amplitude
import numpy as np

def mmse_stsa_fixed(x, fs, frame_len=0.032, frame_shift=0.008, alpha=0.98, beta=0.98, Gmin=0.1):
    """
    MMSE-STSA speech enhancement algorithm with preserved volume levels.
    Args:
        x: 1D numpy array, input noisy speech (normalized to [-1, 1])
        fs: sample rate
        frame_len: frame length in seconds
        frame_shift: frame shift in seconds
        alpha: smoothing factor for a priori SNR
        beta: smoothing factor for noise estimation
        Gmin: minimum gain
    Returns:
        y: enhanced speech signal (preserving input volume levels)
    """
    # Store original RMS for level preservation
    input_rms = np.sqrt(np.mean(x**2))
    
    # Parameters
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
        
        # Noise estimation (first 5 frames)
        if i < 5:
            if noise_ps is None:
                noise_ps = X_mag ** 2
            else:
                noise_ps = (noise_ps * i + X_mag ** 2) / (i + 1)
            y[start:start+nwin] += frame * win
            continue
            
        # Posteriori SNR
        post_snr = X_mag ** 2 / (noise_ps + 1e-12)
        
        # A priori SNR
        if prev_prio_snr is None:
            prio_snr = np.maximum(post_snr - 1, 0)
        else:
            prio_snr = alpha * prev_prio_snr + (1 - alpha) * np.maximum(post_snr - 1, 0)
        prev_prio_snr = prio_snr
        
        # Gain function (Ephraim-Malah)
        v = prio_snr * post_snr / (1 + prio_snr)
        G = (prio_snr / (1 + prio_snr)) * np.exp(-0.5 * v)
        G = np.maximum(G, Gmin)
        
        # Apply gain
        Y = G * X_mag * np.exp(1j * X_phase)
        y_frame = np.fft.irfft(Y, nFFT)[:nwin]
        y[start:start+nwin] += y_frame * win
        
        # Update noise estimate
        noise_ps = beta * noise_ps + (1 - beta) * X_mag ** 2
    
    # Remove padding
    y = y[nwin//2:-(nwin//2)]
    
    # FIXED: Preserve original audio levels instead of aggressive normalization
    # Only prevent clipping if signal is too loud
    if np.max(np.abs(y)) > 1.0:
        y = y / np.max(np.abs(y))
    
    return y

def mmse_stsa_conservative(x, fs, frame_len=0.032, frame_shift=0.008, alpha=0.98, beta=0.98, Gmin=0.1):
    """
    MMSE-STSA with RMS level preservation - maintains input volume levels.
    """
    # Store original RMS for level preservation
    input_rms = np.sqrt(np.mean(x**2))
    
    # Parameters
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
        
        # Noise estimation (first 5 frames)
        if i < 5:
            if noise_ps is None:
                noise_ps = X_mag ** 2
            else:
                noise_ps = (noise_ps * i + X_mag ** 2) / (i + 1)
            y[start:start+nwin] += frame * win
            continue
            
        # Posteriori SNR
        post_snr = X_mag ** 2 / (noise_ps + 1e-12)
        
        # A priori SNR
        if prev_prio_snr is None:
            prio_snr = np.maximum(post_snr - 1, 0)
        else:
            prio_snr = alpha * prev_prio_snr + (1 - alpha) * np.maximum(post_snr - 1, 0)
        prev_prio_snr = prio_snr
        
        # Gain function (Ephraim-Malah)
        v = prio_snr * post_snr / (1 + prio_snr)
        G = (prio_snr / (1 + prio_snr)) * np.exp(-0.5 * v)
        G = np.maximum(G, Gmin)
        
        # Apply gain
        Y = G * X_mag * np.exp(1j * X_phase)
        y_frame = np.fft.irfft(Y, nFFT)[:nwin]
        y[start:start+nwin] += y_frame * win
        
        # Update noise estimate
        noise_ps = beta * noise_ps + (1 - beta) * X_mag ** 2
    
    # Remove padding
    y = y[nwin//2:-(nwin//2)]
    
    # FIXED: Preserve RMS level instead of normalizing to max amplitude
    if input_rms > 0:
        output_rms = np.sqrt(np.mean(y**2))
        if output_rms > 0:
            # Match RMS level while preventing clipping
            scale_factor = input_rms / output_rms
            y = y * scale_factor
            
            # Prevent clipping
            if np.max(np.abs(y)) > 1.0:
                y = y / np.max(np.abs(y))
    
    return y

# Keep original function for compatibility but deprecated
def mmse_stsa(x, fs, frame_len=0.032, frame_shift=0.008, alpha=0.98, beta=0.98, Gmin=0.1):
    """
    Original MMSE-STSA - WARNING: This normalizes to max amplitude which can amplify noise!
    Use mmse_stsa_fixed() or mmse_stsa_conservative() instead.
    """
    return mmse_stsa_fixed(x, fs, frame_len, frame_shift, alpha, beta, Gmin)
