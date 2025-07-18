import numpy as np
from scipy.io import wavfile

def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    elif SNR < -5.0:
        a = 4
    else:  # SNR > 20
        a = 1
    return a

def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    elif SNR < -5.0:
        a = 5
    else:  # SNR > 20
        a = 1
    return a

def audio_basic_spectral_subtraction(input_file_name, output_file_name):
    """
    Implements the basic power spectral subtraction algorithm.
    
    Args:
        input_file_name (str): Path to noisy speech file in .wav format
        output_file_name (str): Path to enhanced output file in .wav format
    
    Reference:
        Berouti, M., Schwartz, M., and Makhoul, J. (1979). Enhancement of speech 
        corrupted by acoustic noise. Proc. IEEE Int. Conf. Acoust., Speech, 
        Signal Processing, 208-211.
    """
    # Read input WAV file
    Srate, x = wavfile.read(input_file_name)
    
    # Ensure input is mono and handle 16-bit data
    if len(x.shape) > 1:
        x = x[:, 0]  # Take first channel if stereo
    x = x / 32768.0  # Normalize to [-1, 1] for 16-bit PCM

    # Initialize variables
    len_frame = int(20 * Srate / 1000)  # Frame size in samples (20ms)
    if len_frame % 2 == 1:
        len_frame += 1
    PERC = 50  # Window overlap in percent
    len1 = int(len_frame * PERC / 100)
    len2 = len_frame - len1

    Thres = 3  # VAD threshold in dB SNRseg
    alpha = 2.0  # Power exponent
    FLOOR = 0.002
    G = 0.9

    # Create Hanning window
    win = np.hanning(len_frame)
    winGain = len2 / np.sum(win)  # Normalization gain for overlap-add

    # Noise magnitude calculations - first 5 frames assumed as noise/silence
    nFFT = 2 * 2 ** int(np.ceil(np.log2(len_frame)))
    noise_mean = np.zeros(nFFT)
    j = 0
    for k in range(5):
        noise_mean += np.abs(np.fft.fft(win * x[j:j+len_frame], nFFT))
        j += len_frame
    noise_mu = noise_mean / 5

    # Allocate memory and initialize variables
    k = 0
    x_old = np.zeros(len1)
    Nframes = int(np.floor(len(x) / len2) - 1)
    xfinal = np.zeros(Nframes * len2)

    # Start processing
    for n in range(Nframes):
        # Windowing
        insign = win * x[k:k+len_frame]
        # Compute Fourier transform
        spec = np.fft.fft(insign, nFFT)
        sig = np.abs(spec)  # Magnitude spectrum
        
        # Save phase information
        theta = np.angle(spec)
        
        # Compute SNR for VAD
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2)**2 / np.linalg.norm(noise_mu, 2)**2)
        
        # Select beta based on alpha
        if alpha == 1.0:
            beta = berouti1(SNRseg)
        else:
            beta = berouti(SNRseg)
        
        # Spectral subtraction
        sub_speech = sig**alpha - beta * noise_mu**alpha
        diffw = sub_speech - FLOOR * noise_mu**alpha
        
        # Floor negative components
        z = np.where(diffw < 0)[0]
        if len(z) > 0:
            sub_speech[z] = FLOOR * noise_mu[z]**alpha
        
        # Ensure conjugate symmetry for real reconstruction Phyllo in reconstruction
        sub_speech[nFFT//2+1:] = np.flipud(sub_speech[1:nFFT//2+1])
        
        # Apply phase and take IFFT
        x_phase = (sub_speech**(1/alpha)) * (np.cos(theta) + 1j * np.sin(theta))
        xi = np.real(np.fft.ifft(x_phase))[:len_frame]
        
        # Overlap and add
        xfinal[k:k+len2] = x_old + xi[:len1]
        x_old = xi[len1:len_frame]
        
        k += len2

    # Write output WAV file
    xfinal = (winGain * xfinal * 32768).astype(np.int16)  # Convert back to 16-bit PCM
    wavfile.write(output_file_name, Srate, xfinal)