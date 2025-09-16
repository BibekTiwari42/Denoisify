# denoiser/audio_utils.py

import os
import tempfile
import wave
import warnings

try:
    # Suppress pydub FFmpeg warnings since we primarily use WAV files
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
        warnings.filterwarnings("ignore", message="Couldn't find ffprobe or avprobe")
        from pydub import AudioSegment
        from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

def convert_to_wav(input_file_path, output_file_path=None):
    """
    Convert audio file (MP3, M4A, etc.) to WAV format
    
    Args:
        input_file_path (str): Path to the input audio file
        output_file_path (str, optional): Path for the output WAV file. 
                                        If None, creates a temporary file.
    
    Returns:
        str: Path to the converted WAV file
    """
    if not PYDUB_AVAILABLE:
        raise Exception(
            "Audio conversion requires pydub library with FFmpeg. "
            "Please convert your file to WAV format manually using online converters like "
            "https://cloudconvert.com/mp3-to-wav or install FFmpeg from https://ffmpeg.org/"
        )
    
    try:
        # Check if FFmpeg is available
        if not (which("ffmpeg") or which("avconv") or which("ffprobe")):
            raise Exception(
                "FFmpeg is required for MP3 conversion but not found on your system. "
                "Please install FFmpeg from https://ffmpeg.org/ or convert your file to WAV format manually."
            )
        
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_file_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set standard sample rate (16kHz for the model)
        audio = audio.set_frame_rate(16000)
        
        # Set to 16-bit
        audio = audio.set_sample_width(2)
        
        # Create output file path if not provided
        if output_file_path is None:
            # Create a temporary file
            temp_fd, output_file_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close the file descriptor
        
        # Export as WAV
        audio.export(output_file_path, format="wav")
        
        return output_file_path
        
    except Exception as e:
        raise Exception(f"Error converting audio file: {str(e)}")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set standard sample rate (16kHz for the model)
        audio = audio.set_frame_rate(16000)
        
        # Set to 16-bit
        audio = audio.set_sample_width(2)
        
        # Create output file path if not provided
        if output_file_path is None:
            # Create a temporary file
            temp_fd, output_file_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close the file descriptor
        
        # Export as WAV
        audio.export(output_file_path, format="wav")
        
        return output_file_path
        
    except Exception as e:
        # Provide more specific error messages
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            raise Exception(f"Audio conversion failed. FFmpeg is required for this audio format. Error: {str(e)}")
        else:
            raise Exception(f"Error converting audio file: {str(e)}")

def validate_audio_file(file_path):
    """
    Validate if the audio file can be processed (supports WAV, MP3, M4A, etc.)
    """
    try:
        # First try to handle WAV files directly (no pydub needed)
        if file_path.lower().endswith('.wav'):
            return validate_wav_file_direct(file_path)
        
        # For other formats, use pydub if available
        if not PYDUB_AVAILABLE:
            return False, "Only WAV files are supported (pydub not available for other formats)"
        
        # Try to load with pydub (supports many formats)
        audio = AudioSegment.from_file(file_path)
        
        # Check duration
        duration_seconds = len(audio) / 1000.0  # pydub duration is in milliseconds
        if duration_seconds < 0.5:  # Less than 0.5 seconds
            return False, "Audio file is too short (less than 0.5 seconds)"
        if duration_seconds > 300:  # More than 5 minutes
            return False, "Audio file is too long (more than 5 minutes)"
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 50:  # More than 50MB
            return False, "Audio file is too large (more than 50MB)"
        
        # Get audio info
        channels = audio.channels
        frame_rate = audio.frame_rate
        sample_width = audio.sample_width
        
        return True, f"Valid audio file: {channels} channels, {frame_rate}Hz, {duration_seconds:.2f}s"
        
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            return False, f"Audio format not supported without FFmpeg. Please use WAV format or install FFmpeg."
        else:
            return False, f"Invalid audio file: {str(e)}"

def validate_wav_file_direct(file_path):
    """
    Validate WAV file using built-in wave module (no external dependencies)
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Check duration
            duration_seconds = n_frames / framerate
            if duration_seconds < 0.5:  # Less than 0.5 seconds
                return False, "Audio file is too short (less than 0.5 seconds)"
            if duration_seconds > 300:  # More than 5 minutes
                return False, "Audio file is too long (more than 5 minutes)"
            
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 50:  # More than 50MB
                return False, "Audio file is too large (more than 50MB)"
            
            # Check if it's a valid WAV format
            if sampwidth not in [1, 2, 4]:  # 8-bit, 16-bit, or 32-bit
                return False, "Unsupported WAV format (only 8-bit, 16-bit, or 32-bit supported)"
            
            return True, f"Valid WAV file: {n_channels} channels, {framerate}Hz, {duration_seconds:.2f}s"
    except Exception as e:
        return False, f"Invalid WAV file: {str(e)}"

def get_file_extension(filename):
    """Get the file extension in lowercase"""
    return os.path.splitext(filename.lower())[1]

def is_supported_audio_format(filename):
    """Check if the file format is supported"""
    supported_formats = ['.wav', '.mp3']  # Limiting to formats we can reliably handle
    extension = get_file_extension(filename)
    return extension in supported_formats
