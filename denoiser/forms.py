# denoiser/forms.py

from django import forms
import os
import warnings
from .audio_utils import is_supported_audio_format

# Check if pydub is available for multi-format support
try:
    # Suppress pydub FFmpeg warnings since we primarily use WAV files
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
        warnings.filterwarnings("ignore", message="Couldn't find ffprobe or avprobe")
        from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class AudioUploadForm(forms.Form):
    PROCESSING_CHOICES = [
        ('voice_preserving', 'Voice-Preserving (Best voice quality)'),
        ('standard', 'Standard Denoising (Balanced)'),
        ('aggressive', 'Aggressive Denoising (Max noise reduction)'),
        ('waveunet_only', 'WaveUNet Only (Minimal processing)'),
    ]
    
    file = forms.FileField(
        label="Upload a noisy audio file",
        help_text="WAV format only (16-bit recommended). For MP3 files, please convert to WAV first using online converters like cloudconvert.com. Max file size: 50MB, Duration: 0.5s - 5min",
        widget=forms.FileInput(attrs={
            'accept': '.wav',
            'class': 'block w-full text-sm text-gray-300 bg-slate-700 border border-slate-600 rounded-md cursor-pointer focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent p-3'
        })
    )
    

    
    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        
        if uploaded_file is None:
            raise forms.ValidationError("Please select a file to upload.")
        
        # Check file extension - only WAV for now
        file_name = uploaded_file.name.lower()
        if not file_name.endswith('.wav'):
            raise forms.ValidationError(
                "Only WAV files are supported. "
                "For MP3 files, please convert them to WAV format first using online converters like "
                "https://cloudconvert.com/mp3-to-wav"
            )
        
        # Check file size (50MB limit)
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB in bytes
            raise forms.ValidationError("File size must be less than 50MB.")
        
        # Check if file is empty
        if uploaded_file.size == 0:
            raise forms.ValidationError("The uploaded file is empty.")
        
        return uploaded_file
