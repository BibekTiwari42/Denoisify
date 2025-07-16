# denoiser/forms.py

from django import forms

class AudioUploadForm(forms.Form):
    PROCESSING_CHOICES = [
        ('waveunet_ssbse', 'WaveUNet + SSBSE (Recommended)'),
        ('ssbse_only', 'SSBSE Only'),
    ]
    
    file = forms.FileField(
        label="Upload a noisy WAV file",
        help_text="Supported formats: WAV, MP3. Max file size: 50MB"
    )
    
    processing_method = forms.ChoiceField(
        choices=PROCESSING_CHOICES,
        initial='waveunet_ssbse',
        label="Processing Method",
        help_text="Choose the denoising method"
    )
